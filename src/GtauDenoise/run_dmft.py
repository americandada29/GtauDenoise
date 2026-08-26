from triqs_dft_tools.sumk_dft import *
from triqs.gf import *
from h5 import HDFArchive
from triqs.operators.util import *
from triqs_cthyb import *
import triqs.utility.mpi as mpi

from triqs.gf import gf_fnt
import importlib.util
from triqs.gf.dlr_crm_dyson_solver import minimize_dyson
from triqs.operators import *
from h5 import *
from triqs_cthyb import Solver
from triqs_cthyb.tail_fit import tail_fit as cthyb_tail_fit
# from w2dyn_cthyb import Solver
import numpy as np
from triqs.atom_diag import trace_rho_op
from triqs.operators import c, c_dag, n
import triqs.utility.mpi as mpi
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import torch
from utils import *
# from models import *
from models_chat import *

from hankel_utils import hankel_projection, HankelCaches
from scipy.integrate import simpson
np.set_printoptions(threshold=np.inf)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
set_utils_device(device)


def make_low_kernel(n_tau, k_low):
    eye = torch.eye(n_tau, device=device, dtype=torch.float32)
    return make_gw_from_gtau_integrate(eye)[:, :k_low].T.contiguous().to(torch.complex64)

def second_derivative(x):
    return torch.gradient(torch.gradient(x, dim=1)[0], dim=1)[0]

def adaptive_grid_hankel_project(Gtaus, densities, max_iter=3, tol=1e-7, verbose=False):
    new_ntau = Gtaus.shape[1]
    taus_org = get_taus().to(device)
    taus_adapted = torch.linspace(0.0, get_beta(), new_ntau, device=device)
    cache = HankelCaches(new_ntau, device)
    Gtaus_adapted = interp1d_nonuniform_torch(taus_org, Gtaus, taus_adapted)
    Gtaus_adapted = hankel_projection(Gtaus_adapted, densities, cache, max_iter=max_iter, tol=tol, verbose=verbose)
    return interp1d_nonuniform_torch(taus_adapted, Gtaus_adapted, taus_org)

def interp1d_nonuniform_torch(x, y, x_new):
    x_new = x_new.clamp(x[0], x[-1])
    idx_hi = torch.searchsorted(x, x_new, right=False).clamp(1, x.numel() - 1)
    idx_lo = idx_hi - 1
    w = (x_new - x[idx_lo]) / (x[idx_hi] - x[idx_lo])
    y0 = torch.gather(y, -1, idx_lo.expand(*y.shape[:-1], -1))
    y1 = torch.gather(y, -1, idx_hi.expand(*y.shape[:-1], -1))
    return y0 * (1.0 - w) + y1 * w

def interp1d_nonuniform(x, y, x_new):
    x_new = np.clip(x_new, x[0], x[-1])
    idx_hi = np.clip(np.searchsorted(x, x_new, side="left"), 1, x.size - 1)
    idx_lo = idx_hi - 1
    w = (x_new - x[idx_lo]) / (x[idx_hi] - x[idx_lo])
    y0 = np.take(y, idx_lo, axis=-1)
    y1 = np.take(y, idx_hi, axis=-1)
    return y0 * (1.0 - w) + y1 * w

def get_gtau_from_giw(iws, giw, beta, ntau, density):
    gtau = np.zeros(ntau, dtype=np.complex128)
    taus = np.linspace(0, beta, ntau)
    for i in range(ntau):
        if i==0:
            gtau[i] = density-1.0
        elif i==ntau-1:
            gtau[i] = -density
        else:
            gtau[i] = np.sum(giw/np.exp(iws*taus[i]))/beta
    return taus, gtau

def get_giw_from_gtau(taus, gtau, niw):
    beta = taus[-1]
    iws = (2*np.arange(0, niw)+1)*np.pi/beta
    giw = np.zeros(niw, dtype=np.complex128)
    for n in range(niw):
        giw[n] = simpson(gtau*np.exp(1j*iws[n]*taus), taus)
    full_giw = np.zeros(2*niw, dtype=np.complex128)
    full_giw[niw:] = giw 
    full_giw[:niw] = np.conj(giw[::-1])
    return full_giw

@torch.no_grad()
def estimate_gtau_noise_timestep(gtaus, density, sigmas, device, projection_iters=30, residual_window=41, edge_drop=8, safety=1.0):
    x = torch.as_tensor(gtaus, device=device).real.float().reshape(1, -1)
    dens = torch.as_tensor(density, device=device).real.float().reshape(1)
    schedule = sigmas.detach().to(device=device, dtype=torch.float32).flatten()
    x = x.clone()
    x[:, 0] = dens - 1.0
    x[:, -1] = -dens
    x_clean = adaptive_grid_hankel_project(x, dens, max_iter=projection_iters, tol=1e-7, verbose=False)
    x_clean[:, 0] = dens - 1.0
    x_clean[:, -1] = -dens
    curv = torch.gradient(torch.gradient(x_clean, dim=1)[0], dim=1)[0]
    curv = curv / curv.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    env = 0.02 + 0.005 * curv.abs()
    r = x - x_clean
    radius = residual_window // 2
    kernel = torch.ones(1, 1, residual_window, device=device, dtype=torch.float32) / residual_window
    r_lp = torch.nn.functional.conv1d(torch.nn.functional.pad(r[:, None, :], (radius, radius), mode="reflect"), kernel).squeeze(1)
    z = ((r - r_lp) / env).abs()[:, edge_drop:-edge_drop].flatten()
    sigma_hat = safety * torch.median(z) / (0.6744897501960817 * (1.0 - 1.0 / residual_window) ** 0.5)
    sigma_hat = sigma_hat.clamp(schedule.min(), schedule.max())
    timestep = torch.argmin((schedule - sigma_hat).abs()).reshape(1).long()
    return timestep, sigma_hat

def make_training_noise(model, gtaus, timesteps, K_LOW):
    curv = second_derivative(gtaus)
    curv = curv / curv.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    gtaus_t = gtaus + model.sigmas[timesteps][:, None] * torch.randn_like(gtaus) * (0.0125 + 0.0175 * curv.abs())
    z = (torch.randn(gtaus.shape[0], K_LOW, device=device, dtype=torch.float32) + 1j * torch.randn(gtaus.shape[0], K_LOW, device=device, dtype=torch.float32)) / math.sqrt(2.0)
    giw_low_target = model.low_transform(gtaus_t) + (0.03 + 0.35 * model.sigmas[timesteps])[:, None].float() * model.giw_scale[None, :].to(torch.float32) * z
    return model.project_to_constraints(gtaus_t, giw_low_target, gtaus[:, -1]).detach()

def upsample_gtau(taus, gtau, tau_desired):
    beta = taus[-1]
    new_gtau = np.interp(tau_desired, taus, gtau)
    return tau_desired, new_gtau

def make_gtau_from_Aw_integrate_with_taus(taus, beta, ws, Aw):
    FKernel = 1/(np.exp(taus[:,None]*ws[None,:]) + np.exp((taus[:,None]-beta)*ws[None, :]))
    Integrand = -1*FKernel[:, :]*Aw[None, :]
    return np.trapz(Integrand, ws, axis=-1)

def make_gw_from_Aw_integrate_with_iws(iws, beta, ws, Aw):
    kernel = 1/(1j*iws[:,None] - ws[None, :])
    Giw = np.trapz(kernel[:, :]*Aw[None, :], ws, axis=-1).astype(np.complex128)
    return Giw


def gen_clean_giw(density, DenoiseModel, Giw, Gtau, G_moments, Siw, G0iw, S_moments, niw_fit_start=50, niw_fit_end=100, hankel_project=False, plot=False, orb="odd"):
    ## Just rebin G(tau) since rebinned G(tau) is usually always below zero anyway. Might introduce eror though
    # Gtau_binned = rebinning_tau(Gtau, DenoiseModel.ntau)

    ## Need to now project Gtau onto adaptive grid for denoising
    Gtau_triqs = Gtau.data[:,0,0]
    taus_triqs = np.array([float(t) for t in Gtau.mesh])

    Giw_triqs = Giw.data[:,0,0]
    Siw_triqs = Siw.data[:,0,0]
    G0iw_triqs = G0iw.data[:,0,0]
    iws_triqs = np.array([w.imag for w in Giw.mesh])

    Gtau_rebin = rebinning_tau(Gtau, len(DenoiseModel.taus))
    taus_rebin = np.array([float(t) for t in Gtau_rebin.mesh])
    # Gtau_proj = interp1d_nonuniform(torch.from_numpy(taus_rebin), torch.from_numpy(Gtau_rebin.data[:,0,0]), DenoiseModel.taus)
    Gtau_proj = interp1d_nonuniform(taus_rebin, Gtau_rebin.data[:,0,0], DenoiseModel.taus.cpu().numpy())

    # Gtau_data, Giw_data, Aw_data = DenoiseModel.denoise_gtau(Gtau_proj.numpy(), Giw_triqs, Siw_triqs, G0iw_triqs, density, G_moments, 
    #                                                          S_moments, hankel_project=hankel_project, skip_model=False)
    Gtau_data, Giw_data, Aw_data = DenoiseModel.denoise_gtau(Gtau_proj, Giw_triqs, Siw_triqs, G0iw_triqs, density, G_moments, 
                                                             S_moments, hankel_project=hankel_project, skip_model=False)


    ### Since we have Gtau and Giw from spectral function, just integrate that
    # Gtau_full_data = make_gtau_from_Aw_integrate_with_taus(torch.from_numpy(taus_triqs), DenoiseModel.beta, DenoiseModel.ws, torch.from_numpy(Aw_data))
    # Giw_full_data = make_gw_from_Aw_integrate_with_iws(torch.from_numpy(iws_triqs), DenoiseModel.beta, DenoiseModel.ws, torch.from_numpy(Aw_data))
    Gtau_full_data = make_gtau_from_Aw_integrate_with_taus(taus_triqs, DenoiseModel.beta, DenoiseModel.ws.cpu().numpy(), Aw_data)
    Giw_full_data = make_gw_from_Aw_integrate_with_iws(iws_triqs, DenoiseModel.beta, DenoiseModel.ws.cpu().numpy(), Aw_data)
    Gtau_tmp = GfImTime(mesh=MeshImTime(beta=DenoiseModel.beta, n_tau=len(Gtau_full_data), statistic='Fermion'), data=Gtau_full_data.reshape(-1,1,1).astype(np.complex128))
    Giw_manual = GfImFreq(mesh=MeshImFreq(beta=DenoiseModel.beta, n_iw=len(Giw_full_data)//2, statistic='Fermion'), data=Giw_full_data.reshape(-1,1,1).astype(np.complex128))
    ##########################################################################

    if plot:
        plt.plot(DenoiseModel.ws.detach().cpu().numpy(), Aw_data)
        plt.xlim(-4, 10)
        plt.show()

    if plot:
        # A1 = HDFArchive(f"../../VB_HUBBARD_test_2/scan_doping_ref/data/results-{mu:.3f}.h5", "r")
        # Giw_ref = A1['G_iw-9'][f'{orb}-up']
        # iws_ref = np.array([complex(w) for w in Giw_ref.mesh])

        fig, axs = plt.subplots(2)
        iws = np.array([w.imag for w in Giw_manual.mesh])
        axs[0].plot(iws_triqs, Giw_manual.data[:,0,0].real)
        axs[0].plot(iws_triqs, Giw_triqs.real)
        # axs[0].plot(iws_ref.imag, Giw_ref.data[:,0,0].real, linestyle="--",c="black", zorder=20)

        axs[1].plot(iws_triqs, Giw_manual.data[:,0,0].imag, label="triqs transformed", marker="o", markersize=1, zorder=11)
        axs[1].plot(iws_triqs, Giw_triqs.imag, marker="x", label="org",zorder=0)
        axs[1].plot(DenoiseModel.iws.cpu().numpy(), Giw_data.imag, marker="o", label="denoised", zorder=10)
        # axs[1].plot(iws_ref.imag, Giw_ref.data[:,0,0].imag, linestyle="--",c="black", zorder=20)
        axs[1].axvline(x=0, linestyle="--", c='black')
        axs[1].set_xlim(-0.1, 10.0)
        # axs[1].set_ylim(-2.0, 0.1)
        axs[1].legend()
        plt.title(f"{orb}")
        plt.show()
        # exit()
    
    ### Plotting Gtau for debugging ####
    if plot:
        plt.plot(DenoiseModel.taus.cpu().numpy(), Gtau_data, marker="o", label="denoised", zorder=10)
        plt.plot(taus_triqs, Gtau_tmp.data[:,0,0], label="denoised-triqs", zorder=10)
        plt.plot(taus_triqs, Gtau_triqs, marker="x", label="triqs", zorder=5)
        # plt.plot(taus_rebin, Gtau_rebin.data[:,0,0], linestyle="--", label="rebinned", zorder=20)
        # plt.plot(DenoiseModel.taus.numpy(), Gtau_proj.numpy(), linestyle="--", label="proj", zorder=20)
        plt.legend()
        plt.show()
    ##############################
    
    return Gtau_tmp, Giw_manual, Aw_data

def calc_conv(deltai, deltaip1):
    # return np.sqrt(np.sum(np.abs(deltai.data[:,0,0]-deltaip1.data[:,0,0])**2)/np.sum(np.abs(deltaip1.data[:,0,0])**2))
    meas_start = deltai.data.shape[0]//2
    meas_len = 100
    # weights = np.flip(np.arange(0, meas_len))
    return np.sqrt(np.sum(np.abs(deltai.data[meas_start:meas_start+meas_len,0,0]-deltaip1.data[meas_start:meas_start+meas_len,0,0])**2))


class HankelDenoise():
    def __init__(self, device, beta, nt=100, ntau=2**10+1):
        nt = 200

        ### Custom model ###
        # self.model = FNOHankelNet(ntau=ntau, nt=nt, device=device, temb=32, cond_hidden=128, modes=32, num_fourier_layers=6).to(device)
        # self.model.load_state_dict(torch.load("../../DDPM_14_ADJGRID/saved_models/epsnet.pth"))
        # self.model.eval()
        # self.sigmas = torch.linspace(1e-2, 0.03, nt, device=device)
        # set_beta(beta, ntau)
        #################

        ### Older modified model ###
        ws_max = 10
        N_ws = 1000
        K_LOW = 16
        self.K_LOW = K_LOW
        K_MID = 64
        batch_size = 4
        n_batches = 100
        sigmas = torch.linspace(1e-2, 1.0, nt, device=device)
        set_beta(beta, ntau)
        set_ws(N_ws, ws_max, meshform="linear")
        taus = get_taus().to(device)
        iws = get_wn().to(device)
        data = torch.load("../../DDPM_14_CHAT/Data/train_test_data.pt", map_location=torch.device(device))
        giws_train = data["giws_train"].to(device=device, dtype=torch.complex64)
        self.gtaus_train = data["gtaus_train"].to(device=device, dtype=torch.float32)
        kernel_low = make_low_kernel(ntau, K_LOW)
        giw_scale_low = giws_train[:, :K_LOW].abs().std(dim=0).float().clamp_min(1e-4)
        self.model = GiwProjectedDenoiser(ntau=ntau, nt=nt, kernel_low=kernel_low, sigmas=sigmas, giw_scale=giw_scale_low, n_ws=N_ws, width=96, modes=16, depth=6, emb_dim=32, cond_dim=24).to(device)
        self.model.load_state_dict(torch.load("../../DDPM_14_CHAT/saved_models/epsnet.pth", map_location=torch.device(device)))
        self.model.eval()
        ###########################


        ### Chat modifications to my model ###
        # ws_max = 10
        # N_ws = 1001
        # K_LOW = 16
        # self.K_LOW = K_LOW
        # K_MID = 64
        # batch_size = 4
        # n_batches = 100
        # sigmas = torch.linspace(1e-2, 1.0, nt, device=device)
        # set_utils_device(device)
        # set_beta(beta, ntau)
        # set_ws(N_ws, ws_max, meshform="hyperbolic")
        # taus = get_taus().to(device)
        # iws = get_wn().to(device)
        # data = torch.load("../../DDPM_20_ADJGRID/Data/train_test_data.pt", map_location=torch.device(device))
        # giws_train = data["giws_train"].to(device=device, dtype=torch.complex64)
        # self.gtaus_train = data["gtaus_train"].to(device=device, dtype=torch.float32)
        # kernel_low = make_low_kernel(ntau, K_LOW)
        # giw_scale_low = giws_train[:, :K_LOW].abs().std(dim=0).float().clamp_min(1e-4)
        # self.model = GiwProjectedDenoiser(ntau=ntau, nt=nt, kernel_low=kernel_low, sigmas=sigmas, giw_scale=giw_scale_low, n_ws=N_ws, width=96, modes=16, depth=6, emb_dim=48, cond_dim=32).to(device)
        # self.model.load_state_dict(torch.load("../../DDPM_20_ADJGRID/saved_models/epsnet_amarel.pth", map_location=torch.device(device)))
        # self.model.eval()
        ######################################


        self.ntau = ntau
        self.nt = nt
        self.device = device
        self.beta = beta
        self.taus = taus
        self.iws = iws
        self.ws = get_ws()
     
    @torch.no_grad()
    def denoise_gtau(self, gtaus, giws, siws, g0iws, density, giws_moments, siws_moments, hankel_project=False, skip_model=False):
        gtaus_tm1 = torch.from_numpy(gtaus).to(self.device).to(torch.float32).unsqueeze(0)
        giws_triqs = torch.from_numpy(giws).to(self.device).to(torch.complex64).unsqueeze(0)[:,giws.shape[0]//2:]
        siws_tm1 = torch.from_numpy(siws).to(self.device).to(torch.complex64).unsqueeze(0)[:,giws.shape[0]//2:]
        g0iws_tm1 = torch.from_numpy(g0iws).to(self.device).to(torch.complex64).unsqueeze(0)[:,giws.shape[0]//2:]
        tdens = torch.tensor(density).unsqueeze(0).to(device)
        g_moments = torch.from_numpy(giws_moments[:,0,0]).to(torch.float64).to(self.device)
        s_moments = torch.from_numpy(siws_moments[:,0,0]).to(torch.complex128).to(self.device)
        # if not(skip_model):
        #     timesteps, sigma_hat = estimate_gtau_noise_timestep(gtaus_tm1, tdens, self.model.sigmas, self.device)
        #     gtaus_tm1, aux = self.model.forward_with_aux(gtaus_tm1, timesteps, -tdens)
        #     giw_target = aux["giw_low_pred"].clone()
        #     hankel_project = False
            ######################################
            
             
        #### Set up optimization loop for self energy matching ####
        timesteps, sigma_hat = estimate_gtau_noise_timestep(gtaus_tm1, tdens, self.model.sigmas, self.device)
        _, aux = self.model.forward_with_aux(gtaus_tm1, timesteps, -tdens)
        aw_base = aux['aw_pred']

        #region: Adam only optimization
        # with torch.enable_grad():
        #     aw_raw = torch.nn.Parameter(torch.log(torch.expm1(aw_base)))
        #     optimizer = torch.optim.Adam([aw_raw], lr=1e-1)
        #     sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
        #     n_fit = 300
        #     weights = 300.0/(1.0 + torch.arange(n_fit, device=self.device, dtype=torch.float32)/100)
        #     # weights = torch.arange(n_fit, device=self.device, dtype=torch.float32)
        #     # weights = 100*torch.ones(n_fit, device=self.device, dtype=torch.float32)
        #     fd_dist = torch.sigmoid(-self.beta*self.ws).to(self.device)
        #     for i in range(1000):
        #         optimizer.zero_grad()
        #         aw_opt = torch.nn.functional.softplus(aw_raw)
        #         aw_opt = aw_opt/torch.trapezoid(aw_opt, self.ws.to(aw_opt), dim=-1).unsqueeze(-1).clamp_min(1e-8)
        #         giws_opt = make_gw_from_Aw_integrate(aw_opt)
        #         gtau_opt = make_gtau_from_Aw_integrate(aw_opt)
        #         siws_opt = 1/g0iws_tm1 - 1/giws_opt 
        #         dyson_residual1 = giws_triqs[:,:n_fit] - g0iws_tm1[:,:n_fit]*siws_opt[:,:n_fit]*giws_triqs[:,:n_fit] - g0iws_tm1[:,:n_fit]
        #         dyson_residual2 = giws_opt[:,:n_fit] - g0iws_tm1[:,:n_fit]*siws_tm1[:,:n_fit]*giws_opt[:,:n_fit] - g0iws_tm1[:,:n_fit]
        #         dyson_residual3 = (torch.reciprocal(g0iws_tm1[:,:n_fit]) - siws_tm1[:,:n_fit])*giws_opt[:,:n_fit] - 1.0
        #         calc_dens = torch.trapz(aw_opt*fd_dist, self.ws, dim=-1)
        #         gmom2 = torch.trapz(aw_opt*self.ws, self.ws, dim=-1)
        #         sigma0_opt = (siws_opt[:,-100:].real).mean(dim=-1)
        #         # sigma1_opt = (-self.iws[-100:].to(self.device)*siws_opt[:,-100:].imag).mean(dim=-1)
        #         sigma1_opt = (-self.iws[siws_opt.shape[1]-300:].to(self.device)*siws_opt[:,siws_opt.shape[1]-300:].imag).mean(dim=-1)
                
        #         ##### Designing loss, can include self energy, density and self energy moments
        #         loss_sigma_1 = (weights*dyson_residual1.abs().square()).mean()
        #         loss_sigma_2 = (weights*dyson_residual2.abs().square()).mean()
        #         loss_sigma_3 = (weights*dyson_residual3.abs().square()).mean()
        #         loss_sigma = loss_sigma_3 #+ loss_sigma_1 #+ loss_sigma_2

        #         loss_low = (giws_opt[:,:2] - giws_triqs[:,:2]).abs().square().mean()
        #         loss_aw = (aw_opt - aw_base).square().mean()
        #         loss_dens = (calc_dens-tdens).abs()
        #         loss_gmom2 = (gmom2 - g_moments[2]).abs()
        #         loss_smom0 = (sigma0_opt - s_moments[0].real).abs()
        #         loss_smom1 = (sigma1_opt - s_moments[1].real).abs()
        #         loss_gtau = (gtaus_tm1 - gtau_opt).abs().square().mean()

        #         loss = loss_sigma + loss_smom0 + loss_smom1 + loss_dens + 1000*loss_gtau
        #         ######################################################################

        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_([aw_raw], 1.0)
        #         optimizer.step()
        #         sched.step(loss.item())
        #         if optimizer.param_groups[0]["lr"] < 1e-7:
        #             break
        #         # if i % 25 == 0: print(i, loss.item(), loss_sigma.item(), loss_low.item(), optimizer.param_groups[0]["lr"])
        #         if i % 25 == 0: print(f'{i}, {loss.item():.2e}, sig: {loss_sigma.item():.2e}, giw_low: {loss_low.item():.2e}, dens: {loss_dens.item():.2e}, gmom2: {loss_gmom2.item():.2e} smom[0,1]: {loss_smom0.item():.2e},{loss_smom1.item():.2e}, gtau: {loss_gtau.item():.2e} lr: {optimizer.param_groups[0]["lr"]}')

        # aw_opt = torch.nn.functional.softplus(aw_raw.detach())
        # aw_opt = aw_opt/torch.trapezoid(aw_opt, self.ws.to(aw_opt), dim=-1).unsqueeze(-1).clamp_min(1e-8)
        # gtaus_tm1 = make_gtau_from_Aw_integrate(aw_opt)
        # giws_tm1 = make_gw_from_Aw_integrate(aw_opt)
        #endregion

        # region LBFGS optimization
        ### New Adam + LBFGS optimization
        with torch.enable_grad():
            aw_raw = torch.nn.Parameter(torch.log(torch.expm1(aw_base)))
            n_fit = 200
            weights = 300.0/(1.0 + torch.arange(n_fit, device=self.device, dtype=torch.float32)/100)
            # weights = torch.arange(n_fit, device=self.device, dtype=torch.float32)
            # weights = 100*torch.ones(n_fit, device=self.device, dtype=torch.float32)
            fd_dist = torch.sigmoid(-self.beta*self.ws).to(self.device)

            def calculate_loss():
                aw_opt = torch.nn.functional.softplus(aw_raw)
                aw_opt = aw_opt/torch.trapezoid(aw_opt, self.ws.to(aw_opt), dim=-1).unsqueeze(-1).clamp_min(1e-8)
                giws_opt = make_gw_from_Aw_integrate(aw_opt)
                gtau_opt = make_gtau_from_Aw_integrate(aw_opt)
                siws_opt = 1/g0iws_tm1 - 1/giws_opt
                # dyson_residual = (torch.reciprocal(g0iws_tm1[:,:n_fit]) - siws_tm1[:,:n_fit])*giws_opt[:,:n_fit] - 1.0
                # dyson_residual = giws_triqs[:,:n_fit] - g0iws_tm1[:,:n_fit]*siws_opt[:,:n_fit]*giws_triqs[:,:n_fit] - g0iws_tm1[:,:n_fit]
                dyson_residual = giws_opt[:,:n_fit] - g0iws_tm1[:,:n_fit]*siws_tm1[:,:n_fit]*giws_opt[:,:n_fit] - g0iws_tm1[:,:n_fit]
                calc_dens = torch.trapz(aw_opt*fd_dist, self.ws, dim=-1)
                gmom2 = torch.trapz(aw_opt*self.ws, self.ws, dim=-1)
                sigma0_opt = siws_opt[:,-100:].real.mean(dim=-1)
                # sigma1_opt = (-self.iws[-100:].to(self.device)*siws_opt[:,-100:].imag).mean(dim=-1)
                sigma1_opt = (-self.iws[siws_opt.shape[1]-300:].to(self.device)*siws_opt[:,siws_opt.shape[1]-300:].imag).mean(dim=-1)

                loss_sigma_re = (weights*dyson_residual.real.square()).mean()
                loss_sigma_im = (weights*dyson_residual.imag.square()).mean()
                loss_sigma = loss_sigma_re + loss_sigma_im

                loss_low = (giws_opt[:,:2] - giws_triqs[:,:2]).abs().square().mean()
                loss_aw = (aw_opt - aw_base).square().mean()
                loss_dens = (calc_dens-tdens).square().mean()
                loss_gmom2 = (gmom2 - g_moments[2]).square().mean()
                loss_smom0 = (sigma0_opt - s_moments[0].real).square().mean()
                loss_smom1 = (sigma1_opt - s_moments[1].real).square().mean()
                loss_gtau = (gtaus_tm1 - gtau_opt).abs().square().mean()

                # if i<300:
                #     loss = loss_sigma
                # else:
                #     loss = loss_sigma + 0.001*(loss_smom1 + loss_gmom2)
                # loss = loss_sigma + loss_smom0 + loss_smom1 + loss_dens + loss_gmom2 + loss_aw/100
                loss = loss_sigma + loss_smom0 + loss_smom1 + loss_dens

                return loss, loss_sigma_re, loss_sigma_im, loss_low, loss_dens, loss_gmom2, loss_smom0, loss_smom1, loss_gtau

            optimizer = torch.optim.Adam([aw_raw], lr=1e-1)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
            for i in range(200):
                optimizer.zero_grad()
                loss, loss_sigma_re, loss_sigma_im, loss_low, loss_dens, loss_gmom2, loss_smom0, loss_smom1, loss_gtau = calculate_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([aw_raw], 1.0)
                optimizer.step()
                sched.step(loss.item())
                if i % 25 == 0:
                    print(f'{i}, {loss.item()}, sig: {loss_sigma_re.item()},{loss_sigma_im.item()}, giw_low: {loss_low.item()}, dens: {loss_dens.item()}, gmom2: {loss_gmom2.item()} smom[0,1]: {loss_smom0.item()},{loss_smom1.item()}, lr: {optimizer.param_groups[0]["lr"]}')
            optimizer = torch.optim.LBFGS([aw_raw], lr=0.1, max_iter=2000, history_size=100, line_search_fn="strong_wolfe")
            lbfgs_eval = [0]

            def closure():
                optimizer.zero_grad()
                loss, loss_sigma_re, loss_sigma_im, loss_low, loss_dens, loss_gmom2, loss_smom0, loss_smom1, loss_gtau = calculate_loss()
                loss.backward()
                if lbfgs_eval[0] % 25 == 0:
                    i = 200 + lbfgs_eval[0]
                    print(f'{i}, {loss.item():.2e}, sig: {loss_sigma_re.item():.2e},{loss_sigma_im.item():.2e}, giw_low: {loss_low.item():.2e}, dens: {loss_dens.item():.2e}, gmom2: {loss_gmom2.item():.2e} smom[0,1]: {loss_smom0.item():.2e},{loss_smom1.item():.2e}, gtau: {loss_gtau.item():.2e} lr: {optimizer.param_groups[0]["lr"]}')
                lbfgs_eval[0] += 1
                return loss
            optimizer.step(closure)

        aw_opt = torch.nn.functional.softplus(aw_raw.detach())
        aw_opt = aw_opt/torch.trapezoid(aw_opt, self.ws.to(aw_opt), dim=-1).unsqueeze(-1).clamp_min(1e-8)
        gtaus_tm1 = make_gtau_from_Aw_integrate(aw_opt)
        giws_tm1 = make_gw_from_Aw_integrate(aw_opt)
        #endregion
        # ###################################################################

        
        gtaus_tm1 = gtaus_tm1.squeeze(0).cpu().numpy().astype(np.complex128)
        giws_tm1 = giws_tm1.squeeze(0).cpu().numpy().astype(np.complex128)    
        aw_tm1 = aw_opt.squeeze(0).cpu().numpy().astype(np.float64)      
        return gtaus_tm1, giws_tm1, aw_tm1



dft_filename = 'wannier_files/srvo3'          # filename
U = 4.0                         # interaction parameters
J = 0.6
beta = 200                       # inverse temperature
loops = 15                      # number of DMFT loops
mix = 0.5                       # mixing factor of Sigma after solution of the AIM
dc_type = 1                     # DC type: 0 FLL, 1 Held, 2 AMF
use_blocks = True               # use bloc structure from DFT input
prec_mu = 0.0001                # precision of chemical potential

## Define mesh since Triqs decided to update itself
niw = 1025
mesh = MeshImFreq(beta=beta, S="Fermion", n_iw = niw)

### Set up diffusion model
if mpi.is_master_node():
    HD = HankelDenoise(device, beta=beta, ntau=2**10+1)

### SumkDFT stuff
SK = SumkDFT(hdf_file=dft_filename+'.h5',use_dft_blocks=use_blocks, mesh=mesh, beta=beta)
# Use GF structure determined by DFT blocks:
gf_struct = SK.gf_struct_solver_list[0]

# print("density_required =", SK.density_required)
# print("charge_below =", SK.charge_below)
# print("density in projection window =", SK.density_required - SK.charge_below)
# print("DFT density =", SK.total_density(with_Sigma=False))
# exit()

Sigma_iw = BlockGf(mesh=mesh, gf_struct=gf_struct)
Sigma_iw = mpi.bcast(Sigma_iw)
SK.dc_imp = mpi.bcast(SK.dc_imp)
SK.dc_energ = mpi.bcast(SK.dc_energ)
SK.chemical_potential = mpi.bcast(SK.chemical_potential)
SK.put_Sigma(Sigma_imp=[Sigma_iw])




p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 200
p["n_warmup_cycles"] = int(1e3)
p["n_cycles"] = int(1e4)

# tail fit
p["perform_tail_fit"] = False
# p["fit_max_moment"] = 4
# p["fit_min_n"] = 30
# p["fit_max_n"] = 60

# measure impurity density matrix to get self-energy moments for improved tail fit
p["measure_density_matrix"] = True
p["use_norm_as_weight"] = True

n_orb = SK.corr_shells[0]['dim']
spin_names = ["up","down"]


dmft_output = f"dmft_output/results_{int(p['n_cycles']*mpi.size/int(10**int(np.log10(p['n_cycles']*mpi.size))))}e{int(np.log10(p['n_cycles']*mpi.size))}_{p['length_cycle']}.h5"



# Construct U matrix for density-density calculations:
Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)

h_int = h_int_density(spin_names, n_orb, map_operator_structure=SK.sumk_to_solver[0], U=Umat, Uprime=Upmat)
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=niw)

if mpi.is_master_node():
    with HDFArchive(dmft_output) as ar:
        if (not ar.is_group('dmft_output')):
            ar.create_group('dmft_output')

for iteration_number in range(1,loops+1):
    if mpi.is_master_node(): print("Iteration = ", iteration_number)

    SK.symm_deg_gf(S.Sigma_iw,ish=0)                        # symmetrizing Sigma
    SK.set_Sigma([ S.Sigma_iw ])                            # put Sigma into the SumK class
    chemical_potential = SK.calc_mu( precision = prec_mu )  # find the chemical potential for given density
    S.G_iw << SK.extract_G_loc()[0]                         # calc the local Green function
    mpi.report("Total charge of Gloc : %.6f"%S.G_iw.total_density().real)

    # In the first loop, init the DC term and the real part of Sigma:
    if (iteration_number==1):
        dm = S.G_iw.density()
        SK.calc_dc(dm, U_interact = U, J_hund = J, orb = 0, use_dc_formula = dc_type)
        S.Sigma_iw << SK.dc_imp[0]['up'][0,0]

    # Calculate new G0_iw to input into the solver:
    S.G0_iw << S.Sigma_iw + inverse(S.G_iw)
    S.G0_iw << inverse(S.G0_iw)

    # Solve the impurity problem:
    Sigma_old = S.Sigma_iw.copy()
    S.solve(h_int=h_int, **p)


    # Solved. Now do post-solution stuff:
    mpi.report("Total charge of impurity problem : %.6f"%S.G_iw.total_density().real)

    if mpi.is_master_node():
        # with HDFArchive(dft_filename+'.h5','r') as ar:
        #     mpi.report("Mixing Sigma and G with factor %s"%mix)
        #     S.Sigma_iw << mix * S.Sigma_iw + (1.0-mix) * ar['dmft_output']['Sigma_iw']
        #     S.G_iw << mix * S.G_iw + (1.0-mix) * ar['dmft_output']['G_iw']

        with HDFArchive(dmft_output) as ar:
              ar['dmft_output'][f'G0-{iteration_number}'] = S.G0_iw
              ar['dmft_output'][f'Gtau_org-{iteration_number}'] = S.G_tau
              ar['dmft_output'][f'Giw_org-{iteration_number}'] = S.G_iw
              ar['dmft_output'][f'Siw_org-{iteration_number}'] = S.Sigma_iw

        conv_values = {}
        dens_matrix = {}
        Aws = {}

        #region: Denoise all the G(taus) separately 
        # for name, Siw in S.Sigma_iw:
        #     plot = True

        #     density = trace_rho_op(S.density_matrix, n(name,0), S.h_loc_diagonalization)
        #     dens_matrix[name] = density
        #     gtau, g, aw = gen_clean_giw(dens_matrix[name], HD, S.G_iw[name], S.G_tau[name], S.G_moments[name],\
        #                             S.Sigma_iw[name], S.G0_iw[name], S.Sigma_moments[name], niw_fit_start=200, \
        #                                 niw_fit_end=300, hankel_project=True, plot=plot, orb=name)
        #     Sig_calc = inverse(S.G0_iw[name]) - inverse(g)

        #     if plot:
        #         iws = np.array([complex(w).imag for w in Sig_calc.mesh])
        #         plt.plot(iws, Sig_calc.data[:,0,0].imag, c="blue", label="denoised", zorder=10)
        #         plt.plot(iws, S.Sigma_iw[name].data[:,0,0].imag, marker="o", c="green", label="org", zorder=0)
        #         plt.legend()
        #         plt.show()

        #     ### Just take new self energy as solution
        #     Sigma_iw_meas = S.Sigma_iw[name].copy()
        #     S.Sigma_iw[name] << Sig_calc
        #     S.G_iw[name] << g 
        #     S.G_tau[name] << gtau
        #     Aws[name] = aw

            

        #     conv = calc_conv(Sigma_old[name], S.Sigma_iw[name])
        #     conv_values[name] = conv
        #     print(iteration_number, name, " Convergence:", conv, "Density:", dens_matrix[name])
        #endregion

        #region: Denoise only the average G(tau) #############
        names = [name for name,_ in S.Sigma_iw]
        nblocks = len(names)
        density_avg = sum(trace_rho_op(S.density_matrix, n(name,0), S.h_loc_diagonalization) for name in names) / nblocks

        ### Custom averaging ###
        # Giw_avg = S.G_iw[names[0]].copy()
        # Gtau_avg = S.G_tau[names[0]].copy()
        # Siw_avg = S.Sigma_iw[names[0]].copy()
        # G0iw_avg = S.G0_iw[names[0]].copy()
        # Sigma_old_avg = Sigma_old[names[0]].copy()
        # Gmom_avg = S.G_moments[names[0]].copy()
        # Smom_avg = S.Sigma_moments[names[0]].copy()
        # for name in names[1:]:
        #     Giw_avg += S.G_iw[name]
        #     Gtau_avg += S.G_tau[name]
        #     Siw_avg += S.Sigma_iw[name]
        #     G0iw_avg += S.G0_iw[name]
        #     Sigma_old_avg += Sigma_old[name]
        #     Gmom_avg += S.G_moments[name]
        #     Smom_avg += S.Sigma_moments[name]
        # Giw_avg /= nblocks
        # Gtau_avg /= nblocks
        # Siw_avg /= nblocks
        # G0iw_avg /= nblocks
        # Sigma_old_avg /= nblocks
        # Gmom_avg /= nblocks
        # Smom_avg /= nblocks
        # print("MADE IT HERE 1")
        #########################


        ### More intelligent averaging ###
        Siw_avg = S.Sigma_iw[names[0]].copy()
        G0iw_avg = S.G0_iw[names[0]].copy()
        Gmom_avg = S.G_moments[names[0]].copy()
        Smom_avg = S.Sigma_moments[names[0]].copy()
        for name in names[1:]:
            Siw_avg += S.Sigma_iw[name]
            G0iw_avg += S.G0_iw[name]
            Gmom_avg += S.G_moments[name]
            Smom_avg += S.Sigma_moments[name]
        Siw_avg /= nblocks
        G0iw_avg /= nblocks
        Gmom_avg /= nblocks
        Smom_avg /= nblocks

        Gtau_avg = S.G_tau[names[0]].copy()
        Giw_avg = inverse(inverse(G0iw_avg)-Siw_avg)
        Gtau_avg << Fourier(Giw_avg, Gmom_avg)
        ##################################


        plot = False
        gtau, g, aw = gen_clean_giw(density_avg, HD, Giw_avg, Gtau_avg, Gmom_avg, Siw_avg, G0iw_avg, Smom_avg, \
                                    niw_fit_start=200, niw_fit_end=300, hankel_project=True, plot=plot, orb="avg")
        Sig_calc = inverse(G0iw_avg) - inverse(g)
        conv = calc_conv(Sigma_old[names[0]], Sig_calc)

        if plot:
            iws = np.array([complex(w).imag for w in Sig_calc.mesh])
            plt.plot(iws, Sig_calc.data[:,0,0].imag, c="blue", label="denoised", zorder=10)
            plt.plot(iws, Siw_avg.data[:,0,0].imag, marker="o", c="green", label="org", zorder=0)
            plt.legend()
            plt.show()

        for name in names:
            dens_matrix[name] = density_avg
            S.G_iw[name] << g
            S.G_tau[name] << gtau
            S.G0_iw[name] << G0iw_avg
            S.Sigma_iw[name] << Sig_calc
            S.G_moments[name] = Gmom_avg.copy()
            S.Sigma_moments[name] = Smom_avg.copy()
            Aws[name] = aw.copy()
            conv_values[name] = conv
        print(iteration_number, "Average Convergence:", conv, "Density:", density_avg)
        #endregion ###############################################################

        with HDFArchive(dmft_output) as ar:
              ar['dmft_output'][f'G0-{iteration_number}'] = S.G0_iw
              ar['dmft_output'][f'Gtau-{iteration_number}'] = S.G_tau
              ar['dmft_output'][f'Giw-{iteration_number}'] = S.G_iw
              ar['dmft_output'][f'Siw-{iteration_number}'] = S.Sigma_iw
              ar['dmft_output'][f'Aws-{iteration_number}'] = Aws
              ar['dmft_output'][f'Densities-{iteration_number}'] = dens_matrix
              ar['dmft_output'][f'Conv-{iteration_number}'] = conv_values
              ar['dmft_output'][f'mu-{iteration_number}'] = SK.chemical_potential

    S.G_tau << mpi.bcast(S.G_tau)
    S.G_iw << mpi.bcast(S.G_iw)
    S.Sigma_iw << mpi.bcast(S.Sigma_iw)

    # Write the final Sigma and G to the hdf5 archive:
    # if mpi.is_master_node():
    #     with HDFArchive(dft_filename+'.h5') as ar:
    #           ar['dmft_output']['iterations'] = iteration_number
    #           ar['dmft_output']['G_0'] = S.G0_iw
    #           ar['dmft_output']['G_tau'] = S.G_tau
    #           ar['dmft_output']['G_iw'] = S.G_iw
    #           ar['dmft_output']['Sigma_iw'] = S.Sigma_iw

    # Set the new double counting:
    # dm = S.G_iw.density() # compute the density matrix of the impurity problem
    dm = {}
    for name, g in S.G_iw:
        dm[name] = np.array([[trace_rho_op(S.density_matrix, n(name,0), S.h_loc_diagonalization)]])

    SK.calc_dc(dm, U_interact = U, J_hund = J, orb = 0, use_dc_formula = dc_type)

    # Save stuff into the user_data group of hdf5 archive in case of rerun:
    SK.save(['chemical_potential','dc_imp','dc_energ'])

