from models import *
from utils import *
from hankel_utils import hankel_projection, HankelCaches
from triqs.gf import *
import numpy as np
import torch
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def make_gtau_from_Aw_integrate_with_taus(taus, beta, ws, Aw):
    FKernel = 1/(np.exp(taus[:,None]*ws[None,:]) + np.exp((taus[:,None]-beta)*ws[None, :]))
    Integrand = -1*FKernel[:, :]*Aw[None, :]
    return np.trapz(Integrand, ws, axis=-1)

def make_gw_from_Aw_integrate_with_iws(iws, beta, ws, Aw):
    kernel = 1/(1j*iws[:,None] - ws[None, :])
    Giw = np.trapz(kernel[:, :]*Aw[None, :], ws, axis=-1).astype(np.complex128)
    return Giw


class HankelDenoise():
    def __init__(self, device, beta, nt=100, ntau=2**10+1, optim="lbfgs"):
        nt = 200
        ws_max = 10
        N_ws = 1000
        K_LOW = 16
        self.K_LOW = K_LOW
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
        self.ntau = ntau
        self.nt = nt
        self.device = device
        self.beta = beta
        self.taus = taus
        self.iws = iws
        self.ws = get_ws()
        self.optim_choice = optim
     
    @torch.no_grad()
    def denoise_gtau(self, gtaus, giws, siws, g0iws, density, giws_moments, siws_moments, siw_fit_range):
        gtaus_tm1 = torch.from_numpy(gtaus).to(self.device).to(torch.float32).unsqueeze(0)
        giws_triqs = torch.from_numpy(giws).to(self.device).to(torch.complex64).unsqueeze(0)[:,giws.shape[0]//2:]
        siws_tm1 = torch.from_numpy(siws).to(self.device).to(torch.complex64).unsqueeze(0)[:,giws.shape[0]//2:]
        g0iws_tm1 = torch.from_numpy(g0iws).to(self.device).to(torch.complex64).unsqueeze(0)[:,giws.shape[0]//2:]
        tdens = torch.tensor(density).unsqueeze(0).to(device)
        g_moments = torch.from_numpy(giws_moments[:,0,0]).to(torch.float64).to(self.device)
        s_moments = torch.from_numpy(siws_moments[:,0,0]).to(torch.complex128).to(self.device)
            
             
        #### Set up optimization loop for self energy matching ####
        timesteps, sigma_hat = estimate_gtau_noise_timestep(gtaus_tm1, tdens, self.model.sigmas, self.device)
        _, aux = self.model.forward_with_aux(gtaus_tm1, timesteps, -tdens)
        aw_base = aux['aw_pred']

        if self.optim_choice=="adam":
            #region: Adam only optimization
            with torch.enable_grad():
                aw_raw = torch.nn.Parameter(torch.log(torch.expm1(aw_base)))
                optimizer = torch.optim.Adam([aw_raw], lr=1e-1)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
                n_fit = siw_fit_range
                weights = 300.0/(1.0 + torch.arange(n_fit, device=self.device, dtype=torch.float32)/100)
                # weights = torch.arange(n_fit, device=self.device, dtype=torch.float32)
                # weights = 100*torch.ones(n_fit, device=self.device, dtype=torch.float32)
                fd_dist = torch.sigmoid(-self.beta*self.ws).to(self.device)
                for i in range(1000):
                    optimizer.zero_grad()
                    aw_opt = torch.nn.functional.softplus(aw_raw)
                    aw_opt = aw_opt/torch.trapezoid(aw_opt, self.ws.to(aw_opt), dim=-1).unsqueeze(-1).clamp_min(1e-8)
                    giws_opt = make_gw_from_Aw_integrate(aw_opt)
                    gtau_opt = make_gtau_from_Aw_integrate(aw_opt)
                    siws_opt = 1/g0iws_tm1 - 1/giws_opt 
                    dyson_residual1 = giws_triqs[:,:n_fit] - g0iws_tm1[:,:n_fit]*siws_opt[:,:n_fit]*giws_triqs[:,:n_fit] - g0iws_tm1[:,:n_fit]
                    dyson_residual2 = giws_opt[:,:n_fit] - g0iws_tm1[:,:n_fit]*siws_tm1[:,:n_fit]*giws_opt[:,:n_fit] - g0iws_tm1[:,:n_fit]
                    dyson_residual3 = (torch.reciprocal(g0iws_tm1[:,:n_fit]) - siws_tm1[:,:n_fit])*giws_opt[:,:n_fit] - 1.0
                    calc_dens = torch.trapz(aw_opt*fd_dist, self.ws, dim=-1)
                    gmom2 = torch.trapz(aw_opt*self.ws, self.ws, dim=-1)
                    sigma0_opt = (siws_opt[:,-100:].real).mean(dim=-1)
                    # sigma1_opt = (-self.iws[-100:].to(self.device)*siws_opt[:,-100:].imag).mean(dim=-1)
                    sigma1_opt = (-self.iws[siws_opt.shape[1]-300:].to(self.device)*siws_opt[:,siws_opt.shape[1]-300:].imag).mean(dim=-1)
                    
                    ##### Designing loss, can include self energy, density and self energy moments
                    loss_sigma_1 = (weights*dyson_residual1.abs().square()).mean()
                    loss_sigma_2 = (weights*dyson_residual2.abs().square()).mean()
                    loss_sigma_3 = (weights*dyson_residual3.abs().square()).mean()
                    loss_sigma = loss_sigma_3 #+ loss_sigma_1 #+ loss_sigma_2

                    loss_low = (giws_opt[:,:2] - giws_triqs[:,:2]).abs().square().mean()
                    loss_aw = (aw_opt - aw_base).square().mean()
                    loss_dens = (calc_dens-tdens).abs()
                    loss_gmom2 = (gmom2 - g_moments[2]).abs()
                    loss_smom0 = (sigma0_opt - s_moments[0].real).abs()
                    loss_smom1 = (sigma1_opt - s_moments[1].real).abs()
                    loss_gtau = (gtaus_tm1 - gtau_opt).abs().square().mean()

                    loss = loss_sigma + loss_smom0 + loss_smom1 + loss_dens + 1000*loss_gtau
                    ######################################################################

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([aw_raw], 1.0)
                    optimizer.step()
                    sched.step(loss.item())
                    if optimizer.param_groups[0]["lr"] < 1e-7:
                        break
                    # if i % 25 == 0: print(i, loss.item(), loss_sigma.item(), loss_low.item(), optimizer.param_groups[0]["lr"])
                    if i % 25 == 0: print(f'{i}, {loss.item():.2e}, sig: {loss_sigma.item():.2e}, giw_low: {loss_low.item():.2e}, dens: {loss_dens.item():.2e}, gmom2: {loss_gmom2.item():.2e} smom[0,1]: {loss_smom0.item():.2e},{loss_smom1.item():.2e}, gtau: {loss_gtau.item():.2e} lr: {optimizer.param_groups[0]["lr"]}')

            aw_opt = torch.nn.functional.softplus(aw_raw.detach())
            aw_opt = aw_opt/torch.trapezoid(aw_opt, self.ws.to(aw_opt), dim=-1).unsqueeze(-1).clamp_min(1e-8)
            gtaus_tm1 = make_gtau_from_Aw_integrate(aw_opt)
            giws_tm1 = make_gw_from_Aw_integrate(aw_opt)
            #endregion

        else:
            # region Adam + LBFGS optimization ##############
            with torch.enable_grad():
                aw_raw = torch.nn.Parameter(torch.log(torch.expm1(aw_base)))
                n_fit = siw_fit_range
                weights = 300/(1.0 + torch.arange(n_fit, device=self.device, dtype=torch.float32)/100)
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

    def gen_clean_giw(self, density, Giw, Gtau, G_moments, Siw, G0iw, S_moments, siw_fit_range=300):
        Giw_triqs = Giw.data[:,0,0]
        Siw_triqs = Siw.data[:,0,0]
        G0iw_triqs = G0iw.data[:,0,0]
        iws_triqs = np.array([w.imag for w in Giw.mesh])

        ## Need to now project Gtau onto adaptive grid for denoising
        taus_triqs = np.array([float(t) for t in Gtau.mesh])
        Gtau_rebin = rebinning_tau(Gtau, len(self.taus))
        taus_rebin = np.array([float(t) for t in Gtau_rebin.mesh])
        Gtau_proj = interp1d_nonuniform(taus_rebin, Gtau_rebin.data[:,0,0], self.taus.cpu().numpy())

        Gtau_data, Giw_data, Aw_data = self.denoise_gtau(Gtau_proj, Giw_triqs, Siw_triqs, G0iw_triqs, density, G_moments, S_moments, siw_fit_range)

        ### Since we have Gtau and Giw from spectral function, just integrate that
        Gtau_full_data = make_gtau_from_Aw_integrate_with_taus(taus_triqs, self.beta, self.ws.cpu().numpy(), Aw_data)
        Giw_full_data = make_gw_from_Aw_integrate_with_iws(iws_triqs, self.beta, self.ws.cpu().numpy(), Aw_data)
        Gtau_tmp = GfImTime(mesh=MeshImTime(beta=self.beta, n_tau=len(Gtau_full_data), statistic='Fermion'), data=Gtau_full_data.reshape(-1,1,1).astype(np.complex128))
        Giw_manual = GfImFreq(mesh=MeshImFreq(beta=self.beta, n_iw=len(Giw_full_data)//2, statistic='Fermion'), data=Giw_full_data.reshape(-1,1,1).astype(np.complex128))
        ##########################################################################

        return Gtau_tmp, Giw_manual, Aw_data






