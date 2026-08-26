import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from scipy.interpolate import PchipInterpolator
device="cpu"

### Lets pretend beta=1.0 for now. Corresponds to T=11,1604K. Can always rescale Tau anyway 
### Also add small finite offset since G(0+) = -G(beta+0+), endpoints will be a bit wack
M = 2**5 - 1
beta = 20.0
eps = 1e-8

### Tanh space grid
alpha = 1.0                                    ## Tune this to adjust edge density
t_grid = 0.5 * beta * (1.0 + torch.tanh(alpha * torch.linspace(-1.0, 1.0, M, device=device)) / torch.tanh(torch.tensor(alpha, device=device)))
t_grid[0] = 0.0
t_grid[-1] = beta


### Lets start with integrating by discretizing by 100
### Also, base 4 might be better to decrease density of points in interior
w_len = 128
ws = torch.linspace(-5, 5, w_len, device=device)

### Define matsubara frequencies
n_matsubara = M//6
wn = np.pi*(2*torch.arange(n_matsubara, device=device) + 1)/beta
wn = wn.to(torch.float64)

#### Build covariance matrix in beginning for generation of Gaussian noise samples
#### Basically just doing Cij = wi - wj, C= LC*(LC)^T and gaussian noise = N(0,1)*L^T
w_matrix = ws[:,None] - ws[None,:]
C = torch.exp(-torch.pow(w_matrix, 2)/(2*1.0**2)) + 1e-4*torch.eye(len(ws), device=device)  #Can add extra term for numerical stability
LC = torch.linalg.cholesky(C)


### Change training target to u_i instead of A_i with p_i = softmax(u_i) and A_i = p_i/h_i 
### where h_i is defined as 2*h_0 = w_1-w_0, 2*h_i = w_i - w_(i-1) i=[1,N_w-1], 2*h_(N_w) = w_(N_w) - w_(N_w-1)
dw = (ws[1:] - ws[:w_len-1])/2
hs = torch.empty_like(ws)
hs[0] = dw[0]
hs[-1] = dw[-1]
hs[1:w_len-1] = dw[1:] + dw[:-1]

dw_l2 = (ws[1]-ws[0]) * torch.ones_like(ws)
dw_l2[0] *= 0.5
dw_l2[-1] *= 0.5


### Precompute the simpson weights for better integration for A(w)
sw = torch.empty_like(ws)
dw = ws[1]-ws[0]
sw[0::2] = 2*dw/3
sw[1::2] = 4*dw/3
sw[0] = dw/3
sw[-1] = dw/3

### Precompute trapz weights for tau values
dtau = (t_grid[1:] - t_grid[:len(t_grid)-1])/2
tz_taus = torch.empty_like(t_grid)
tz_taus[0] = dtau[0]
tz_taus[-1] = dtau[-1]
tz_taus[1:len(t_grid)-1] = dtau[1:] + dtau[:-1] 

## Precompute Fermionic Kernel, so less calculations 
FKernel = 1/(torch.exp(t_grid[:,None]*ws[None,:]) + torch.exp((t_grid[:,None]-beta)*ws[None, :]))

## Precompute Legendre Transform Kernel, call it Ll(w). Need to use hs from above for delta2 w
## Lets compute up to lmax=100, should be fine for vast majority of cases
taus_np = t_grid.cpu().numpy()
Llw = torch.zeros(100, len(ws), device=device)
for l in range(100):
    leg_polys = torch.from_numpy(scipy.special.eval_legendre(l, 2*taus_np/beta-1)).to(torch.get_default_dtype()).to(device)
    integrand = FKernel*leg_polys[:,None]
    Llw[l] = np.sqrt(2*l+1)*torch.trapz(integrand, t_grid, dim=0)

def set_utils_device(device_):
    global device
    device = device_

def get_beta():
    return beta

def set_beta(betain, ntau=M):
    global beta, taus_np, t_grid, wn, tz_taus, n_matsubara, M, FKernel
    beta = betain
    M = ntau

    ### Uniform grid
    # t_grid = torch.linspace(0.0 + eps, beta-eps, ntau, device=device)

    ### Logspace grid
    # n_side = (M - 3) // 2
    # t_grid_left = torch.logspace(-8, np.log2(beta / 2.0), steps=n_side + 1, base=2.0, device=device)
    # t_grid_right = beta - torch.flip(t_grid_left[:-1], dims=[0])
    # t_grid = torch.cat((torch.tensor([0.0], device=device), t_grid_left, t_grid_right, torch.tensor([beta], device=device)), dim=0)

    ### Tanh grid
    alpha = max(np.log10(beta), 1.0) ## 1.0 is about equidistant spacing
    t_grid = 0.5 * beta * (1.0 + torch.tanh(alpha * torch.linspace(-1.0, 1.0, M, device=device)) / torch.tanh(torch.tensor(alpha, device=device)))
    t_grid[0] = 0.0
    t_grid[-1] = beta

    # plt.scatter(t_grid.cpu().numpy(), t_grid.cpu().numpy())
    # plt.show()
    # exit()

    n_matsubara = ntau//1
    wn = np.pi*(2*torch.arange(n_matsubara, device=device) + 1)/beta
    wn = wn.to(torch.float64)

    dtau = (t_grid[1:] - t_grid[:len(t_grid)-1])/2
    tz_taus = torch.empty_like(t_grid)
    tz_taus[0] = dtau[0]
    tz_taus[-1] = dtau[-1]
    tz_taus[1:len(t_grid)-1] = dtau[1:] + dtau[:-1] 

    # FKernel = 1/(torch.exp(t_grid[:,None]*ws[None,:]) + torch.exp((t_grid[:,None]-beta)*ws[None, :]))

def get_ws():
    return ws

def set_ws(w_lenin, w_max, meshform="hyperbolic"):
    global w_len, ws, dw, hs, FKernel, Llw, sw
    w_len = w_lenin

    if meshform=="linear":
        ws = torch.linspace(-w_max, w_max, w_len, device=device)
    elif meshform=="hyperbolic":
        ### Hyperbolic mesh ###
        alpha = 3.0
        x = torch.linspace(-1, 1, w_len, device=device)
        ws = w_max*torch.sinh(alpha*x)/torch.sinh(torch.tensor(alpha, device=device))
    else:
        print("ERROR:", meshform, "not recognized! Valid forms are linear and hyperbolic")
        exit()

    dw = (ws[1:] - ws[:w_len-1])/2
    hs = torch.empty_like(ws)
    hs[0] = dw[0]
    hs[-1] = dw[-1]
    hs[1:w_len-1] = dw[1:] + dw[:-1]

    dw_l2 = hs.clone()
    sw = torch.zeros_like(ws)
    h0 = ws[1:-1:2] - ws[:-2:2]
    h1 = ws[2::2] - ws[1:-1:2]
    sw[:-2:2] += (h0+h1)*(2*h0-h1)/(6*h0)
    sw[1:-1:2] += (h0+h1)**3/(6*h0*h1)
    sw[2::2] += (h0+h1)*(2*h1-h0)/(6*h1)

    if w_len % 2 == 0:
        sw[-2] += (ws[-1]-ws[-2])/2
        sw[-1] += (ws[-1]-ws[-2])/2

    ## Precompute Fermionic Kernel, so less calculations 
    # FKernel = torch.exp(-t_grid[:,None]*ws[None,:])/(1+torch.exp(-beta*ws[None, :]))
    FKernel = 1/(torch.exp(t_grid[:,None]*ws[None,:]) + torch.exp((t_grid[:,None]-beta)*ws[None, :]))

    ## Precompute Legendre Transform Kernel, call it Ll(w). Need to use hs from above for delta2 w
    ## Lets compute up to lmax=100, should be fine for vast majority of cases
    taus_np = t_grid.cpu().numpy()
    Llw = torch.zeros(100, len(ws), device=device)
    for l in range(100):
        leg_polys = torch.from_numpy(scipy.special.eval_legendre(l, 2*taus_np/beta-1)).to(torch.get_default_dtype()).to(device)
        integrand = FKernel*leg_polys[:,None]
        Llw[l] = np.sqrt(2*l+1)*torch.trapz(integrand, t_grid, dim=0)

def get_hs():
    return hs

def get_wn():
    return wn

def get_taus():
    return t_grid

def get_wn():
    return wn

def get_fkernel():
    return FKernel

### Need to add finite shift so log doesn't go crazy when Aw = 0
def gen_u_from_Aw(Aw, eps=1e-10):
    u = torch.log(hs[None, :]*Aw + eps)
    u = u - torch.mean(u, dim=-1, keepdim=True)
    return u

def gen_Aw_from_u(u):
    ps = torch.softmax(u, dim=-1)
    Aw = ps/hs
    return Aw

def make_gtau_from_Aw_integrate(Aw):
    Integrand = -1*FKernel
    return torch.einsum("tw,bw->bt", Integrand, hs*Aw)

def make_gtau_from_Aw_trapz(Aw):
    Integrand = -1*FKernel[None, :, :]*Aw[:, None, :]
    return torch.trapz(Integrand, ws, dim=-1)

def make_gtau_from_Aw_integrate_simpson(Aw):
    # Integrand = -1*FKernel[None, :, :]*Aw[:, None, :]
    # return torch.trapz(Integrand, ws, dim=-1)
    assert len(ws)%2 == 1
    Integrand = -1*FKernel
    return torch.einsum('nw,bw->bn', Integrand, Aw * sw[None, :])

def gen_integrand(w, tau, ar, wr, sigma_r):
    return np.sum(-ar*np.exp(-0.5*((w-wr)/sigma_r)**2)/(np.exp(tau*w)+np.exp(-(beta-tau)*w)))

def make_gtau_from_Aw_integrate_quad(ars, wrs, sigma_rs):
    ars = ars.cpu().numpy()
    wrs = wrs.cpu().numpy()
    sigma_rs = sigma_rs.cpu().numpy()
    ws_np = ws.cpu().numpy()
    gtaus = np.zeros((sigma_rs.shape[0], t_grid.shape[0]))
    for i in tqdm(range(len(ars)), desc="Generate G(tau)'s..."):
        for ti in tqdm(range(t_grid.shape[0]), desc=f"Going through {i}"):
            tau = t_grid.cpu().numpy()[ti]
            gtaus[i, ti], _ = scipy.integrate.quad(gen_integrand, ws_np[0], ws_np[-1], 
                           args=(tau, ars[i], wrs[i], sigma_rs[i]))
    return torch.from_numpy(gtaus).to(device)

def make_gtau_from_Aw_params(ars, wrs, sigma_rs):
    Gtaus = torch.einsum("brw,tw,w->bt", -ars[:,:,None]*torch.exp(-0.5*((ws[None, None,:]-wrs[:, :,None])/sigma_rs[:, :,None])**2), FKernel, hs)
    return Gtaus

def make_gl_from_Aw_integrate(Aw, lmax=30):
    # Gl = torch.zeros(Aw.shape[0], lmax, device=device)
    # for l in range(lmax):

    #     ### Old way of calculating 
    #     # x = -t_grid[None, :] * ws[:, None] 
    #     # x = x.clamp(max=50.0) 
    #     # tau_integrand = torch.from_numpy(scipy.special.eval_legendre(l, 2*taus_np/beta-1)).to(device)[None,:]*torch.exp(x)
    #     # tau_integrated = torch.trapz(tau_integrand, t_grid, dim=-1)
    #     # ws_integrand = -1*np.sqrt(2*l+1)*Aw*tau_integrated[None, :]*torch.sigmoid(beta*ws)
    #     # Gl[:,l] = torch.trapz(ws_integrand, ws, dim=-1)

    #     ### New more numerically stable way of calculating
    #     # tau_integrand = torch.from_numpy(scipy.special.eval_legendre(l, 2*taus_np/beta-1)).to(device)[None,:] \
    #     #                 /(torch.exp(t_grid[None,:]*ws[:,None]) + torch.exp((t_grid[None,:]-beta)*ws[:,None]))
    #     # tau_integrated = torch.trapz(tau_integrand, t_grid, dim=-1)
    #     # ws_integrand = -1*np.sqrt(2*l+1)*Aw*tau_integrated[None, :]
    #     # Gl[:,l] = torch.trapz(ws_integrand, ws, dim=-1)

    #     ### Even newer, just using matrix multiplciation to make it quick
    #     Gl[:,l] = torch.sum(-1*hs[None,:]*Llw[l][None, :]*Aw, dim=-1)

    # ### Just get rid of loop altogether
    # Gl = torch.sum(-1*hs[None, None, :]*Llw[None, :lmax, :]*Aw[:, None, :], dim=-1)

    ### Make efficient with einsum
    Gl = -torch.einsum('bw,lw->bl', hs[None, :]*Aw, Llw[:lmax])
    return Gl

def make_gl_from_gtau_integrate(Gtaus, lmax=50):
    Gls = torch.zeros(Gtaus.shape[0], lmax, device=device)
    xtau = 2*t_grid/beta - 1
    leg_polys = torch.special.legendre_polynomial_p(xtau[:, None], torch.arange(lmax, device=device))
    Gls = torch.einsum("tl,bt,t->bl", torch.sqrt(2*torch.arange(lmax, device=device)+1)[None,:]*leg_polys, Gtaus, tz_taus)
    return Gls

def make_gtau_from_gl(Gls):
    Gtaus = torch.zeros(Gls.shape[0], len(t_grid), device=device)
    lmax = Gls.shape[1]
    xtau = 2*t_grid/beta - 1
    leg_polys = torch.special.legendre_polynomial_p(xtau[:, None], torch.arange(lmax, device=device))
    Gtaus = torch.einsum("bl,tl->bt",torch.sqrt(2*torch.arange(lmax, device=device)+1)[None,:]*Gls/beta,leg_polys)
    return Gtaus

def make_gw_from_Aw_integrate(Aw):
    # kernel = 1/(1j*wn[:,None] - ws[None, :])
    # Gw = torch.trapz(kernel[None, :, :]*Aw[:, None, :], ws, dim=-1).to(torch.complex128)
    kernel = 1/(1j*wn[:,None] - ws[None, :])
    kernel = kernel.to(torch.complex128)
    Aw = Aw.to(torch.float64)
    Gw = torch.einsum('nw,bw->bn', kernel, (Aw * hs[None, :]).to(torch.complex128))
    return Gw

def make_gw_from_gtau_integrate(Gtaus):
    Giws = torch.zeros(Gtaus.shape[0], n_matsubara, device=device, dtype=torch.complex128)
    kernel = torch.exp(1j*wn[:,None]*t_grid[None,:])*tz_taus[None,:]
    Giws = torch.einsum("nt,bt->bn", kernel.to(torch.complex128), Gtaus.to(torch.complex128))
    return Giws

def make_gw_from_gtau_triqs(Gtaus):
    Giws = torch.zeros(Gtaus.shape[0], n_matsubara, device=device, dtype=torch.complex128)
    for b in range(Gtaus.shape[0]):
        giw_tmp = GfImFreq(mesh=MeshImFreq(beta=beta, n_iw=n_matsubara, statistic='Fermion'), target_shape=[1,1])
        interp = PchipInterpolator(t_grid.cpu().numpy(), Gtaus[b].cpu().numpy())
        upscaled_gtau = interp(np.linspace(0, beta, 10001))
        gtau_tmp = GfImTime(mesh=MeshImTime(beta=beta, n_tau=len(upscaled_gtau), statistic='Fermion'), data=upscaled_gtau.reshape(-1,1,1).astype(np.complex128))
        giw_tmp.set_from_fourier(gtau_tmp)
        Giws[b] = torch.from_numpy(giw_tmp.data[n_matsubara:,0,0])
    return Giws

def make_gw_from_gtau_torch_interp(Gtaus):
    Giws = torch.zeros(Gtaus.shape[0], n_matsubara, device=device, dtype=torch.complex128)
    taus_upscaled = torch.linspace(0, beta, 10001, device=device)
    Gtaus_upscaled=interp1d_cubic_custom(Gtaus, 10001)
    dt = (taus_upscaled[1:] - taus_upscaled[:10001-1])/2
    tz_up = torch.empty_like(taus_upscaled)
    tz_up[0] = dt[0]
    tz_up[-1] = dt[-1]
    tz_up[1:10001-1] = dt[1:] + dt[:-1] 
    Giws = torch.einsum('nt,bt->bn', torch.exp(1j*wn[:,None]*taus_upscaled[None,:]), (Gtaus_upscaled*tz_up[None,:]).to(torch.complex128))
    return Giws

def make_gtau_from_gw(Giws, densities=None):
    Gtaus = torch.zeros(Giws.shape[0], len(t_grid), dtype=torch.complex64)
    Gtau = torch.einsum("tn,bn->bt", torch.exp(-1j*wn[None,:]*t_grid[:,None]).to(torch.complex128), Giws.to(torch.complex128) - 1/(1j*get_wn()[None,:]))/beta
    Gtau = Gtau + Gtau.conj()
    Gtau = Gtau - 0.5

    if densities is not None:
        Gtau[:,0] = densities-1
        Gtau[:,-1] = -densities
    return Gtau.to(torch.float32)

def make_gtau_from_gw_triqs(Giws):
    Gtaus = torch.zeros(Giws.shape[0], len(t_grid), device=device, dtype=torch.complex128)
    for b in range(Gtaus.shape[0]):
        gtau_tmp = GfImTime(mesh=MeshImTime(beta=beta, n_tau=len(t_grid), statistic='Fermion'), target_shape=[1,1])
        giw_tmp_data = np.concatenate((np.flip(np.conj(Giws[b].cpu().numpy())), Giws[b].cpu().numpy())).astype(np.complex128).reshape(-1,1,1)
        giw_tmp = GfImFreq(mesh=MeshImFreq(beta=beta, n_iw=n_matsubara, statistic='Fermion'), data = giw_tmp_data)
        gtau_tmp.set_from_fourier(giw_tmp)
        Gtaus[b] = torch.from_numpy(gtau_tmp.data[:,0,0])
    return Gtaus

def make_gtau_from_gw_fit_tail(Giws, densities=None, tail_order=3, fit_n_tail=100, fix_M1=True, M1_value=1.0, enforce_real_moments=True):
    import torch
    Giws = Giws.to(torch.complex128)
    wn = get_wn()
    t_grid = get_taus()
    beta = get_beta()
    batch, n_iw = Giws.shape
    fit_n_tail = min(fit_n_tail, n_iw)
    iw = 1j * wn
    x = 1.0 / iw
    x_fit = x[-fit_n_tail:]
    y_fit = Giws[:, -fit_n_tail:]
    moments = torch.zeros(batch, tail_order, dtype=torch.complex128, device=Giws.device)
    if fix_M1:
        M1 = M1_value.to(device=Giws.device, dtype=torch.complex128).expand(batch) if torch.is_tensor(M1_value) and M1_value.ndim == 0 else torch.full((batch,), M1_value, dtype=torch.complex128, device=Giws.device) if not torch.is_tensor(M1_value) else M1_value.to(device=Giws.device, dtype=torch.complex128)
        moments[:, 0] = M1
        if tail_order > 1:
            A = torch.stack([x_fit ** k for k in range(2, tail_order + 1)], dim=1)
            y_res = y_fit - M1[:, None] * x_fit[None, :]
            moments[:, 1:] = torch.linalg.lstsq(A, y_res.T).solution.T
    else:
        A = torch.stack([x_fit ** k for k in range(1, tail_order + 1)], dim=1)
        moments[:, :] = torch.linalg.lstsq(A, y_fit.T).solution.T
    if enforce_real_moments:
        moments = moments.real.to(torch.complex128)
    Giw_tail = sum(moments[:, k - 1, None] * (x[None, :] ** k) for k in range(1, tail_order + 1))
    Giws_reg = Giws - Giw_tail
    phase = torch.exp(-1j * t_grid[:, None] * wn[None, :]).to(torch.complex128)
    Gtau = 2.0 * (torch.einsum("tn,bn->bt", phase, Giws_reg) / beta).real + _fermionic_tail_tau_from_moments(moments, t_grid, beta).real
    if densities is not None:
        densities = densities.to(device=Giws.device, dtype=Gtau.dtype)
        Gtau[:, 0] = densities - 1.0
        Gtau[:, -1] = -densities
    return Gtau.to(torch.float32) #, moments

def make_gtau_from_gw_legendre(Giws,densities=None, lmax=80, trunc_start=6, patience=4):
    import torch
    Giws = Giws.to(torch.complex128)
    device = Giws.device
    wn = get_wn()
    t_grid = get_taus()
    beta = get_beta()
    ell = torch.arange(lmax, device=device, dtype=torch.float64)
    ell_f = ell.to(torch.float64)
    z = wn[:, None] * beta / 2.0
    j0 = torch.sin(z) / z
    j1 = torch.sin(z) / z**2 - torch.cos(z) / z
    js = [j0, j1]
    for l in range(1, lmax - 1): js.append((2 * l + 1) / z * js[-1] - js[-2])
    J = torch.cat(js, dim=1)
    n = torch.arange(wn.numel(), device=device, dtype=torch.float64)
    T = ((-1.0) ** n[:, None]) * ((1j) ** (ell_f[None, :] + 1.0)) * torch.sqrt(2.0 * ell_f[None, :] + 1.0) * J
    phase = ((-1.0) ** (ell_f + 1.0))[None, :]
    Gl = torch.einsum("nl,bnl->bl", T.conj(), Giws[:, :, None] + phase * Giws.conj()[:, :, None]).real.to(torch.complex128)
    a = Gl.abs().mean(0)
    bad = a[1:] >= a[:-1]
    bad[:trunc_start] = False
    hit = torch.nn.functional.conv1d(bad.to(torch.float64)[None, None, :], torch.ones(1, 1, patience, device=device, dtype=torch.float64)).squeeze() >= patience
    cut = torch.cat([torch.where(hit, torch.arange(hit.numel(), device=device) + 1, lmax * torch.ones_like(torch.arange(hit.numel(), device=device))).min()[None], torch.tensor([lmax], device=device)]).min()
    Gl = Gl * (ell[None, :] < cut)
    
    x = 2.0 * t_grid / beta - 1.0
    P = torch.stack([torch.special.legendre_polynomial_p(x, int(l)) for l in ell], dim=1).to(torch.complex128)
    Gtau = ((P * (torch.sqrt(2.0 * ell_f + 1.0) / beta).to(torch.complex128)[None, :]) @ Gl.T).T.real
    if densities is not None:
        densities = densities.to(device=device, dtype=Gtau.dtype)
        Gtau[:, 0] = densities - 1.0
        Gtau[:, -1] = -densities
    return Gtau.to(torch.float32)#, Gl, cut

def make_gtau_from_gw_triqs_like(Giws, densities=None, n_moments=3, fit_fraction=0.35):
    Giws = Giws.to(torch.complex128)
    beta = get_beta()
    device = Giws.device
    B, N = Giws.shape
    t_grid = get_taus()
    n_pos = torch.arange(N, device=device, dtype=torch.float64)
    w_pos = (2 * n_pos + 1) * torch.pi / beta
    w_full = torch.cat((-torch.flip(w_pos, dims=[0]), w_pos))
    G_full = torch.cat((torch.flip(Giws.conj(), dims=[1]), Giws), dim=1)
    iw = 1j * w_full
    n_fit = max(n_moments + 2, int(fit_fraction * N))
    idx = torch.cat((torch.arange(n_fit, device=device), torch.arange(2 * N - n_fit, 2 * N, device=device)))
    A = torch.stack([1.0 / iw[idx] ** k for k in range(1, n_moments + 1)], dim=1)
    M = torch.linalg.lstsq(A, G_full[:, idx].T).solution.T
    M = M.real.to(torch.complex128)
    G_tail_full = sum(M[:, k - 1, None] / iw[None, :] ** k for k in range(1, n_moments + 1))
    G_reg_full = G_full - G_tail_full
    phase = torch.exp(-1j * t_grid[:, None] * w_full[None, :]).to(torch.complex128)
    Gtau_reg = torch.einsum("tw,bw->bt", phase, G_reg_full).real / beta
    tau = t_grid
    A_tau = []
    A_tau.append(-0.5 * torch.ones_like(tau))
    A_tau.append(tau / 2.0 - beta / 4.0)
    A_tau.append(beta * tau / 4.0 - tau**2 / 4.0)
    A_tau = torch.stack(A_tau[:n_moments], dim=0).to(torch.complex128)
    Gtau = Gtau_reg + torch.einsum("bk,kt->bt", M, A_tau).real
    if densities is not None:
        densities = densities.to(device=device, dtype=Gtau.dtype)
        Gtau[:, 0] = densities - 1.0
        Gtau[:, -1] = -densities
    else:
        Gtau[:,-1] = -(1+Gtau[:,0])
    return Gtau.to(torch.float32)

def _fermionic_tail_tau_from_moments(moments, t_grid, beta):
    import torch
    tau = t_grid.to(device=moments.device, dtype=torch.float64)
    beta = torch.as_tensor(beta, device=moments.device, dtype=torch.float64)
    A = [-0.5 * torch.ones_like(tau)]
    if moments.shape[1] >= 2: A.append(tau / 2.0 - beta / 4.0)
    if moments.shape[1] >= 3: A.append(beta * tau / 4.0 - tau**2 / 4.0)
    if moments.shape[1] >= 4: A.append(tau**3 / 12.0 - beta * tau**2 / 8.0 + beta**3 / 48.0)
    if moments.shape[1] >= 5: A.append(-tau**4 / 48.0 + beta * tau**3 / 24.0 - beta**3 * tau / 48.0)
    if moments.shape[1] >= 6: A.append(tau**5 / 240.0 - beta * tau**4 / 192.0 + beta**3 * tau**2 / 96.0 - beta**5 / 480.0)
    return torch.einsum("bk,kt->bt", moments, torch.stack(A, dim=0).to(moments.dtype))

def noise_gtau(gtaus, giws, timesteps, sigmas):
    noise = sigmas[timesteps][:,None]*torch.normal(torch.zeros(giws.shape, device=device))
    

    ## Try adding noise procedurally to G(iw_n)
    N_full = 20
    prog_sig = N_full/np.sqrt(12)
    prog_sched = (1.03 - torch.exp(-0.5*(torch.arange(0, giws.shape[1], device=device)/prog_sig)**2))[None,:]
    giws_t = giws + prog_sched*noise + 1j*prog_sched*noise 
    # gtaus_t = make_gtau_from_gw_fit_tail(giws)
    gtaus_t = make_gtau_from_gw(giws_t)
    # giws_t = make_gw_from_gtau_integrate(gtaus_t)

    # fig, axs = plt.subplots(2)
    # taus = get_taus()
    # axs[0].plot(giws_t[0].imag.cpu().numpy(), c="blue")
    # axs[0].plot(giws[0].imag.cpu().numpy(), c='black', linestyle="--")
    # axs[0].plot(giws_t[0].real.cpu().numpy(), c="red")
    # axs[0].plot(giws[0].real.cpu().numpy(), c='black', linestyle="--")
    # axs[1].plot(taus.cpu().numpy(), gtaus_t[0].cpu().numpy(), c="red", zorder=10)
    # axs[1].scatter(taus.cpu().numpy(), gtaus[0].cpu().numpy(), c='black', linestyle="--", zorder=0)
    # plt.show()
    # exit()

    return gtaus_t, giws_t

def interp1d_cubic_custom(y, out_len, align_corners=True):
    B, T = y.shape
    x = torch.linspace(0, T - 1, out_len, device=y.device, dtype=y.dtype)
    x0 = torch.floor(x).long()
    t = (x - x0.to(y.dtype))[None, :]
    i0 = (x0 - 1).clamp(0, T - 1)
    i1 = x0.clamp(0, T - 1)
    i2 = (x0 + 1).clamp(0, T - 1)
    i3 = (x0 + 2).clamp(0, T - 1)
    p0 = y[:, i0]
    p1 = y[:, i1]
    p2 = y[:, i2]
    p3 = y[:, i3]
    a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3
    b = p0 - 2.5*p1 + 2.0*p2 - 0.5*p3
    c = -0.5*p0 + 0.5*p2
    d = p1
    return ((a*t + b)*t + c)*t + d

def calc_giw0(Gtaus):
    if len(t_grid)%2==1:
        Gtaus_mod = interp1d_cubic_custom(Gtaus, len(t_grid)+1)
        taus = torch.linspace(0, beta, len(t_grid)+1, device=device)
    else:
        Gtaus_mod = Gtaus
    half_point = len(taus)//2
    left_gtau = Gtaus_mod[:,:half_point]
    right_gtau = torch.flip(Gtaus_mod[:,half_point:], dims=(-1,))
    half_taus = t_grid[:half_point]
    im_giw0 = torch.trapz(torch.sin(np.pi*half_taus[None,:]/beta)*(left_gtau + right_gtau), half_taus, dim=-1)
    rl_giw0 = torch.trapz(torch.cos(np.pi*half_taus[None,:]/beta)*(left_gtau - right_gtau), half_taus, dim=-1)
    return rl_giw0 + 1j*im_giw0

def compute_phis(wn, Giw):
    batch_size = Giw.shape[0]
    M = len(wn)
    iw = 1j*wn
    th = (-Giw - 1j)/(-Giw + 1j)

    iw = torch.flip(iw, dims=[-1])
    th = torch.flip(th, dims=[-1])

    phis = torch.zeros(batch_size, M, dtype=torch.complex128, device=device)
    abcds = torch.zeros(batch_size, M, 2, 2, dtype=torch.complex128, device=device)
    phis[:, 0] = th[:, 0]  # :contentReference[oaicite:13]{index=13}
    for k in range(M):
        abcds[:, k] = torch.eye(2, dtype=torch.complex128, device=device).repeat(batch_size, 1, 1)  
    

    for j in range(M - 1):
        for k in range(j, M):
            frac = (iw[k] - iw[j]) / (iw[k] - torch.conj(iw[j]))
            pj = phis[:, j]  # (B,)

            prod = torch.zeros(batch_size, 2, 2, dtype=torch.complex128, device=device)
            prod[:, 0, 0] = frac      
            prod[:, 0, 1] = pj
            prod[:, 1, 0] = torch.conj(pj) * frac
            prod[:, 1, 1] = 1.0
            abcds[:, k] = torch.bmm(abcds[:, k], prod)

            ### Renormalize for stablity?
            s = torch.amax(torch.abs(abcds[:, k]), dim=(1,2), keepdim=True)
            abcds[:, k] = abcds[:, k] / s
    
        A = abcds[:, j + 1]
        phis[:, j + 1] = (-A[:, 1, 1] * th[:, j + 1] + A[:, 0, 1]) / (A[:, 1, 0] * th[:, j + 1] - A[:, 0, 0])

    return phis
    
def compute_phis_numpy(wn, Giw):
        """
        Memoize intermediate 2x2 products and compute phis exactly as in Schur<T>::core(). :contentReference[oaicite:12]{index=12}
        """
        M = len(wn)
        iw = 1j*np.asarray(wn, dtype=np.complex128)
        th = (-Giw - 1j)/(-Giw + 1j)
        iw = iw[::-1]
        th = th[::-1]
        phis = np.empty(M, dtype=np.complex128)
        abcds = np.zeros((M, 2, 2), dtype=np.complex128)


        phis[0] = th[0]  # :contentReference[oaicite:13]{index=13}
        for k in range(M):
            abcds[k] = np.eye(2, dtype=np.complex128)  # :contentReference[oaicite:14]{index=14}

        # j = 0..M-2 :contentReference[oaicite:15]{index=15}
        for j in range(M - 1):
            for k in range(j, M):
                frac = (iw[k] - iw[j]) / (iw[k] - np.conj(iw[j]))  # :contentReference[oaicite:16]{index=16}
                prod = np.array(
                    [
                        [frac, phis[j]],
                        [np.conj(phis[j]) * frac, 1.0 + 0j],
                    ],
                    dtype=np.complex128,
                )
                abcds[k] = abcds[k] @ prod  # :contentReference[oaicite:17]{index=17}

            # phis[j+1] update :contentReference[oaicite:18]{index=18}
            A = abcds[j + 1]
            phis[j + 1] = (-A[1, 1] * th[j + 1] + A[0, 1]) / (A[1, 0] * th[j + 1] - A[0, 0])
            # if j < 3:
            #     print(f"j={j}: phis[{j+1}] = {phis[j+1]}")
            #     print(f"       abcds[{j+1}] = {abcds[j+1]}")

        return phis

def inv_mobius(theta):
    return 1j*(1.0 + theta)/(1.0 - theta)

def hardy_theta_mplus1(z, ak, bk):
    kvals = torch.arange(0, ak.shape[-1], device=device)
    fk = torch.pow((z-1j)/(z+1j), kvals)/(np.sqrt(np.pi)*(z+1j))
    return torch.sum(ak*fk[None,:] + bk*torch.conj(fk)[None,:], dim=-1)

def build_Aw_hardy_basis(wn, ws, phis, ak, bk, eta=1e-3):
    batch_size = len(phis)
    M = len(wn)
    ws_len = len(ws)
    iw = 1j*torch.flip(wn, dims=(0,)).to(torch.complex128)
    z = ws + 1j * eta
    z = z.to(torch.complex128)
    phis = phis.reshape(batch_size, M, 2)
    phis = torch.complex(phis[:,:,0], phis[:,:,1]).to(torch.complex128)

    Aw = torch.zeros(batch_size, ws_len, device=device, dtype=torch.float64)

    for i in range(ws_len):
        result = torch.eye(2, dtype=torch.complex128, device=device)
        zi = z[i]
        for j in range(M):
            frac = (zi - iw[j]) / (zi - torch.conj(iw[j]))
            pj = phis[:, j]  # batch size vector
            prod = torch.zeros(batch_size, 2, 2, dtype=torch.complex128, device=device)
            prod[:, 0, 0] = frac
            prod[:, 0, 1] = pj
            prod[:, 1, 0] = torch.conj(pj)*frac
            prod[:, 1, 1] = 1.0
            result = result @ prod
        param = hardy_theta_mplus1(zi, ak, bk)
        theta = (result[:, 0, 0] * param + result[:, 0, 1]) / (result[:, 1, 0] * param + result[:, 1, 1])  
        NG = inv_mobius(theta)
        Aw[:, i] = (1.0 / np.pi) * NG.imag
    Aw = Aw.to(torch.float32)
    return Aw

def hardy_theta_mplus1_vec(z, ak, bk):
    K = ak.shape[-1]
    kvals = torch.arange(K, device=z.device)
    w = (z - 1j) / (z + 1j)                           
    fk = (w[:, None] ** kvals[None, :]) / (math.sqrt(math.pi) * (z[:, None] + 1j))  
    return ak @ fk.T + bk @ torch.conj(fk).T

def build_Aw_hardy_basis_fast(wn, ws, phis, ak, bk, eta=1e-3):
    device = ws.device
    B = phis.shape[0]
    M = wn.numel()
    W = ws.numel()

    ak = ak.to(torch.complex128)
    bk = bk.to(torch.complex128)
    cdtype = ak.dtype
    z = (ws + 1j * eta).to(cdtype)                 
    wn_flip = torch.flip(wn, dims=(0,)).to(z.real.dtype) 
    iw = (1j * wn_flip).to(cdtype) 
    if phis.is_complex():
        phic = phis.to(cdtype)
    else:                      
        phis = phis.reshape(B, M, 2)
        phic = torch.complex(phis[..., 0], phis[..., 1]).to(cdtype)
    param = hardy_theta_mplus1_vec(z, ak, bk)   
    theta = param
    for j in range(M - 1, -1, -1):
        p = phic[:, j][:, None]
        f = (z - iw[j]) / (z + iw[j])
        f = f[None, :]
        theta = (f * theta + p) / (torch.conj(p) * f * theta + 1.0)
    NG = inv_mobius(theta)
    Aw = (NG.imag / math.pi).to(torch.float32)
    return Aw

def gen_Aw(ars, wrs, sigma_rs, insulating=False):
    # Aw = ars[:,:,None]*torch.exp(-0.5*torch.pow(ws[None, None, :]-wrs[:,:,None], 2)/torch.pow(sigma_rs[:,:,None],2))
    # Aw = torch.sum(Aw, dim=1)
    Aw = torch.zeros(ars.shape[0], len(ws), device=ars.device)
    for g in range(ars.shape[1]):  # Loop over num_gauss
        Aw += ars[:, g:g+1] * torch.exp(-0.5 * ((ws[None,:] - wrs[:,g:g+1])**2) / (sigma_rs[:,g:g+1]**2))
    if insulating:
        mask = (get_ws() >= -0.5) & (get_ws() <= 0.5)
        inds = torch.nonzero(mask, as_tuple=True)[0]
        decay_func = torch.linspace(-1, 1, len(inds), device=device)**4
        Aw[:,inds] = Aw[:,inds]*decay_func[None,:]
    Aw_integrated = torch.trapz(Aw, ws, dim=1)
    Aw = Aw/Aw_integrated[:, None]
    return Aw

def normalize_ars(ars, sigma_rs):
    return ars/torch.sum(ars*sigma_rs*np.sqrt(2*np.pi), dim=1, keepdim=True)

### wr_cutoff is just the max w where the gaussians for A(w) can be centered
### num_gauss is the max number of gaussian peaks to consider, must be 3 or greater 
def sample_Aw(batch, num_gauss=3, wr_cutoff=3, sigma_0_point=0.02):
    assert num_gauss >= 3
    ### Split into 3 regions, w<-0.2, -0.2<w<0.2 and w>0.2
    ### Also make the width in the inner regions small as per Arsenault 2017
    ### Actually lets use treatment in Fournier 2020, where w = [-10, 10] (should be [-15,15] but whatever)
    ### w_r = [-6,6] but w_0 = [-0.5, 0.5], \sigma_r = [0.1, 1], R=[3,21] (but we choose R=10 for now)
    ### Apparently they don't use variable heights so we can set a_r = 1
    sigma_rs = (1-sigma_0_point)*torch.rand(batch, num_gauss, device=device) + sigma_0_point

    ars = torch.rand(batch, num_gauss, device=device)/2 + 0.5
    n_active = torch.randint(3, num_gauss + 1, (batch,), device=device)  
    indices = torch.arange(num_gauss, device=device).unsqueeze(0).expand(batch, -1)  
    mask = (indices < n_active.unsqueeze(1)).float() 
    ars = ars*mask
    ars = normalize_ars(ars, sigma_rs)

    #w0 =  torch.rand(batch, 1, device=device)-0.5
    #wrs = torch.cat([w0, wr_cutoff*(2*(torch.rand(batch, num_gauss-1, device=device)-0.5))], dim=1)
    wrs = wr_cutoff*(2*(torch.rand(batch, num_gauss, device=device)-0.5))

    ### Lets instead make 20% of the A(w) insulating, and 80% can be whatever
    ### Protect a region -0.1 to 0.1 around Fermi level
    N_ins = batch//5
    Aw_ins = gen_Aw(ars[:N_ins], wrs[:N_ins], sigma_rs[:N_ins], insulating=True)
    Aw_any = gen_Aw(ars[N_ins:], wrs[N_ins:], sigma_rs[N_ins:], insulating=False)
    Aw = torch.cat([Aw_any, Aw_ins], dim=0)

    Giws = make_gw_from_Aw_integrate(Aw)
    Gtaus = make_gtau_from_Aw_integrate(Aw)

    #### Compare Gtau using adaptive mesh 
    # Gtaus_trapz = make_gtau_from_Aw_trapz(Aw)
    # Gtaus_params = make_gtau_from_Aw_params(ars, wrs, sigma_rs)
    # fig, axs = plt.subplots(2,1)
    # axs[0].plot(t_grid.cpu().numpy(), Gtaus[0].cpu().numpy(), marker="o", label="Integrate")
    # axs[0].plot(t_grid.cpu().numpy(), Gtaus_trapz[0].cpu().numpy(), marker="o", label="Trapz")
    # axs[0].plot(t_grid.cpu().numpy(), Gtaus_params[0].cpu().numpy(), marker="o", label="Params")
    # axs[0].legend()
    # axs[1].plot(t_grid.cpu().numpy(), Gtaus[1].cpu().numpy(), marker="o", label="Integrate")
    # axs[1].plot(t_grid.cpu().numpy(), Gtaus_trapz[1].cpu().numpy(), marker="o", label="Trapz")
    # axs[1].plot(t_grid.cpu().numpy(), Gtaus_params[1].cpu().numpy(), marker="o", label="Params")
    # axs[1].legend()
    # plt.show()
    # exit()

    #### Compare Gtau using adaptive mesh 
    # plt.plot(wn.cpu().numpy(), Giws[0].cpu().numpy().imag, marker="o", label="Integrate")
    # plt.legend()
    # plt.show()
    # exit()

    ### Compare trapz to quadrature using numerically exact function as well as simpson integration
    # Gtaus_simpson = make_gtau_from_Aw_integrate_simpson(Aw)
    # Gtaus_precise = make_gtau_from_Aw_integrate_quad(ars, wrs, sigma_rs)
    # for i in range(Gtaus.shape[0]):
    #     # if i == 0:
    #     #     plt.plot(t_grid.cpu().numpy(), Gtaus[i].cpu().numpy() + i/20, c='red', label="Trapz")
    #     #     plt.plot(t_grid.cpu().numpy(), Gtaus_precise[i].cpu().numpy() + i/20, c='blue', label="Quad")
    #     #     plt.plot(t_grid.cpu().numpy(), Gtaus_simpson[i].cpu().numpy() + i/20, c='green', label="Simpson")
    #     # else:
    #     #     plt.plot(t_grid.cpu().numpy(), Gtaus[i].cpu().numpy() + i/20, c='red')
    #     #     plt.plot(t_grid.cpu().numpy(), Gtaus_precise[i].cpu().numpy() + i/20, c='blue')
    #     #     plt.plot(t_grid.cpu().numpy(), Gtaus_simpson[i].cpu().numpy() + i/20, c='green')
    #     plt.plot(t_grid.cpu().numpy(), Gtaus[i].cpu().numpy() - Gtaus_precise[i].cpu().numpy() + i/20, c='red')
    #     plt.plot(t_grid.cpu().numpy(), Gtaus_simpson[i].cpu().numpy() - Gtaus_precise[i].cpu().numpy() + i/20, c='blue')
    # plt.legend()
    # plt.show()
    # exit()
    #######################################################
    

    return Giws, Gtaus, Aw

### gaussian noise = N(0,1)*LC^T where LC comes from above
def gen_gauss_noise(batch):
    ucn = torch.randn(batch, w_len, device=device)
    eps = ucn@LC.T
    return eps

### Select Aw based on which one best reproduces G(tau)
def select_Aw_from_us(us, gtau):
    Aws = gen_Aw_from_u(us)
    gtaus = make_gtau_from_Aw_integrate(Aws)
    best_gtau_ind = torch.argmin(torch.sum(torch.abs(gtaus-gtau[None, :]), dim=-1))
    return Aws[best_gtau_ind]

def calc_Aw_loss(Aw_true, Aw_pred):
    return torch.sum(torch.abs(Aw_true-Aw_pred))

def grad_log_Aw(Aw, y_target, sig_y = 1e-3):
    Aw_req = Aw.detach().requires_grad_(True)
    y_pred = make_gtau_from_Aw_integrate(Aw_req)
    loss = (torch.pow(y_pred-y_target, 2).mean())/(2.0*sig_y**2)
    loss.backward()
    return -Aw_req.grad.detach()

def enforce_Aw_constraint(Aw):
    # Aw = torch.clamp(Aw, min=0.0)
    # Aw = torch.abs(Aw)
    # Z = torch.trapz(Aw, ws, dim=-1)[:, None]
    # # print(Z)
    # Aw = Aw/Z
    return Aw

def get_noise_schedule(nt=100, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, nt, device=device)
    # betas = torch.pow(torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), nt, device=device),2)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    return betas, alphas, alphas_cumprod

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    alphas = 1 - betas  # This was missing
    # alphas_cumprod = alphas_cumprod[1:]  
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-12, max=1.0)
    return betas, alphas, alphas_cumprod

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.pow(torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), timesteps, device=device),2)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return betas, alphas, alphas_cumprod

def geometric_log_sigma_schedule(nt=100, sigma_start=1e-3, sigma_end=0.2):
    return torch.exp(torch.linspace(math.log(sigma_start), math.log(sigma_end), nt, device=device))

def reverse_noise_schedule_avg_with_guidance(model, ws, wn, nt, alphas, alphas_cumprod, betas, gtaus, zeta=0.1, final_opt_steps=1000, n_samples=8):
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.)
    timesteps = torch.arange(nt-1, -1, -1, device=device)    
    batch_size = gtaus.shape[0]
    gtaus = gtaus.repeat(n_samples, 1)
    zt = torch.randn(gtaus.shape[0], len(ws), device=device)
    for t in tqdm(timesteps, desc="Running through timesteps..."):
        tinp = t.repeat(gtaus.shape[0]).reshape(gtaus.shape[0], 1)

        ## Apparently need to give zt gradients for guided reverse diffusion gradient calculation
        zt = zt.detach().requires_grad_(True)
        aks, bks = model(zt, gtaus, tinp.squeeze(-1))
        z0 = build_Aw_hardy_basis_fast(wn, ws, gtaus, aks, bks, eta=1e-3)

        grad = None
        zeta_scaled=None
        # if t > (nt-1)//2 or t==0 or zeta==0:
        if alphas_cumprod[t] < 0.0 or t==0 or zeta==0:
            zeta_scaled = 0.0/(torch.sqrt(1-alphas_cumprod[tinp[:,0]]) + 1e-6)
            grad = torch.zeros_like(z0)
        else:
            ### Just set equal to z0 for physics guided loss
            z0p = z0

            ### This modifies z0 to DDIM estimate, should be slightly more accurate for gradients
            # t1_times = torch.zeros(tinp[:,0].shape, dtype=int, device=device)
            # alphas_cm_1 = alphas_cumprod[t1_times][:, None]
            # alphas_cm_t = alphas_cumprod[tinp[:,0]][:, None]
            # x1 = torch.sqrt(alphas_cm_1) * z0 + \
            #     torch.sqrt((1 - alphas_cm_1) / (1 - alphas_cm_t)) * (zt - torch.sqrt(alphas_cm_t) * z0)
            # z0p = model(x1, gtaus, t1_times)

            ### Recompute z0 with detached gradients so that it doesn't backprop through model weights 
            # with torch.no_grad():
            #     z0p = model(zt, gtaus, tinp.squeeze(-1))

            gtau_pred = make_gl_from_Aw_integrate(z0p)
            measurement_loss_vec = ((gtaus - gtau_pred)**2)
            # measurement_loss_vec = (gtaus - gtau_pred).abs()/gtaus.abs()
            measurement_loss = measurement_loss_vec.mean()
            grad = torch.autograd.grad(measurement_loss, zt)[0]
            grad = torch.clamp(grad, -5.0, 5.0)

            zeta_scaled = zeta/(torch.linalg.norm(measurement_loss_vec, dim=-1) + 1e-8)
            # zeta_scaled = torch.clamp(zeta_scaled, 0.0, 50)


        print(f"t={t.item()}, grad norm: {grad[0].norm().item():.12f}, \
        grad max: {grad[0].abs().max().item():.12f}, zeta_scaled: {zeta_scaled[0].item():.6f}")
        if t > 0:
            noise = torch.randn_like(zt)
            var = betas[tinp[:,0]]*(1-alphas_cumprod_prev[tinp[:,0]])/(1-alphas_cumprod[tinp[:,0]])
            var = torch.sqrt(var.clip(1e-20)[:,None])*noise

            term1 = betas[tinp[:,0]]*torch.sqrt(alphas_cumprod_prev[tinp[:,0]])/(1-alphas_cumprod[tinp[:,0]])
            term2 = (1-alphas_cumprod_prev[tinp[:,0]])*torch.sqrt(alphas[tinp[:,0]])/(1-alphas_cumprod[tinp[:,0]])
            ztminusone = term1[:,None]*z0.detach() + term2[:,None]*zt.detach() - zeta_scaled[:,None] * grad + var
        else:
            print(aks)
            print(bks)
            ztminusone = z0 

        zt = ztminusone
    
    

    #### Maybe we can try further optimizing it using gradients at the end?
    

    ###Lets try logits optimization
    if final_opt_steps > 0:
        z0 = torch.nn.functional.relu(zt.clone().detach())
        u0 = gen_u_from_Aw(z0)
        u0 = u0.detach().requires_grad_(True)
        opt = torch.optim.Adam([u0], lr=1e-2)
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=final_opt_steps, eta_min=1e-12)
        # sched = torch.optim.lr_scheduler.StepLR(opt, step_size=final_opt_steps//10, gamma=1.0)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50)
        for i in range(final_opt_steps):
            opt.zero_grad()
            
            Aw_pred = gen_Aw_from_u(u0)
            gtau_pred = make_gl_from_Aw_integrate(Aw_pred)

            # diff_loss = ((gtaus - gtau_pred).abs()/gtaus.abs()).mean()
            diff_loss = ((gtaus-gtau_pred)**2).sum()
            # diff_loss = ((torch.log(gtaus.abs()) - torch.log(gtau_pred.abs()))**2).mean()
            # plt.plot(diff_loss.detach().cpu().numpy()[0])
            # plt.show()
            # exit()
            measurement_loss = diff_loss 
            
            measurement_loss.backward()
            torch.nn.utils.clip_grad_norm_([u0], max_norm=1.0)  
            opt.step()
            sched.step(measurement_loss.item())
            
            if i % 1 == 0:  
                print(f"Step {i}: loss = {measurement_loss.item()}, diff_loss = {diff_loss.item()}", \
                    f"lr = {opt.param_groups[0]['lr']}")
        ##############################################################
        z0 = gen_Aw_from_u(u0)
    else:
        z0 = zt.clone()
    z0 = z0.detach().view(n_samples, batch_size, len(ws)).mean(dim=0)
    print(torch.trapz(z0, ws.repeat(batch_size, 1), dim=-1)[:,None])
    return z0

# Updated project_to_measurement with better optimization
def project_to_measurement(Aw, gtau_target, num_iters=10, lr=0.01):
    ### Project A(ω) to satisfy G(tau) constraint with gradient descent####
    Aw_opt = Aw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([Aw_opt], lr=lr)
    
    for i in range(num_iters):
        optimizer.zero_grad()
        gtau_pred = make_gl_from_Aw_integrate(Aw_opt)
        loss = ((gtau_pred - gtau_target)**2).mean()
        loss.backward()
        optimizer.step()
        
        if i == 0 or i == num_iters - 1:
            print(f"  Projection iter {i+1}/{num_iters}, loss: {loss.item():.8f}")
    
    return Aw_opt.detach()

def gaussian_log_likelihood(x, means, variance, return_full = False):
    centered_x = x - means
    squared_diffs = (centered_x ** 2) / variance[:,None]
    if return_full:
        log_likelihood = -0.5 * (squared_diffs + torch.log(variance)[:,None] + torch.log(2 * torch.pi)) # full log likelihood with constant terms
    else:
        log_likelihood = -0.5 * squared_diffs

    # avoid log(0)
    log_likelihood = torch.clamp(log_likelihood, min=-27.6310211159)

    return log_likelihood

def get_res_log_likelihood(Aws, gtaus, ws, var, gtau_match_coeff = 1.0, int_coeff=1.0):
    # res1 = gtaus-make_gtau_from_Aw_integrate(Aws)
    gls = make_gl_from_Aw_integrate(Aws)
    # print(gls)
    res1 = gtaus - gls

    ### Maybe try rescaling lower l?
    # scale = gtaus.abs()
    # res1 = res1/scale

    res1_log_likelihood = gaussian_log_likelihood(torch.zeros_like(res1), means=res1, variance=var)
    res = res1_log_likelihood.mean()    

    return res

def get_ineq_log_likelihood(Aws, var):
    term1 = torch.nn.functional.relu(-Aws)
    # ineq_log_likelihood = torch.sum(gaussian_log_likelihood(torch.zeros_like(term1), means=term1, variance=var), dim=-1, keepdim=True)
    ineq_log_likelihood = gaussian_log_likelihood(torch.zeros_like(term1), means=term1, variance=var).mean()
    # print(ineq_log_likelihood.item())
    return ineq_log_likelihood
    # return term1



