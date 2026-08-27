import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class FourierFeatureEmbedding(nn.Module):
    def __init__(self, dim, max_freq=256.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, math.log(max_freq), dim // 2)))

    def forward(self, x):
        args = 2.0 * math.pi * x[:, None] * self.freqs[None, :].to(x.device, x.dtype)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) / math.sqrt(in_channels * out_channels))

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=x_ft.dtype)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m].to(x_ft.dtype))
        return torch.fft.irfft(out_ft, n=x.shape[-1])

class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.gate = nn.Conv1d(width, width, 1)

    def forward(self, x):
        y = self.spectral(x) + self.pointwise(x)
        return x + F.silu(y) * torch.sigmoid(self.gate(x))

class FourierNeuralOperator1D(nn.Module):
    def __init__(self, in_channels, out_channels=1, width=96, modes=16, depth=6):
        super().__init__()
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, out_channels)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x):
        x = self.lift(x).permute(0, 2, 1)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training else block(x)
        x = x.permute(0, 2, 1)
        return self.proj2(F.silu(self.proj1(x)))[..., 0]

class LowConstraintProjector(nn.Module):
    def __init__(self, kernel_low, ridge=1e-6):
        super().__init__()
        kernel_low = kernel_low.to(torch.complex64)
        n_tau = kernel_low.shape[1]
        e0 = torch.zeros(1, n_tau, device=kernel_low.device, dtype=torch.float32)
        e1 = torch.zeros(1, n_tau, device=kernel_low.device, dtype=torch.float32)
        e0[0, 0] = 1.0
        e1[0, -1] = 1.0
        A = torch.cat([kernel_low.real, kernel_low.imag, e0, e1], dim=0).to(torch.float32)
        P = torch.linalg.solve(A @ A.T + ridge * torch.eye(A.shape[0], device=A.device, dtype=A.dtype), A).T.contiguous()
        self.register_buffer("kernel_low", kernel_low)
        self.register_buffer("A", A)
        self.register_buffer("P", P)

    def forward(self, gtau, giw_low, gtau_end):
        g0 = -1.0 - gtau_end.float()
        target = torch.cat([giw_low.real.float(), giw_low.imag.float(), g0[:, None], gtau_end.float()[:, None]], dim=-1)
        current = gtau.float() @ self.A.T
        correction = (target - current) @ self.P.T
        return gtau + correction.to(gtau.dtype)

class TauResidualDenoiser(nn.Module):
    def __init__(self, n_tau, nt, sigmas, kernel_low, giw_scale, width=96, modes=16, depth=6, emb_dim=32, cond_dim=24):
        super().__init__()
        self.nt = nt
        self.n_tau = n_tau
        self.t_embed = FourierFeatureEmbedding(emb_dim)
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.register_buffer("tau", torch.linspace(0.0, 1.0, n_tau, device=kernel_low.device))
        log_sigmas = torch.log(sigmas.float().clamp_min(1e-12))
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigma_min", log_sigmas[0])
        self.register_buffer("log_sigma_span", (log_sigmas[-1] - log_sigmas[0]).clamp_min(1e-12))
        self.fno = FourierNeuralOperator1D(in_channels=4 + cond_dim, out_channels=1, width=width, modes=modes, depth=depth)

    def normalized_t(self, t_idx):
        return (torch.log(self.sigmas[t_idx.long()].clamp_min(1e-12)) - self.log_sigma_min) / self.log_sigma_span

    def forward(self, x_t, t_idx, gtau_end):
        b = x_t.shape[0]
        t_feat = self.t_mlp(self.t_embed(self.normalized_t(t_idx))).unsqueeze(1).expand(b, self.n_tau, -1)
        tau_feat = self.tau[None, :, None].expand(b, -1, -1)
        end_feat = torch.stack([gtau_end.float(), -1.0 - gtau_end.float()], dim=-1).unsqueeze(1).expand(b, self.n_tau, -1)
        fno_in = torch.cat([x_t[:, :, None].float(), tau_feat, end_feat, t_feat], dim=-1)
        return x_t + self.fno(fno_in).to(x_t.dtype)

class LowGiwHead(nn.Module):
    def __init__(self, k_low, nt, sigmas, giw_scale, hidden=256, emb_dim=32):
        super().__init__()
        self.k_low = k_low
        self.t_embed = FourierFeatureEmbedding(emb_dim)
        log_sigmas = torch.log(sigmas.float().clamp_min(1e-12))
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("giw_scale", giw_scale.float())
        self.register_buffer("log_sigma_min", log_sigmas[0])
        self.register_buffer("log_sigma_span", (log_sigmas[-1] - log_sigmas[0]).clamp_min(1e-12))
        self.net = nn.Sequential(nn.Linear(6 * k_low + emb_dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * k_low))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def normalized_t(self, t_idx):
        return (torch.log(self.sigmas[t_idx.long()].clamp_min(1e-12)) - self.log_sigma_min) / self.log_sigma_span

    def forward(self, giw_noisy_low, giw_raw_low, t_idx, gtau_end):
        scale = self.giw_scale[None, :].to(giw_raw_low.real.dtype)
        noisy = torch.cat([giw_noisy_low.real / scale, giw_noisy_low.imag / scale], dim=-1).float()
        raw = torch.cat([giw_raw_low.real / scale, giw_raw_low.imag / scale], dim=-1).float()
        diff = raw - noisy
        feat = torch.cat([noisy, raw, diff, self.t_embed(self.normalized_t(t_idx)).float(), gtau_end[:, None].float(), (-1.0 - gtau_end)[:, None].float()], dim=-1)
        delta = self.net(feat).view(-1, self.k_low, 2).to(giw_raw_low.real.dtype)
        return torch.complex(giw_raw_low.real + delta[..., 0] * scale, giw_raw_low.imag + delta[..., 1] * scale)

class SpectralAuxHead(nn.Module):
    def __init__(self, n_tau, k_low, n_ws, nt, sigmas, giw_scale, hidden=256, emb_dim=32):
        super().__init__()
        self.t_embed = FourierFeatureEmbedding(emb_dim)
        log_sigmas = torch.log(sigmas.float().clamp_min(1e-12))
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("giw_scale", giw_scale.float())
        self.register_buffer("log_sigma_min", log_sigmas[0])
        self.register_buffer("log_sigma_span", (log_sigmas[-1] - log_sigmas[0]).clamp_min(1e-12))
        self.net = nn.Sequential(nn.Linear(2 * n_tau + 4 * k_low + emb_dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_ws))

    def normalized_t(self, t_idx):
        return (torch.log(self.sigmas[t_idx.long()].clamp_min(1e-12)) - self.log_sigma_min) / self.log_sigma_span

    def forward(self, x_t, gtau_pred, giw_noisy_low, giw_low_pred, t_idx, gtau_end):
        scale = self.giw_scale[None, :].to(giw_low_pred.real.dtype)
        low_feat = torch.cat([giw_noisy_low.real / scale, giw_noisy_low.imag / scale, giw_low_pred.real / scale, giw_low_pred.imag / scale], dim=-1).float()
        feat = torch.cat([x_t.float(), gtau_pred.float(), low_feat, self.t_embed(self.normalized_t(t_idx)).float(), gtau_end[:, None].float(), (-1.0 - gtau_end)[:, None].float()], dim=-1)
        return F.softplus(self.net(feat))

class GiwProjectedDenoiser(nn.Module):
    def __init__(self, ntau, nt, kernel_low, sigmas, giw_scale, n_ws=1000, width=96, modes=16, depth=6, emb_dim=32, cond_dim=24):
        super().__init__()
        self.ntau = ntau
        self.nt = nt
        self.k_low = kernel_low.shape[0]
        self.register_buffer("kernel_low", kernel_low.to(torch.complex64))
        self.register_buffer("giw_scale", giw_scale.float())
        self.projector = LowConstraintProjector(kernel_low)
        self.base = TauResidualDenoiser(ntau, nt, sigmas, kernel_low, giw_scale, width=width, modes=modes, depth=depth, emb_dim=emb_dim, cond_dim=cond_dim)
        self.low_head = LowGiwHead(self.k_low, nt, sigmas, giw_scale, hidden=2 * width, emb_dim=emb_dim)
        self.aw_head = SpectralAuxHead(ntau, self.k_low, n_ws, nt, sigmas, giw_scale, hidden=2 * width, emb_dim=emb_dim)
        self.sigmas = sigmas

    def low_transform(self, gtau):
        return torch.einsum("kn,bn->bk", self.kernel_low, gtau.to(torch.complex64))

    def project_to_constraints(self, gtau, giw_low, gtau_end):
        return self.projector(gtau, giw_low, gtau_end)

    def forward_with_aux(self, x_t, t_idx, gtau_end):
        giw_noisy_low = self.low_transform(x_t)
        gtau_raw = self.base(x_t, t_idx, gtau_end)
        giw_raw_low = self.low_transform(gtau_raw)
        giw_low_pred = self.low_head(giw_noisy_low, giw_raw_low, t_idx, gtau_end)
        gtau_pred = self.projector(gtau_raw, giw_low_pred, gtau_end)
        aw_pred = self.aw_head(x_t, gtau_pred, giw_noisy_low, giw_low_pred, t_idx, gtau_end)
        return gtau_pred, {"gtau_raw": gtau_raw, "giw_noisy_low": giw_noisy_low, "giw_raw_low": giw_raw_low, "giw_low_pred": giw_low_pred, "aw_pred": aw_pred}

    def forward(self, x_t, t_idx, gtau_end):
        return self.forward_with_aux(x_t, t_idx, gtau_end)[0]

class FNOHankelNet(GiwProjectedDenoiser):
    def __init__(self, ntau, nt, kernel_low, sigmas, giw_scale, temb=32, cond_hidden=24, modes=16, num_fourier_layers=6, n_ws=1000, width=96):
        super().__init__(ntau=ntau, nt=nt, kernel_low=kernel_low, sigmas=sigmas, giw_scale=giw_scale, n_ws=n_ws, width=width, modes=modes, depth=num_fourier_layers, emb_dim=temb, cond_dim=cond_hidden)
