import math
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from gtaudenoise.utilities.utils import *
from gtaudenoise.models import AuxModel

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
nt = 200
K_LOW = 16
K_MID = 48
epochs = 600
batch_size = 1024
lr = 3e-4
weight_decay = 1e-4
sigmas = torch.linspace(1e-2, 1.0, nt, device=device)
Path("saved_models").mkdir(parents=True, exist_ok=True)
data = torch.load("Data/train_test_data.pt", map_location=torch.device(device))
gtaus_train = data["gtaus_train"].to(device=device, dtype=torch.float32)
giws_train = data["giws_train"].to(device=device, dtype=torch.complex64)
Aws_train = data["Aws_train"].to(device=device, dtype=torch.float32)
beta = data["beta"]
ws_max = data["ws_max"]
N = gtaus_train.shape[0]
N_tau = gtaus_train.shape[1]
N_ws = Aws_train.shape[1]
set_beta(beta, N_tau)
set_ws(N_ws, ws_max)

def make_kernel(n_tau, k):
    eye = torch.eye(n_tau, device=device, dtype=torch.float32)
    return make_gw_from_gtau_integrate(eye)[:, :k].T.contiguous().to(torch.complex64)

def transform_with_kernel(gtau, kernel):
    return torch.einsum("kn,bn->bk", kernel, gtau.to(torch.complex64))

def second_derivative(x):
    return torch.gradient(torch.gradient(x, dim=1)[0], dim=1)[0]

def make_training_noise(model, gtaus, timesteps):
    curv = second_derivative(gtaus)
    curv = curv / curv.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    gtaus_t = gtaus + sigmas[timesteps][:, None] * torch.randn_like(gtaus) * (0.0125 + 0.0175 * curv.abs())
    z = (torch.randn(gtaus.shape[0], K_LOW, device=device, dtype=torch.float32) + 1j * torch.randn(gtaus.shape[0], K_LOW, device=device, dtype=torch.float32)) / math.sqrt(2.0)
    giw_low_target = model.low_transform(gtaus_t) + (0.03 + 0.35 * sigmas[timesteps])[:, None].float() * model.giw_scale[None, :].to(torch.float32) * z
    return model.project_to_constraints(gtaus_t, giw_low_target, gtaus[:, -1]).detach()

kernel_low = make_kernel(N_tau, K_LOW)
kernel_mid = make_kernel(N_tau, K_MID)
giw_scale_low = giws_train[:, :K_LOW].abs().std(dim=0).float().clamp_min(1e-4)
giw_scale_mid = giws_train[:, :K_MID].abs().std(dim=0).float().clamp_min(1e-4)
low_w = 1.0 / (1.0 + torch.arange(K_LOW, device=device).float())**1.5
low_w = low_w / low_w.mean()
mid_w = torch.exp(-torch.arange(K_MID, device=device).float() / 14.0)
mid_w = mid_w / mid_w.mean()
ws_grid = torch.linspace(-ws_max, ws_max, N_ws, device=device)
aw_w = 1.0 + 50.0 * torch.exp(-0.5 * (ws_grid / 0.25)**2) + 8.0 * torch.exp(-0.5 * (ws_grid / 1.0)**2)
aw_w = aw_w / aw_w.mean()
gtau_scale = gtaus_train.std().clamp_min(1e-4)
aw_scale = Aws_train.abs().mean().clamp_min(1e-5)
curv_scale = second_derivative(gtaus_train[:2048]).std().clamp_min(1e-5)
model = AuxModel(ntau=N_tau, nt=nt, kernel_low=kernel_low, sigmas=sigmas, giw_scale=giw_scale_low, n_ws=N_ws, width=96, modes=16, depth=6, emb_dim=32, cond_dim=24).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-5)
metrics = []
model.train()
for epoch in range(epochs):
    perm = torch.randperm(N, device=device)
    running = torch.zeros(8, device=device)
    nbatches = 0
    for start in tqdm(range(0, N - batch_size + 1, batch_size), desc=f"epoch {epoch + 1}/{epochs}"):
        idx = perm[start:start + batch_size]
        gtaus = gtaus_train[idx]
        giws = giws_train[idx]
        Aws = Aws_train[idx]
        random_t = torch.randint(0, nt, (gtaus.shape[0],), device=device)
        max_t = torch.full((gtaus.shape[0],), nt - 1, device=device, dtype=torch.long)
        timesteps = torch.where(torch.rand(gtaus.shape[0], device=device) < 0.7, max_t, random_t)
        gtaus_t = make_training_noise(model, gtaus, timesteps)
        gtau_pred, aux = model.forward_with_aux(gtaus_t, timesteps, gtaus[:, -1])
        giw_low_pred_final = model.low_transform(gtau_pred)
        giw_mid_pred = transform_with_kernel(gtau_pred, kernel_mid)
        err_low = (giw_low_pred_final - giws[:, :K_LOW]).abs() / giw_scale_low[None, :]
        err_head = (aux["giw_low_pred"] - giws[:, :K_LOW]).abs() / giw_scale_low[None, :]
        err_mid = (giw_mid_pred - giws[:, :K_MID]).abs() / giw_scale_mid[None, :]
        loss_low = (low_w[None, :] * err_low.pow(2)).mean()
        loss_head = (low_w[None, :] * err_head.pow(2)).mean()
        loss_mid = (mid_w[None, :] * err_mid.pow(2)).mean()
        loss_tau = F.smooth_l1_loss(gtau_pred / gtau_scale, gtaus / gtau_scale)
        loss_curv = F.smooth_l1_loss(second_derivative(gtau_pred) / curv_scale, second_derivative(gtaus) / curv_scale)
        loss_end = F.mse_loss(gtau_pred[:, -1], gtaus[:, -1]) + F.mse_loss(gtau_pred[:, 0], gtaus[:, 0])
        loss_aw = (aw_w[None, :] * ((aux["aw_pred"] - Aws) / aw_scale).pow(2)).mean()
        loss = 50.0 * loss_low + 25.0 * loss_head + 3.0 * loss_mid + 0.35 * loss_tau + 0.15 * loss_curv + 0.05 * loss_aw + 2.0 * loss_end
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += torch.tensor([loss.detach(), loss_low.detach(), loss_head.detach(), loss_mid.detach(), loss_tau.detach(), loss_curv.detach(), loss_aw.detach(), loss_end.detach()], device=device)
        nbatches += 1
    sched.step()
    avg = running / nbatches
    metrics.append({"epoch": epoch + 1, "loss": avg[0].item(), "low": avg[1].item(), "head": avg[2].item(), "mid": avg[3].item(), "tau": avg[4].item(), "curv": avg[5].item(), "aw": avg[6].item(), "end": avg[7].item(), "lr": opt.param_groups[0]["lr"]})
    torch.save(model.state_dict(), "saved_models/epsnet.pth")
    torch.save({"metrics": metrics, "K_LOW": K_LOW, "K_MID": K_MID, "nt": nt}, "saved_models/training_metrics.pt")
    print(f"epoch {epoch + 1} loss {avg[0].item():.6e} low {avg[1].item():.6e} head {avg[2].item():.6e} mid {avg[3].item():.6e} tau {avg[4].item():.6e} curv {avg[5].item():.6e} aw {avg[6].item():.6e} end {avg[7].item():.6e} lr {opt.param_groups[0]['lr']:.6e}")
torch.save(model.state_dict(), "saved_models/epsnet.pth")
