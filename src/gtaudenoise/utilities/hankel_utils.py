import torch

class HankelCaches:
    def __init__(self, ntau, device):
        if ntau % 2 != 1:
            raise ValueError("ntau must be odd")
        self.ntau = ntau
        self.nhalf = ntau // 2 + 1
        idx = torch.arange(self.nhalf, device=device)
        self.v2h_idx = idx[:, None] + idx[None, :]
        i = torch.arange(self.nhalf, device=device)
        self.k_idx = (i[:, None] + i[None, :]).reshape(-1)
        self.counts = torch.bincount(self.k_idx, minlength=2 * self.nhalf - 1)

def vector_to_hankel(f, cache):
    squeeze = f.dim() == 1
    if squeeze:
        f = f.unsqueeze(0)
    H = f[:, cache.v2h_idx]
    return H.squeeze(0) if squeeze else H

def hankel_to_vector(H, ntau):
    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)
    B, nhalf, _ = H.shape
    f = torch.empty((B, ntau), device=H.device, dtype=H.dtype)
    f[:, :nhalf] = H[:, 0, :]
    f[:, nhalf - 1:] = H[:, nhalf - 1, :]
    return f.squeeze(0) if squeeze else f

def project_to_hankel(H, cache):
    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)
    B, n, _ = H.shape
    H_flat = H.reshape(B, -1)
    K = cache.counts.numel()
    sums = torch.zeros((B, K), device=H.device, dtype=H.dtype)
    sums.scatter_add_(1, cache.k_idx.unsqueeze(0).expand(B, -1), H_flat)
    avgs = sums / cache.counts.to(dtype=H.dtype).unsqueeze(0).clamp_min(1)
    out = avgs.gather(1, cache.k_idx.unsqueeze(0).expand(B, -1)).reshape(B, n, n)
    return out.squeeze(0) if squeeze else out

def project_to_psd(H):
    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)
    H = 0.5 * (H + H.transpose(-1, -2))
    w, V = torch.linalg.eigh(H)
    w = w.clamp_min_(0)
    H = (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
    return H.squeeze(0) if squeeze else H

def project_to_density(H, density):
    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)
    H = H.clone()
    H[:, 0, 0] = 1.0 - density
    H[:, -1, -1] = density
    return H.squeeze(0) if squeeze else H

def project_top_to_psd(H):
    squeeze = H.dim() == 2
    if squeeze:
        H = H.unsqueeze(0)
    H = H.clone()
    H[:, :-1, 1:] = project_to_psd(H[:, :-1, 1:])
    H[:, 1:, 0] = H[:, 0, 1:]
    H[:, -1, :-1] = H[:, :-1, -1]
    return H.squeeze(0) if squeeze else H

@torch.no_grad()
def hankel_projection(Gtau, density, cache, max_iter=200, tol=1e-10, verbose=False):
    squeeze = Gtau.dim() == 1
    if squeeze:
        Gtau = Gtau.unsqueeze(0)

    #if (Gtau[:, 0] > 0).any():
    #    raise RuntimeError("expected G(tau)<0 (obvious)")

    B, ntau = Gtau.shape
    if ntau != cache.ntau:
        raise ValueError("Your cache has wrong ntau")

    f = (-Gtau).contiguous()
    H = vector_to_hankel(f, cache)
    x = H.clone()

    projections = [
        lambda h, y: project_to_density(h - y, density),
        lambda h, y: project_to_hankel(h - y, cache),
        lambda h, y: project_to_psd(h - y),
        lambda h, y: project_top_to_psd(h - y),
    ]
    y = [torch.zeros_like(x) for _ in range(4)]

    for it in range(max_iter):
        cI = 0.0

        for i in range(0,4):
            proj = projections[i]
            prev_x = x
            prev_y = y[i]
            x = proj(prev_x, prev_y)
            y[i] = x - (prev_x - prev_y)
            d = (prev_y - y[i]).norm()
            cI += (d * d).item()

        if verbose:
            corr = (prev_x - x).norm().item()
            print(f"### iteration: {it+1} tolerance: {cI} correction: {corr} ###")

        if cI < tol:
            break

    f_out = hankel_to_vector(x, ntau)
    G_out = -f_out
    return G_out.squeeze(0) if squeeze else G_out
