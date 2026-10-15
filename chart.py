"""Text-aligned curvilinear chart geometry for CurviRoPE.

A per-image Bézier midline + half-width defines an intrinsic coordinate system
on the feature grid:
    u(p) = normalized arc-length of p's projection onto the midline  (in [0,1])
    v(p) = signed normal offset of p from the midline / half-width    (in [-1,1])
Projection is a differentiable soft-argmin over densely sampled curve points,
so gradients flow to the control points without any geometry labels.
"""
import torch

HALF_WIDTH_FLOOR = 0.02


def _bernstein(n, t):
    # t: (S,) -> (S, n+1) Bernstein basis of degree n
    k = torch.arange(n + 1, device=t.device, dtype=t.dtype)
    from math import comb
    binom = torch.tensor([comb(n, int(i)) for i in k], device=t.device, dtype=t.dtype)
    tk = t.unsqueeze(1) ** k                      # (S, n+1)
    tn = (1 - t).unsqueeze(1) ** (n - k)          # (S, n+1)
    return binom * tk * tn                        # (S, n+1)


def bezier_curve(control_pts, n_samples):
    """control_pts: (..., K, 2) -> curve (..., n_samples, 2)."""
    K = control_pts.shape[-2]
    t = torch.linspace(0, 1, n_samples, device=control_pts.device,
                       dtype=control_pts.dtype)
    B = _bernstein(K - 1, t)                       # (S, K)
    return torch.einsum("sk,...kd->...sd", B, control_pts)


def _grid_xy(H, W, device, dtype):
    ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)   # (H*W, 2) row-major


def _tangent(curve):
    """Central-difference tangent along the sample axis. curve: (B,S,2)."""
    t = torch.zeros_like(curve)
    t[:, 1:-1] = curve[:, 2:] - curve[:, :-2]
    t[:, 0] = curve[:, 1] - curve[:, 0]
    t[:, -1] = curve[:, -1] - curve[:, -2]
    return t


def grid_to_uv(control_pts, half_width, H, W, n_samples=100, tau=0.01):
    """Soft-project each grid token onto the Bézier midline.

    control_pts: (B, K, 2) in normalized image coords (x right, y down).
    half_width:  (B,) normalized half-width of the text band.
    Returns u, v each (B, H*W)."""
    if control_pts.dim() == 2:
        control_pts = control_pts.unsqueeze(0)
    Bn = control_pts.shape[0]
    curve = bezier_curve(control_pts, n_samples)               # (B, S, 2)
    s = torch.linspace(0, 1, n_samples, device=curve.device,
                       dtype=curve.dtype)                       # arc param proxy
    grid = _grid_xy(H, W, curve.device, curve.dtype)           # (N, 2)
    N = grid.shape[0]
    # pairwise sq distances grid(N) x curve(S)
    d2 = (grid.view(1, N, 1, 2) - curve.view(Bn, 1, n_samples, 2)).pow(2).sum(-1)
    w = torch.softmax(-d2 / tau, dim=-1)                        # (B, N, S) soft-argmin
    u = (w * s.view(1, 1, -1)).sum(-1)                          # (B, N)
    proj = torch.einsum("bns,bsd->bnd", w, curve)              # (B, N, 2) soft proj point
    # tangent at the projected arc position (finite diff along the curve)
    tang = torch.einsum("bns,bsd->bnd", w, _tangent(curve))    # (B, N, 2)
    tang = tang / (tang.norm(dim=-1, keepdim=True) + 1e-8)
    diff = grid.view(1, N, 2) - proj                           # (B, N, 2)
    # signed normal offset: 2D cross product tangent x diff
    signed = tang[..., 0] * diff[..., 1] - tang[..., 1] * diff[..., 0]
    hw = half_width.view(Bn, 1).clamp_min(HALF_WIDTH_FLOOR)
    v = (signed / hw).clamp(-1.0, 1.0)
    return u, v
