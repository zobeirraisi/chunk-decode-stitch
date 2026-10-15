"""Self-supervised warp-consistency loss for the CurviRoPE chart.

The chart is intrinsic to the text: under a known affine warp T of the image,
the chart of the warped image, read at the warped grid points, must match the
chart of the original image. We enforce this with an L1 penalty. No geometry
labels are used."""
import torch
import torch.nn.functional as F


def _affine_theta(B, max_rot_deg, max_scale, device):
    ang = (torch.rand(B, device=device) * 2 - 1) * (max_rot_deg * torch.pi / 180)
    sc = 1.0 + (torch.rand(B, device=device) * 2 - 1) * max_scale
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos;  theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin;  theta[:, 1, 1] = cos
    return theta


def _affine_grid_for(theta, size):
    return F.affine_grid(theta, size, align_corners=False)


def warp_consistency_loss(model, images, max_rot_deg=20.0, max_scale=0.15):
    """L1 mismatch between chart(warp(x)) and warp(chart(x)).

    We compare the predicted (u,v) fields: warping the image should rotate the
    chart's midline the same way, leaving arc-length/normal coordinates of
    corresponding points unchanged. We approximate correspondence by warping the
    per-token (u,v) map with the same affine grid used on the image."""
    B = images.shape[0]
    theta = _affine_theta(B, max_rot_deg, max_scale, images.device)
    grid = _affine_grid_for(theta, images.shape)            # (B,H,W,2)
    warped = F.grid_sample(images, grid, align_corners=False)

    u0, v0 = model.chart_coords_grad(images)                # (B, Hf*Wf)
    u1, v1 = model.chart_coords_grad(warped)
    # ponytail: chart_coords_grad doesn't set model._mem_hw (only encode() does),
    # so derive (Hf, Wf) from the backbone feature map directly rather than
    # trusting a possibly-stale/unset attribute.
    _, _, Hf, Wf = model.cnn(images).shape
    chart0 = torch.stack([u0, v0], 1).view(B, 2, Hf, Wf)
    fgrid = _affine_grid_for(theta, (B, 2, Hf, Wf))
    chart0_w = F.grid_sample(chart0, fgrid, align_corners=False).view(B, 2, -1)
    chart1 = torch.stack([u1, v1], 1).view(B, 2, -1)
    return F.l1_loss(chart1, chart0_w)
