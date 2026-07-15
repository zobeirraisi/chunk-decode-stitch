"""Long + curved synthetic text: the one combination no existing dataset
covers, and the genuinely open question the chunked-decoding pilot could not
close with real data (docs/paperc-chunked-pilot.md, non-horizontal-layout
stress test): does chunk stitching degrade on text that bends across chunk
boundaries, as opposed to LTB's overwhelmingly straight horizontal signage?

Reuses `synth_controlled_length.py`'s real-dictionary-word phrases at
natural, content-fit width (the Phase-1 deferred-item generator: legible
text, image width scales with length like a real photographed sign).

Does NOT reuse `curved_synth.py`'s `random_midline_ctrl`/`warp_along_midline`
directly: that generator fits exactly one Bézier arc over the WHOLE image
width, which is the right design for its own use case (a single short word on
a fixed 128px canvas) but is the wrong one here -- stretched over a
650-1200px long-text canvas, the same one-arc-per-image curve becomes nearly
flat within any single 128px chunk, defeating the point of a per-chunk
curvature stress test. Instead this module generates a curve with a FIXED
PIXEL wavelength (so every ~128px window sees a comparable amount of bend
regardless of overall image width -- the way a real curved sign's physical
curvature doesn't depend on how many words happen to be printed on it) and
warps with the same grid_sample mechanics `warp_along_midline` uses,
factored out here as `warp_along_curve` since it takes an explicit per-column
curve instead of Bézier control points.
"""
import math
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from synth_controlled_length import make_phrase, render_controlled

SEED_BASE = 777_000  # disjoint from synth_controlled_length's own 999_000
WAVELENGTH_PX = 160  # ~one chunk_w (128px) + some slack, so every chunk sees bend


def warp_along_curve(img_t: torch.Tensor, y_curve: torch.Tensor) -> torch.Tensor:
    """img_t: (3,H,W). y_curve: (W,) normalized-y midline in [0,1], one value
    per column. Shifts each column vertically onto this midline. Same
    grid_sample mechanics as curved_synth.warp_along_midline, factored out to
    take an explicit curve instead of Bézier control points."""
    _, H, W = img_t.shape
    disp = (y_curve - 0.5) * 2.0
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    sample_y = gy - disp.unsqueeze(0)
    grid = torch.stack([gx, sample_y], dim=-1).unsqueeze(0)
    return F.grid_sample(img_t.unsqueeze(0), grid, align_corners=True,
                         padding_mode="zeros")[0]


def _fixed_wavelength_midline(width_px: int, max_bend: float, rng: random.Random,
                              wavelength_px: int = WAVELENGTH_PX) -> torch.Tensor:
    """Sine midline with ~wavelength_px period in PIXEL space (not normalized
    to image width), so curvature-per-chunk is comparable across all image
    widths. amp drawn like curved_synth's random_midline_ctrl (uniform in
    [0.08, max_bend]) for a directly comparable curve strength."""
    cycles = max(1, round(width_px / wavelength_px))
    phase = rng.uniform(0, 2 * math.pi)
    amp = rng.uniform(0.08, max_bend)
    xs = torch.linspace(0.0, 1.0, width_px)
    ys = 0.5 + amp * torch.sin(2 * math.pi * cycles * xs + phase)
    return ys.clamp(0.12, 0.88)


def generate_curved_length_samples(target_len: int, n: int, max_bend: float,
                                   seed: int = 0):
    """Return [(PIL.Image, phrase), ...], real-word phrases of exact
    space-stripped length `target_len`, rendered at natural width then bent
    along a fixed-pixel-wavelength sine midline of strength `max_bend`
    (0.0 = perfectly straight, skips warping entirely so the control
    condition has zero resampling artifact). Deterministic per
    (target_len, max_bend, seed)."""
    out = []
    for i in range(n):
        seed_i = SEED_BASE + seed * 1_000_003 + target_len * 131 + int(max_bend * 1000) * 7 + i
        rng = random.Random(seed_i)
        phrase = make_phrase(rng, target_len)
        img = render_controlled(phrase)
        if max_bend > 0:
            img_t = T.ToTensor()(img)
            y_curve = _fixed_wavelength_midline(img.size[0], max_bend, rng)
            img_t = warp_along_curve(img_t, y_curve).clamp(0, 1)
            img = T.ToPILImage()(img_t)
        out.append((img, phrase))
    return out
