"""
2D-RoPE-STR model and positional-encoding variants.

A single `STRModel` implements the full CNN -> Transformer-encoder ->
autoregressive-decoder pipeline. The encoder's positional encoding is fully
pluggable via `pe_type`, so the ablation study compares strictly identical
architectures that differ only in how position is injected:

    none | 1d_sin | 1d_learn | 2d_sin | 2d_learn | 1drope | 2drope

RoPE variants ('1drope', '2drope') apply rotation *inside* the attention
(to queries and keys), which is the mathematically correct formulation.
Additive variants add the encoding to the tokens before the encoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from chart import grid_to_uv


# ============================================================
# CNN backbone with configurable output stride
# ============================================================
def _disable_stride(layer):
    """Set the spatial stride of a ResNet stage's first block to 1."""
    block = layer[0]
    block.conv2.stride = (1, 1)            # 3x3 conv carries the stride (ResNet v1.5)
    if block.downsample is not None:
        block.downsample[0].stride = (1, 1)


def build_backbone(feature_stride: int = 8, pretrained: bool = True):
    """ResNet-50 truncated before avgpool/fc, with reduced total stride so the
    feature map keeps genuine 2D spatial extent (default /8: 32x128 -> 4x16)."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    net = resnet50(weights=weights)
    if feature_stride <= 16:
        _disable_stride(net.layer4)
    if feature_stride <= 8:
        _disable_stride(net.layer3)
    if feature_stride <= 4:
        _disable_stride(net.layer2)
    backbone = nn.Sequential(*list(net.children())[:-2])  # -> (B, 2048, H', W')
    return backbone, 2048


# ============================================================
# Rotary position embeddings
# ============================================================
class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device, scale_mode="none", scale=1.0,
                train_len=None):
        """Optional training-free length-extrapolation scaling (Paper B
        decoder-time axis): ``pi`` position-interpolates, ``ntk`` rescales the
        base, ``yarn`` blends per NTK-by-parts. Defaults (``none``/1.0) leave
        the rotary untouched, so training and any non-scaled call are exact."""
        inv_freq = self.inv_freq.to(device).float()
        if scale_mode != "none" and scale > 1.0:
            if scale_mode == "pi":
                inv_freq = inv_freq / scale
            elif scale_mode == "ntk":
                base = 10000.0 * scale ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (base ** (torch.arange(
                    0, self.dim, 2, device=device).float() / self.dim))
            elif scale_mode == "yarn":
                assert train_len is not None, "yarn scale_mode needs train_len"
                inv_freq = _yarn_blend(inv_freq, scale, train_len)
            else:
                raise ValueError(f"unknown scale_mode {scale_mode!r}")
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # (N, dim/2) complex


def _yarn_blend(inv_freq, scale, train_w_tokens, beta_fast=32.0, beta_slow=1.0):
    """Standard YaRN "NTK-by-parts" correction range (Peng et al. 2024,
    Sec 3.2), with beta_fast/beta_slow at the paper's own defaults. Returns
    a new inv_freq tensor; never mutates the input. `inv_freq` is indexed by
    dimension-pair j=0..n-1 with inv_freq[j] = base^(-2j/dim), monotonically
    decreasing in j, so log(inv_freq) is linear in j -- used below to invert
    "number of full rotations over the training range" back into a j index."""
    n = inv_freq.numel()
    log_inv_freq = torch.log(inv_freq)
    j = torch.arange(n, device=inv_freq.device).float()
    log_step = (log_inv_freq[-1] - log_inv_freq[0]) / max(n - 1, 1)

    def _dim_for_num_rotations(num_rotations):
        # j at which inv_freq(j) * train_w_tokens / (2*pi) == num_rotations
        target = num_rotations * 2 * math.pi / train_w_tokens
        return (math.log(target) - log_inv_freq[0].item()) / log_step.item()

    low = max(math.floor(_dim_for_num_rotations(beta_fast)), 0)
    high = min(math.ceil(_dim_for_num_rotations(beta_slow)), n - 1)
    if low == high:
        high += 1e-3
    ramp = ((j - low) / (high - low)).clamp(0.0, 1.0)
    extrapolation_mask = 1.0 - ramp  # 1 at low j (high-freq/local): keep original; 0 at high j (low-freq/long-range): fully interpolate
    inv_freq_interp = inv_freq / scale
    return inv_freq_interp * (1.0 - extrapolation_mask) + inv_freq * extrapolation_mask


class RotaryEmbedding2D(nn.Module):
    """Half of the head dimension is rotated by the row index, half by the
    column index. Optional learnable per-axis frequency scales (beta) and
    configurable per-axis frequency bases (theta). Defaults (base 10000 on both
    axes) reproduce the original buffer layout exactly, so older checkpoints
    that store only `inv_freq`/`beta_*` still load."""

    def __init__(self, head_dim, learnable_scale=False,
                 base_h=10000.0, base_w=10000.0, dim_split=0.5):
        super().__init__()
        self.base_h = float(base_h)
        self.base_w = float(base_w)
        # Per-axis real-dim counts, matching apply_2d_rope's split EXACTLY so the
        # frequency table length equals the number of complex pairs on each axis.
        d_h = int(head_dim * dim_split)
        if d_h % 2:
            d_h += 1
        d_w = head_dim - d_h

        def _inv(base, d):
            return 1.0 / (base ** (torch.arange(0, d, 2).float() / d))

        # Non-persistent: follow .to(device) but stay out of the state_dict, so
        # checkpoints remain compatible across any split/base choice.
        self.register_buffer("inv_freq_h", _inv(base_h, d_h), persistent=False)
        self.register_buffer("inv_freq_w", _inv(base_w, d_w), persistent=False)
        # Legacy buffer kept only so older checkpoints that stored `inv_freq`
        # (equal split, base 10000) still load strictly.
        self.register_buffer("inv_freq", _inv(10000.0, head_dim // 2))
        if learnable_scale:
            self.beta_h = nn.Parameter(torch.ones(1))
            self.beta_w = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("beta_h", torch.ones(1))
            self.register_buffer("beta_w", torch.ones(1))

    def forward(self, h_pos, w_pos, ntk_width_scale=None, scale_mode="ntk",
                train_w_tokens=None):
        h_freqs = torch.einsum("i,j->ij", h_pos.float() * self.beta_h.float(), self.inv_freq_h).float()
        inv_freq_w = self.inv_freq_w
        if ntk_width_scale is not None and ntk_width_scale > 1.0:
            scale = ntk_width_scale
            d_w = inv_freq_w.numel() * 2
            if scale_mode == "ntk":
                # NTK-aware frequency-base rescaling (training-free): stretches
                # the low-frequency (long-range) dimensions to cover the wider
                # token range while leaving high-frequency (local relative
                # position) dimensions nearly unchanged. Computed fresh each
                # call -- never mutates self.inv_freq_w or the checkpoint.
                scaled_base_w = self.base_w * (scale ** (d_w / (d_w - 2)))
                exponent = torch.arange(0, d_w, 2, device=w_pos.device).float() / d_w
                inv_freq_w = 1.0 / (scaled_base_w ** exponent)
            elif scale_mode == "pi":
                # Position interpolation (Chen et al. 2023): compress the
                # wider token range back into the training range by
                # dividing every frequency uniformly, rather than touching
                # the frequency base -- trades local resolution for range.
                inv_freq_w = inv_freq_w / scale
            elif scale_mode == "yarn":
                # YaRN (Peng et al. 2024), frequency-domain component only:
                # NTK-by-parts blend of extrapolation (unchanged inv_freq,
                # for high-frequency/local dims, via a linear ramp over
                # wavelength) and interpolation (inv_freq/scale, for
                # low-frequency/long-range dims). We omit YaRN's attention-
                # temperature correction, which rescales attention logits
                # rather than the rotary table and would require changing
                # the attention call rather than only the frequency table.
                assert train_w_tokens is not None, "yarn scale_mode needs train_w_tokens"
                inv_freq_w = _yarn_blend(inv_freq_w, scale, train_w_tokens)
        w_freqs = torch.einsum("i,j->ij", w_pos.float() * self.beta_w.float(), inv_freq_w).float()
        h_freqs = torch.polar(torch.ones_like(h_freqs), h_freqs)  # (H, d_h/2)
        w_freqs = torch.polar(torch.ones_like(w_freqs), w_freqs)  # (W, d_w/2)
        return h_freqs, w_freqs


def apply_1d_rope(x, freqs):
    """x: (B, heads, N, head_dim); freqs: (N, head_dim/2) complex."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs.unsqueeze(0).unsqueeze(0)).flatten(-2)
    return out.type_as(x)


def apply_2d_rope(x, h_freqs, w_freqs, H, W, dim_split=0.5):
    """x: (B, heads, N, head_dim) with N = H*W (row-major)."""
    B, nh, N, D = x.shape
    x = x.view(B, nh, H, W, D)
    d_h = int(D * dim_split)
    if d_h % 2:  # complex pairing needs an even split
        d_h += 1
    x_h, x_w = x[..., :d_h], x[..., d_h:]
    x_h = torch.view_as_complex(x_h.float().reshape(*x_h.shape[:-1], -1, 2))
    x_w = torch.view_as_complex(x_w.float().reshape(*x_w.shape[:-1], -1, 2))
    # freqs: (H, d_h/2) and (W, d_w/2) -> broadcast over (B, heads, H, W)
    hf = h_freqs[:, : x_h.shape[-1]].unsqueeze(0).unsqueeze(0).unsqueeze(3)  # (1,1,H,1,d_h/2)
    wf = w_freqs[:, : x_w.shape[-1]].unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1,1,1,W,d_w/2)
    x_h = torch.view_as_real(x_h * hf).flatten(-2)
    x_w = torch.view_as_real(x_w * wf).flatten(-2)
    out = torch.cat([x_h, x_w], dim=-1).view(B, nh, N, D)
    return out.type_as(x)


# ============================================================
# Continuous chart rotary (CurviRoPE): per-token (u,v) positions
# ============================================================
class RotaryChart2D(nn.Module):
    """Like RotaryEmbedding2D but positions are per-token floats (u,v), not
    shared integer axes. u gets the larger dim share (reading axis)."""

    def __init__(self, head_dim, learnable_scale=False,
                 base_u=10000.0, base_v=10000.0, dim_split=0.75):
        super().__init__()
        d_u = int(head_dim * dim_split)
        if d_u % 2:
            d_u += 1
        d_v = head_dim - d_u

        def _inv(base, d):
            return 1.0 / (base ** (torch.arange(0, d, 2).float() / d))

        self.register_buffer("inv_freq_u", _inv(base_u, d_u), persistent=False)
        self.register_buffer("inv_freq_v", _inv(base_v, d_v), persistent=False)
        if learnable_scale:
            self.beta_u = nn.Parameter(torch.ones(1))
            self.beta_v = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("beta_u", torch.ones(1))
            self.register_buffer("beta_v", torch.ones(1))

    def forward(self, u_pos, v_pos):
        # u_pos, v_pos: (B, N) -> freqs (B, N, d_/2) complex
        # .float() after einsum: under autocast the einsum output is fp16, but
        # torch.polar has no CUDA Half kernel — force fp32 (matches RotaryEmbedding2D).
        uf = torch.einsum("bn,j->bnj", u_pos.float() * self.beta_u.float(),
                          self.inv_freq_u.to(u_pos.device)).float()
        vf = torch.einsum("bn,j->bnj", v_pos.float() * self.beta_v.float(),
                          self.inv_freq_v.to(v_pos.device)).float()
        return (torch.polar(torch.ones_like(uf), uf),
                torch.polar(torch.ones_like(vf), vf))


def apply_chart_rope(x, u_freqs, v_freqs, dim_split=0.75):
    """x: (B, heads, N, head_dim); u_freqs/v_freqs: (B, N, d_/2) complex."""
    B, nh, N, D = x.shape
    d_u = int(D * dim_split)
    if d_u % 2:
        d_u += 1
    x_u, x_v = x[..., :d_u], x[..., d_u:]
    x_u = torch.view_as_complex(x_u.float().reshape(*x_u.shape[:-1], -1, 2))
    x_v = torch.view_as_complex(x_v.float().reshape(*x_v.shape[:-1], -1, 2))
    uf = u_freqs[..., : x_u.shape[-1]].unsqueeze(1)   # (B,1,N,d_u/2)
    vf = v_freqs[..., : x_v.shape[-1]].unsqueeze(1)   # (B,1,N,d_v/2)
    x_u = torch.view_as_real(x_u * uf).flatten(-2)
    x_v = torch.view_as_real(x_v * vf).flatten(-2)
    return torch.cat([x_u, x_v], dim=-1).type_as(x)


# ============================================================
# Chart head: Bézier midline + half-width from backbone feats
# ============================================================
class ChartHead(nn.Module):
    """Predicts a text-aligned Bézier midline (n_ctrl control points) + a scalar
    half-width from the pooled backbone feature map. Kept tiny (<0.3M params):
    global-avg-pool -> 1x1 bottleneck -> two small heads.

    Control points are biased toward a left-to-right horizontal line at init so
    early training starts near the grid chart (well-conditioned soft-projection)."""

    def __init__(self, in_ch=2048, n_ctrl=8, hidden=64):
        super().__init__()
        self.n_ctrl = n_ctrl
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.ctrl = nn.Linear(hidden, n_ctrl * 2)
        self.hw = nn.Linear(hidden, 1)
        # init: horizontal midline y=0.5, x evenly 0..1 ; half-width ~0.5
        nn.init.zeros_(self.ctrl.weight)
        xs = torch.linspace(0, 1, n_ctrl)
        bias = torch.stack([xs, torch.full((n_ctrl,), 0.5)], -1).reshape(-1)
        # invert the sigmoid used in forward so sigmoid(bias) ~= target
        self.ctrl.bias.data = torch.logit(bias.clamp(1e-3, 1 - 1e-3))
        nn.init.zeros_(self.hw.weight)
        nn.init.zeros_(self.hw.bias)     # sigmoid(0)=0.5

    def forward(self, feat):
        z = self.reduce(feat)                        # (B, hidden)
        cp = torch.sigmoid(self.ctrl(z)).view(-1, self.n_ctrl, 2)  # (B,K,2) in [0,1]
        hw = torch.sigmoid(self.hw(z)).squeeze(-1)   # (B,) in (0,1)
        return cp, hw


# ============================================================
# Additive (non-rotary) positional encodings
# ============================================================
class Additive1DSinPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, H, W):
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class Additive2DSinPE(nn.Module):
    def __init__(self, d_model, max_h=64, max_w=128):
        super().__init__()
        d_half = d_model // 2

        def sinusoid(length, dim):
            pe = torch.zeros(length, dim)
            pos = torch.arange(0, length).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        self.register_buffer("pe_h", sinusoid(max_h, d_half))
        self.register_buffer("pe_w", sinusoid(max_w, d_half))

    def forward(self, x, H, W):
        B, N, D = x.shape
        d_half = D // 2
        x = x.view(B, H, W, D).clone()
        x[..., :d_half] = x[..., :d_half] + self.pe_h[:H].to(x.dtype).unsqueeze(1)
        x[..., d_half:] = x[..., d_half:] + self.pe_w[:W].to(x.dtype).unsqueeze(0)
        return x.view(B, N, D)


class Learnable1DPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x, H, W):
        return x + self.pe[:, : x.size(1)]


class Learnable2DPE(nn.Module):
    def __init__(self, d_model, max_h=64, max_w=128):
        super().__init__()
        self.pe_h = nn.Parameter(torch.randn(1, max_h, d_model // 2) * 0.02)
        self.pe_w = nn.Parameter(torch.randn(1, max_w, d_model // 2) * 0.02)

    def forward(self, x, H, W):
        B, N, D = x.shape
        d_half = D // 2
        x = x.view(B, H, W, D).clone()
        x[..., :d_half] = x[..., :d_half] + self.pe_h[:, :H].unsqueeze(2)
        x[..., d_half:] = x[..., d_half:] + self.pe_w[:, :W].unsqueeze(1)
        return x.view(B, N, D)


ROPE_TYPES = {"1drope", "2drope"}
ADDITIVE_TYPES = {"1d_sin", "1d_learn", "2d_sin", "2d_learn"}
CHART_TYPES = {"curvirope"}


# ============================================================
# Encoder with RoPE-aware attention
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, rope=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        if rope is not None:
            q, k = rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope=None):
        x = x + self.dropout(self.attn(self.norm1(x), rope))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Decoder layer with optional RoPE in cross-attention keys
# ============================================================
class CrossRoPEDecoderLayer(nn.Module):
    """Pre-norm decoder layer: masked self-attention, then cross-attention to the
    encoder memory, then FFN. The cross-attention can rotate the memory *keys*
    by their position (1D flattened index or 2D row/col grid); queries (text
    tokens) are left unrotated, matching the paper's "RoPE in cross-attention
    keys" formulation. `cross_pe="none"` is the no-PE control."""

    def __init__(self, d_model, n_heads, d_ff, dropout, cross_pe, dim_split,
                 self_pe="none"):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cross_pe = cross_pe
        self.self_pe = self_pe
        self.dim_split = dim_split

        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.cross_out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if cross_pe == "2drope":
            self.rope = RotaryEmbedding2D(self.head_dim, dim_split=dim_split)
        elif cross_pe == "1drope":
            self.rope = RotaryEmbedding1D(self.head_dim)
        else:
            self.rope = None

        # Decoder-time (self-attention) rotary: rotates the causal-self-attention
        # queries and keys by their decode step, so relative decode-time distance
        # is encoded rotationally instead of via an absolute additive table --
        # this is what extrapolates natively to longer sequences and accepts
        # YaRN scaling (Paper B decoder-time axis, proposal item 3).
        self.self_rope = RotaryEmbedding1D(self.head_dim) if self_pe == "1drope" else None

    def _self_attn(self, x, tgt_mask, self_freqs=None):
        B, T, D = x.shape
        qkv = self.self_qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if self_freqs is not None:
            q = apply_1d_rope(q, self_freqs)
            k = apply_1d_rope(k, self_freqs)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=tgt_mask)
        return self.self_out(out.transpose(1, 2).contiguous().view(B, T, D))

    def _cross_attn(self, x, memory, H, W):
        B, T, D = x.shape
        S = memory.size(1)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(memory).view(B, S, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)  # each (B, heads, S, head_dim)
        if self.cross_pe == "2drope":
            h_freqs, w_freqs = self.rope(torch.arange(H, device=x.device),
                                         torch.arange(W, device=x.device))
            k = apply_2d_rope(k, h_freqs, w_freqs, H, W, self.dim_split)
        elif self.cross_pe == "1drope":
            k = apply_1d_rope(k, self.rope(S, x.device))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.cross_out(out.transpose(1, 2).contiguous().view(B, T, D))

    def forward(self, x, memory, tgt_mask, H, W, self_freqs=None):
        x = x + self.dropout(self._self_attn(self.norm1(x), tgt_mask, self_freqs))
        x = x + self.dropout(self._cross_attn(self.norm2(x), memory, H, W))
        x = x + self.ffn(self.norm3(x))
        return x


# ============================================================
# Full STR model
# ============================================================
class STRModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 dropout=0.1, pe_type="2drope", learnable_freq=False,
                 dim_split=0.5, feature_stride=8, pretrained_backbone=True,
                 max_len=27, rope_base_h=10000.0, rope_base_w=10000.0,
                 cross_pe=None, n_ctrl=8, dec_self_pe="none"):
        super().__init__()
        self.pe_type = pe_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dim_split = dim_split
        self.cross_pe = cross_pe
        # Decoder-time positional encoding on the self-attention: "none" keeps
        # the legacy additive sinusoidal target PE; "1drope" replaces it with
        # rotary self-attention (Paper B decoder-time axis).
        self.dec_self_pe = dec_self_pe
        self.train_dec_len = max_len + 2  # SOS + max_len chars + EOS

        self.cnn, cnn_channels = build_backbone(feature_stride, pretrained_backbone)
        self.cnn_proj = nn.Linear(cnn_channels, d_model)

        # Positional encoding module(s)
        self.add_pe = None
        self.rope = None
        self.chart_head = None
        if pe_type in ADDITIVE_TYPES:
            self.add_pe = {
                "1d_sin": Additive1DSinPE, "1d_learn": Learnable1DPE,
                "2d_sin": Additive2DSinPE, "2d_learn": Learnable2DPE,
            }[pe_type](d_model)
        elif pe_type == "1drope":
            self.rope = RotaryEmbedding1D(self.head_dim)
        elif pe_type == "2drope":
            self.rope = RotaryEmbedding2D(self.head_dim, learnable_scale=learnable_freq,
                                          base_h=rope_base_h, base_w=rope_base_w,
                                          dim_split=dim_split)
        elif pe_type == "curvirope":
            self.chart_head = ChartHead(in_ch=cnn_channels, n_ctrl=n_ctrl)
            self.rope = RotaryChart2D(self.head_dim, learnable_scale=learnable_freq,
                                      dim_split=dim_split)
            # per-layer grid<->chart mix gate, stored as logit so sigmoid()=0.5 at init
            self.gamma = nn.Parameter(torch.zeros(num_encoder_layers))

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder positional encoding on the target sequence.
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        # Rotary decoder self-attention carries decode-time position itself, so
        # the additive absolute table is dropped in that mode.
        self.tgt_pe = (None if dec_self_pe == "1drope"
                       else Additive1DSinPE(d_model, max_len=max_len + 4))
        if cross_pe is None and dec_self_pe == "none":
            # Legacy path: standard Transformer decoder (no PE on cross-attn).
            dec_layer = nn.TransformerDecoderLayer(
                d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
            self.cross_decoder = None
        else:
            # Custom layers that can rotate cross-attention keys and/or the
            # decoder self-attention (decoder-time RoPE).
            self.decoder = None
            self.cross_decoder = nn.ModuleList([
                CrossRoPEDecoderLayer(d_model, n_heads, d_ff, dropout, cross_pe,
                                      dim_split, self_pe=dec_self_pe)
                for _ in range(num_decoder_layers)
            ])
        self.generator = nn.Linear(d_model, vocab_size)

    def _build_rope_fn(self, H, W, device, ntk_width_scale=None, scale_mode="ntk",
                        train_w_tokens=None):
        if self.pe_type == "2drope":
            h_pos = torch.arange(H, device=device)
            w_pos = torch.arange(W, device=device)
            h_freqs, w_freqs = self.rope(h_pos, w_pos, ntk_width_scale=ntk_width_scale,
                                          scale_mode=scale_mode, train_w_tokens=train_w_tokens)

            def fn(q, k):
                q = apply_2d_rope(q, h_freqs, w_freqs, H, W, self.dim_split)
                k = apply_2d_rope(k, h_freqs, w_freqs, H, W, self.dim_split)
                return q, k
            return fn
        if self.pe_type == "1drope":
            freqs = self.rope(H * W, device)

            def fn(q, k):
                return apply_1d_rope(q, freqs), apply_1d_rope(k, freqs)
            return fn
        return None

    def _chart_uv(self, feats_map, oracle=None):
        """feats_map: (B, C, H, W) -> per-token (u, v) mixed with normalized grid,
        one (u,v) per encoder layer via its own gamma. Returns list of (u,v).
        oracle=(control_pts, half_width) bypasses the chart head with a known
        ground-truth chart (Phase-2 oracle-chart upper-bound experiment)."""
        B, C, H, W = feats_map.shape
        if oracle is not None:
            cp, hw = oracle
        else:
            cp, hw = self.chart_head(feats_map)
        self._last_hw = hw          # grad-connected; trainer reads it for the hw regularizer
        u_chart, v_chart = grid_to_uv(cp, hw, H, W)          # (B, H*W) each
        # normalized grid coords in the SAME (u in[0,1], v in[-1,1]) ranges
        ys = torch.linspace(0, 1, H, device=feats_map.device)
        xs = torch.linspace(0, 1, W, device=feats_map.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        u_grid = gx.reshape(-1)[None].expand(B, -1)        # column = reading axis, in [0,1]
        v_grid = (gy * 2 - 1).reshape(-1)[None].expand(B, -1)  # row, in [-1,1]
        mixes = []
        for g in torch.sigmoid(self.gamma):
            u = g * u_chart + (1 - g) * u_grid
            v = g * v_chart + (1 - g) * v_grid
            mixes.append((u, v))
        return mixes

    def encode(self, images, oracle=None, ntk_width_scale=None, scale_mode="ntk",
               train_w_tokens=None):
        feats = self.cnn(images)                      # (B, C, H, W)
        feats_map = feats                             # keep 2D map for chart head
        B, C, H, W = feats.shape
        self._mem_hw = (H, W)                          # needed by cross-attn RoPE
        feats = feats.flatten(2).transpose(1, 2)      # (B, H*W, C), row-major
        feats = self.cnn_proj(feats)
        if self.add_pe is not None:
            feats = self.add_pe(feats, H, W)
        if self.pe_type == "curvirope":
            mixes = self._chart_uv(feats_map, oracle=oracle)
            x = feats
            for layer, (u, v) in zip(self.encoder_layers, mixes):
                uf, vf = self.rope(u, v)

                def fn(q, k, uf=uf, vf=vf):
                    return (apply_chart_rope(q, uf, vf, self.dim_split),
                            apply_chart_rope(k, uf, vf, self.dim_split))
                x = layer(x, fn)
            return self.encoder_norm(x)
        rope_fn = self._build_rope_fn(H, W, images.device, ntk_width_scale=ntk_width_scale,
                                       scale_mode=scale_mode, train_w_tokens=train_w_tokens)
        x = feats
        for layer in self.encoder_layers:
            x = layer(x, rope_fn)
        return self.encoder_norm(x)

    @torch.no_grad()
    def chart_coords(self, images):
        """Predicted (u,v) for the chart itself (gamma=1), for warp-consistency
        and visualization. Returns (u, v) each (B, H*W)."""
        feats = self.cnn(images)
        cp, hw = self.chart_head(feats)
        return grid_to_uv(cp, hw, feats.shape[2], feats.shape[3])

    def chart_coords_grad(self, images):
        """Grad-enabled twin of chart_coords, used by the warp-consistency loss
        (Task 6) which needs gradients through the chart head."""
        feats = self.cnn(images)
        cp, hw = self.chart_head(feats)
        return grid_to_uv(cp, hw, feats.shape[2], feats.shape[3])

    def decode(self, tgt_input, memory, dec_scale_mode="none", dec_scale=1.0):
        T = tgt_input.size(1)
        x = self.tgt_embed(tgt_input)
        tgt = x if self.tgt_pe is None else self.tgt_pe(x, 0, 0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_input.device)
        if self.cross_decoder is None:
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        else:
            H, W = self._mem_hw
            self_freqs = None
            if self.dec_self_pe == "1drope":
                # one shared rotary table for all decoder layers
                self_freqs = self.cross_decoder[0].self_rope(
                    T, tgt_input.device, scale_mode=dec_scale_mode,
                    scale=dec_scale, train_len=self.train_dec_len)
            out = tgt
            for layer in self.cross_decoder:
                out = layer(out, memory, tgt_mask, H, W, self_freqs)
        return self.generator(out)

    def forward(self, images, tgt_input, oracle=None, ntk_width_scale=None,
                scale_mode="ntk", train_w_tokens=None,
                dec_scale_mode="none", dec_scale=1.0):
        memory = self.encode(images, oracle=oracle, ntk_width_scale=ntk_width_scale,
                              scale_mode=scale_mode, train_w_tokens=train_w_tokens)
        return self.decode(tgt_input, memory, dec_scale_mode, dec_scale)

    @torch.no_grad()
    def greedy_decode(self, images, sos_idx, eos_idx, max_len=27, oracle=None,
                       ntk_width_scale=None, scale_mode="ntk", train_w_tokens=None,
                       dec_scale_mode="none", dec_scale=1.0):
        memory = self.encode(images, oracle=oracle, ntk_width_scale=ntk_width_scale,
                              scale_mode=scale_mode, train_w_tokens=train_w_tokens)
        B = images.size(0)
        seq = torch.full((B, 1), sos_idx, dtype=torch.long, device=images.device)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)
        for _ in range(max_len):
            logits = self.decode(seq, memory, dec_scale_mode, dec_scale)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            finished |= nxt.squeeze(1) == eos_idx
            if finished.all():
                break
        return seq[:, 1:]  # drop the SOS token
