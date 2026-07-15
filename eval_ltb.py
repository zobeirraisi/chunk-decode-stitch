"""Paper B Phase 1 -- LTB failure characterization ("cliff figure").

Evaluates an existing Paper-A checkpoint on the Long Text Benchmark (LTB,
transcriptions of 26-106 chars, all longer than the 25-char training cap)
under three axis conditions that isolate where length extrapolation fails:

  decoder : image squashed to the standard 32x128 (same distribution as
            training) but the decoder must emit up to 110 chars -- tests
            decoder-time extrapolation alone. Caveat: squashing 26-106 char
            text into a fixed 128px width (~1.2px/char at the long end)
            degrades raw image legibility too, so low accuracy here is not
            purely a decoder positional-table extrapolation failure.
  width   : image resized aspect-preserving (more encoder tokens than
            training saw) but ground truth is cropped to the first 25
            chars -- tests encoder-width extrapolation alone.
  both    : aspect-preserving image + full ground truth -- the real
            end-to-end failure mode.

The decoder axis needs a bigger decoder positional table than the
checkpoint was trained with; Additive1DSinPE.pe is a deterministic
(non-learned) sinusoid buffer, so extend_decoder_pe() rebuilds it at a
larger size with no retraining and no information loss for the original
positions (regression-tested below).

Run from code/:
  .venv/bin/python eval_ltb.py --checkpoint checkpoints/verify_grid_hard_best.pt
  .venv/bin/python eval_ltb.py --checkpoint checkpoints/verify_curvi_hard_best.pt
"""
import argparse
import math
import os

import torch
from PIL import Image

from dataset import IMAGENET_NORM
from eval_u14m import load
from ltb_data import LTB_BUCKETS, load_bucket, resize_aspect_preserving
from model import Additive1DSinPE, _yarn_blend
from torchvision import transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DECODE_LEN = 110  # paper2-longtext/EXPERIMENTS.md Phase 2: "raised to 110"
TRAIN_W_TOKENS = 128 // 8  # ponytail: assumes feature_stride=8, matches the
                           # same assumption already noted at the
                           # resize_aspect_preserving call sites below
TO_TENSOR = T.Compose([T.ToTensor(), IMAGENET_NORM])


def ntk_scale_for_width(w_tokens: int, train_w_tokens: int = TRAIN_W_TOKENS) -> float:
    """Dynamic per-image width-axis scale factor: how many multiples of the
    training encoder width this image's token count is. Never below 1.0
    (narrower-than-training images get no scaling, same as no-op). Despite
    the name (kept for backward compatibility), this scale factor feeds any
    of the three training-free scale_mode variants (ntk/pi/yarn), not only
    NTK-aware rescaling."""
    return max(w_tokens / train_w_tokens, 1.0)


def extend_decoder_pe(model, new_max_len: int, train_len: int = 27,
                      scale_mode: str = "none") -> None:
    """Rebuild the decoder's additive sinusoidal PE table to cover longer
    sequences, optionally with a training-free length-extrapolation scaling
    applied to it -- the decoder-axis analogue of the width-axis Phase 2
    ``scale_mode`` on the 2D-rotary encoder (proposal item 3).

    ``scale_mode``:
      - ``none``: plain longer sinusoid table (naive extrapolation; the model
        sees position indices it never trained on -> the decoder-axis cliff).
      - ``pi``:  position interpolation. Compress test positions into the
        trained range by dividing the argument by ``scale = new_max_len /
        train_len`` before the sinusoid, so no position exceeds what training
        saw. Loses absolute character pitch resolution.
      - ``ntk``: NTK-aware base rescaling. Grow the sinusoid base
        ``10000 -> 10000 * scale^(d/(d-2))`` so the low-frequency (long-range)
        dimensions stretch while the high-frequency (local) dimensions are
        almost untouched.
      - ``yarn``: NTK-by-parts blend (reuses the encoder's ``_yarn_blend``):
        interpolate the low-frequency dims, extrapolate the high-frequency
        dims, with a linear ramp between.

    The additive sinusoid's per-dim frequencies ``div[k] = 10000^(-2k/d)`` have
    the same structure as the rotary ``inv_freq``, so the three modes are the
    exact additive-PE counterparts of the rotary ones used on the width axis.
    ``train_len`` is the decode length the model actually trained on
    (SOS + max_len chars + EOS)."""
    d_model = model.tgt_pe.pe.shape[-1]
    device = model.tgt_pe.pe.device
    max_len = new_max_len + 4
    scale = max(new_max_len / train_len, 1.0)
    k = torch.arange(0, d_model, 2).float()
    inv_freq = torch.exp(k * (-math.log(10000.0) / d_model))  # == Additive1DSinPE div
    pos = torch.arange(0, max_len).float().unsqueeze(1)

    if scale_mode == "pi":
        freqs = (pos / scale) * inv_freq
    elif scale_mode == "ntk":
        base = 10000.0 * scale ** (d_model / (d_model - 2))
        inv_freq_ntk = torch.exp(k * (-math.log(base) / d_model))
        freqs = pos * inv_freq_ntk
    elif scale_mode == "yarn":
        freqs = pos * _yarn_blend(inv_freq, scale, train_len)
    elif scale_mode == "none":
        freqs = pos * inv_freq
    else:
        raise ValueError(f"unknown decoder scale_mode {scale_mode!r}")

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(freqs)
    pe[:, 1::2] = torch.cos(freqs)
    new_pe = Additive1DSinPE(d_model, max_len=max_len).to(device)
    new_pe.pe = pe.unsqueeze(0).to(device)
    model.tgt_pe = new_pe


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insertions/deletions/substitutions, unit cost)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


@torch.no_grad()
def eval_sample(model, conv, device, img_path: str, label: str, axis: str,
                 max_decode_len: int, amp: bool,
                 use_ntk: bool = False, scale_mode: str = "ntk",
                 dec_scale_mode: str = "none",
                 dec_scale: float = 1.0) -> tuple[bool, int, int, int]:
    img = Image.open(img_path).convert("RGB")
    ntk_width_scale = None
    if axis == "decoder":
        image = img.resize((128, 32), Image.BILINEAR)
        decode_len = max_decode_len
    elif axis == "width":
        # ponytail: assumes feature_stride=8 (both current checkpoints are
        # fs8); thread cfg["feature_stride"] through if a non-fs8 checkpoint
        # is ever added
        image = resize_aspect_preserving(img)
        decode_len = conv.max_len + 2
        if use_ntk:
            ntk_width_scale = ntk_scale_for_width(image.size[0] // 8)
    elif axis == "both":
        # ponytail: assumes feature_stride=8 (both current checkpoints are
        # fs8); thread cfg["feature_stride"] through if a non-fs8 checkpoint
        # is ever added
        image = resize_aspect_preserving(img)
        decode_len = max_decode_len
        if use_ntk:
            ntk_width_scale = ntk_scale_for_width(image.size[0] // 8)
    else:
        raise ValueError(f"unknown axis {axis!r}")

    # decoder-time rotary scaling only bites when the decode length exceeds the
    # trained one (decoder/both axes); on the width axis decode_len == train len
    dsm = dec_scale_mode if axis in ("decoder", "both") else "none"
    tensor = TO_TENSOR(image).unsqueeze(0).to(device)
    with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
        pred = model.greedy_decode(tensor, conv.SOS, conv.EOS, max_len=decode_len,
                                    ntk_width_scale=ntk_width_scale, scale_mode=scale_mode,
                                    train_w_tokens=TRAIN_W_TOKENS,
                                    dec_scale_mode=dsm, dec_scale=dec_scale)
    pred_str = conv.decode(pred[0].tolist())
    gt_str = conv.clean(label)
    if axis == "width":
        gt_str = gt_str[: conv.max_len]
    return (pred_str == gt_str, len(pred_str),
            levenshtein(pred_str, gt_str), len(gt_str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ltb_root", default=os.path.expanduser(
        "~/Documents/RESEARCH/STR-NEXT/data/ltb"))
    ap.add_argument("--axis", choices=["decoder", "width", "both", "all"],
                     default="all")
    ap.add_argument("--buckets", nargs="+", default=["26-35", "36-55", "56+"],
                     choices=list(LTB_BUCKETS))
    ap.add_argument("--limit", type=int, default=0,
                     help="cap samples per bucket (0 = all)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ntk_width", action="store_true",
                     help="dynamic training-free width-axis frequency rescaling "
                          "(width/both axes only; no-op for the decoder axis "
                          "and for non-2drope checkpoints); variant chosen by "
                          "--scale_mode")
    ap.add_argument("--scale_mode", choices=["ntk", "pi", "yarn"], default="ntk",
                     help="which training-free rescaling to apply when "
                          "--ntk_width is set: ntk (NTK-aware base rescaling), "
                          "pi (position interpolation), yarn (NTK-by-parts blend)")
    ap.add_argument("--decoder_scale_mode",
                     choices=["none", "ntk", "pi", "yarn"], default="none",
                     help="training-free length-extrapolation scaling applied "
                          "to the decoder's additive sinusoidal PE (decoder/both "
                          "axes; the decoder-axis analogue of --scale_mode). "
                          "none = plain longer table (naive extrapolation).")
    args = ap.parse_args()

    model, conv, pe_type = load(args.checkpoint)
    model.to(DEVICE).eval()
    train_dec_len = conv.max_len + 2  # SOS + max_len chars + EOS
    rotary_decoder = getattr(model, "dec_self_pe", "none") == "1drope"
    # decoder-time rotary scale (only used by rotary-decoder checkpoints)
    dec_scale = max(MAX_DECODE_LEN / train_dec_len, 1.0)
    axes = ["decoder", "width", "both"] if args.axis == "all" else [args.axis]
    if ("decoder" in axes or "both" in axes) and not rotary_decoder:
        # additive-PE decoder: rebuild the sinusoid table (optionally scaled).
        # Rotary-decoder checkpoints carry decode-time position in-attention, so
        # they need no table rebuild; their scaling is applied per-call instead.
        extend_decoder_pe(model, MAX_DECODE_LEN, train_len=train_dec_len,
                          scale_mode=args.decoder_scale_mode)

    print(f"checkpoint={args.checkpoint} pe_type={pe_type} "
          f"rotary_decoder={rotary_decoder}")
    for axis in axes:
        tag = f" [{args.scale_mode}]" if args.ntk_width else ""
        if args.decoder_scale_mode != "none" and axis in ("decoder", "both"):
            tag += f" [dec:{args.decoder_scale_mode}]"
        print(f"\n=== axis: {axis}{tag} ===")
        for bucket in args.buckets:
            samples = load_bucket(args.ltb_root, bucket)
            if args.limit:
                samples = samples[: args.limit]
            # for rotary-decoder checkpoints the scale mode is applied in-attention
            dsm = args.decoder_scale_mode if rotary_decoder else "none"
            results = [
                eval_sample(model, conv, DEVICE, img_path, label, axis,
                            MAX_DECODE_LEN, args.amp, use_ntk=args.ntk_width,
                            scale_mode=args.scale_mode,
                            dec_scale_mode=dsm, dec_scale=dec_scale)
                for img_path, label in samples
            ]
            correct = sum(r[0] for r in results)
            mean_pred_len = sum(r[1] for r in results) / max(len(results), 1)
            acc = 100.0 * correct / max(len(samples), 1)
            # Corpus-level CER: total edit distance / total reference length
            # (standard convention; exact-match floors to ~0% past training
            # length, CER is what shows the actual degradation gradient).
            cer = 100.0 * sum(r[2] for r in results) / max(sum(r[3] for r in results), 1)
            print(f"  {bucket:6s} (n={len(samples):4d}): acc={acc:5.2f}% "
                  f"cer={cer:5.2f}% mean_pred_len={mean_pred_len:.1f}")


if __name__ == "__main__":
    main()
