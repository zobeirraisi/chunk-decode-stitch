"""Training-free chunked decoding for long text (Paper C pilot).

The Table-`sota` diagnosis (paper2-longtext manuscript S4.8/S4.9): the absolute
LTB exact-match gap to SMTR/LISTER is decoding-side -- a vanilla AR decoder must
emit one long in-order sequence and drifts/loops past the training horizon, and
the failure is NOT fixable from the positional-encoding side. Every method above
us in that table avoids long AR emission (SMTR: overlapping substrings + stitch;
LISTER: neighbour decoding; FocalSVTR: CTC).

This module implements SMTR's principle in *image space*, as a pure
inference-time algorithm on the unmodified checkpoints:

  1. Aspect-preserving resize to the trained height (32px), like the `both`
     axis in eval_ltb.py -- but never feed the wide image to the model.
  2. Slide overlapping crops of the trained width (128px) across it. Every
     crop is in-distribution on BOTH axes: trained token count, and a text
     span short enough for the trained decode horizon.
  3. Decode each crop with plain greedy decoding (one batch per image).
  4. Stitch the chunk transcriptions left-to-right with a geometry-anchored
     overlap merge: pixel overlap between consecutive crops predicts roughly
     how many characters they share; an edit-distance search around that
     expectation finds the best suffix/prefix alignment; the merged overlap
     takes its left half from the left chunk (whose right edge is the
     unreliable part of the *right* chunk's view) and its right half from the
     right chunk, so characters are preferentially read from crop interiors.

Eval protocol matches eval_ltb.py's `both` axis (aspect-preserved image, full
ground truth, corpus CER + exact-match word accuracy) so rows are directly
comparable to the manuscript's Table `sota` and `logs/decoderpe_table/`.

Usage:
  .venv/bin/python chunked_decode.py \
      --checkpoint checkpoints/finetune_u14mbase_v6at_long.pt --amp
"""
import argparse
import os

import torch
from PIL import Image

from dataset import IMAGENET_NORM
from eval_ltb import extend_decoder_pe, levenshtein
from eval_u14m import load
from ltb_data import LTB_BUCKETS, load_bucket, resize_aspect_preserving
from torchvision import transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TO_TENSOR = T.Compose([T.ToTensor(), IMAGENET_NORM])


def chunk_offsets(width: int, chunk_w: int, stride: int, phase: int = 0) -> list[int]:
    """Left x of each crop. Standard right-aligned final window: if the last
    strided window would run past the image, replace it with one flush against
    the right edge, so every pixel is covered and no crop is narrower than
    chunk_w. `phase` (0 <= phase < stride) shifts the interior grid boundaries
    while keeping x=0 and the right-aligned final window fixed (both required
    for full coverage); this is the training-free analogue of a "seed" for an
    otherwise fully deterministic algorithm -- it asks whether the arbitrary
    choice of where crop boundaries fall changes the outcome."""
    if width <= chunk_w:
        return [0]
    phase = phase % max(stride, 1)
    xs = [0] + list(range(phase if phase > 0 else stride, width - chunk_w, stride))
    if xs[-1] != width - chunk_w:
        xs.append(width - chunk_w)
    return xs


@torch.no_grad()
def decode_chunks(model, conv, img: Image.Image, chunk_w: int, stride: int,
                  amp: bool, chunk_decode_len: int = 30, phase: int = 0):
    """Aspect-preserving resize, crop, batch-decode. Returns
    (chunk_texts, offsets, resized_width)."""
    resized = resize_aspect_preserving(img)
    w = resized.size[0]
    xs = chunk_offsets(w, chunk_w, stride, phase)
    crops = []
    for x in xs:
        crop = resized.crop((x, 0, min(x + chunk_w, w), 32))
        if crop.size[0] < chunk_w:  # only possible for a single narrow image
            crop = crop.resize((chunk_w, 32), Image.BILINEAR)
        crops.append(TO_TENSOR(crop))
    batch = torch.stack(crops).to(DEVICE)
    with torch.autocast("cuda", enabled=amp and DEVICE.type == "cuda"):
        preds = model.greedy_decode(batch, conv.SOS, conv.EOS,
                                    max_len=chunk_decode_len)
    return [conv.decode(p.tolist()) for p in preds], xs, w


def merge_pair(acc: str, nxt: str, ov_px: int, chunk_w: int,
               min_score: float = 0.4, slack: int = 3) -> str:
    """Merge the next chunk's text into the accumulated transcription.

    ov_px/chunk_w is the fraction of the next chunk's view already covered by
    the previous crop, so both texts should share roughly that fraction of
    their characters. Search overlap length k around that geometric
    expectation, score each k by normalized suffix/prefix edit similarity,
    and merge at the best k (falling back to the expectation when nothing
    aligns, e.g. both reads garbled). k=0 concatenates."""
    if not nxt:
        return acc
    if not acc:
        return nxt
    frac = ov_px / chunk_w
    k_exp = round(frac * len(nxt))
    lo = max(0, k_exp - slack - int(0.25 * k_exp))
    hi = min(len(acc), len(nxt), k_exp + slack + int(0.25 * k_exp))
    best_k, best_score = None, -1.0
    for k in range(lo, hi + 1):
        if k == 0:
            continue
        score = 1.0 - levenshtein(acc[-k:], nxt[:k]) / k
        # prefer k nearest the geometric expectation on near-ties
        score -= 0.01 * abs(k - k_exp)
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None or best_score < min_score:
        best_k = min(k_exp, len(acc), len(nxt))
    half = best_k // 2
    return acc[:len(acc) - best_k + half] + nxt[half:]


def stitch(chunk_texts: list[str], xs: list[int], chunk_w: int) -> str:
    acc = chunk_texts[0]
    for i in range(1, len(chunk_texts)):
        ov_px = max(xs[i - 1] + chunk_w - xs[i], 0)
        acc = merge_pair(acc, chunk_texts[i], ov_px, chunk_w)
    return acc


@torch.no_grad()
def chunked_read(model, conv, img_path: str, chunk_w: int, stride: int,
                 amp: bool, phase: int = 0) -> tuple[str, list[str]]:
    img = Image.open(img_path).convert("RGB")
    texts, xs, _ = decode_chunks(model, conv, img, chunk_w, stride, amp, phase=phase)
    return stitch(texts, xs, chunk_w), texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ltb_root", default=os.path.expanduser(
        "~/Documents/RESEARCH/STR-NEXT/data/ltb"))
    ap.add_argument("--buckets", nargs="+", default=["26-35", "36-55", "56+"],
                     choices=list(LTB_BUCKETS))
    ap.add_argument("--limit", type=int, default=0,
                     help="cap samples per bucket (0 = all)")
    ap.add_argument("--chunk_w", type=int, default=128,
                     help="crop width in pixels (= training width)")
    ap.add_argument("--stride", type=int, default=64,
                     help="crop stride in pixels (64 = 50%% overlap)")
    ap.add_argument("--phase", type=int, default=0,
                     help="crop-grid phase shift in pixels (0 <= phase < stride); "
                          "the training-free analogue of a seed, since decoding "
                          "is otherwise fully deterministic")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--show", type=int, default=0,
                     help="print this many qualitative samples per bucket "
                          "(chunk texts + stitched vs GT)")
    args = ap.parse_args()

    model, conv, pe_type = load(args.checkpoint)
    model.to(DEVICE).eval()
    # Chunk decode budget (chunk_decode_len, default 30) can exceed a base
    # checkpoint's trained decoder-PE table (max_len 25 -> table size 29);
    # grow it (scale_mode "none": these are still in-distribution positions,
    # just past this particular checkpoint's short table, not an
    # extrapolation regime) so the crash mode above can't recur.
    chunk_decode_len = 30
    if getattr(model, "dec_self_pe", "none") != "1drope":
        extend_decoder_pe(model, chunk_decode_len, train_len=conv.max_len + 2,
                          scale_mode="none")
    print(f"checkpoint={args.checkpoint} pe_type={pe_type} "
          f"chunk_w={args.chunk_w} stride={args.stride} phase={args.phase}")
    print(f"\n=== axis: both [chunked w{args.chunk_w} s{args.stride} p{args.phase}] ===")
    for bucket in args.buckets:
        samples = load_bucket(args.ltb_root, bucket)
        if args.limit:
            samples = samples[: args.limit]
        correct = edit_total = ref_total = pred_len_total = 0
        for i, (img_path, label) in enumerate(samples):
            pred, chunk_texts = chunked_read(model, conv, img_path,
                                             args.chunk_w, args.stride, args.amp,
                                             phase=args.phase)
            gt = conv.clean(label)
            if i < args.show:
                print(f"  [show {bucket}] gt  ={gt!r}")
                print(f"  [show {bucket}] pred={pred!r}")
                print(f"  [show {bucket}] chunks={chunk_texts}")
            correct += pred == gt
            edit_total += levenshtein(pred, gt)
            ref_total += len(gt)
            pred_len_total += len(pred)
        n = max(len(samples), 1)
        acc = 100.0 * correct / n
        cer = 100.0 * edit_total / max(ref_total, 1)
        print(f"  {bucket:6s} (n={len(samples):4d}): acc={acc:5.2f}% "
              f"cer={cer:5.2f}% mean_pred_len={pred_len_total / n:.1f}")


if __name__ == "__main__":
    main()
