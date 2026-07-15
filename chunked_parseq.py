"""chunk-decode-stitch on the PUBLIC PARSeq checkpoint (torch.hub baudm/parseq).

Architecture-generality check for the Pattern Recognition paper: the manuscript
claims the algorithm turns *any* word-level autoregressive STR checkpoint into
a long-text reader, but every experiment uses our own 2D-RoPE-STR checkpoints.
This script applies the identical crop grid + stitcher (imported unmodified
from chunked_decode.py) to the public PARSeq base model, whose published plain
both-axis LTB word accuracy is 0.00 across all buckets (Du et al., Table:
PARSeq row). Protocol matches chunked_decode.py exactly: aspect-preserving
resize to height 32, 128px crops, per-crop decode, 36-charset scoring.

The ONLY model-specific parts are (a) PARSeq's 0.5/0.5 normalization instead
of ImageNet stats and (b) its own tokenizer decode; crop gridding, stitching,
and scoring are byte-identical to the 2D-RoPE-STR runs.

Usage:
  .venv/bin/python chunked_parseq.py --amp [--stride 64] [--limit 0]
"""
import argparse
import os

import torch
from PIL import Image
from torchvision import transforms as T

from chunked_decode import chunk_offsets, stitch
from dataset import LabelConverter
from eval_ltb import levenshtein
from ltb_data import LTB_BUCKETS, load_bucket, resize_aspect_preserving

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PARSeq's training normalization (not ImageNet stats).
PARSEQ_TF = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])


@torch.no_grad()
def decode_chunks_parseq(model, conv, img: Image.Image, chunk_w: int,
                         stride: int, amp: bool, phase: int = 0):
    resized = resize_aspect_preserving(img)
    w = resized.size[0]
    xs = chunk_offsets(w, chunk_w, stride, phase)
    crops = []
    for x in xs:
        crop = resized.crop((x, 0, min(x + chunk_w, w), 32))
        if crop.size[0] < chunk_w:  # only possible for a single narrow image
            crop = crop.resize((chunk_w, 32), Image.BILINEAR)
        crops.append(PARSEQ_TF(crop))
    batch = torch.stack(crops).to(DEVICE)
    with torch.autocast("cuda", enabled=amp and DEVICE.type == "cuda"):
        logits = model(batch)
    labels, _ = model.tokenizer.decode(logits.softmax(-1))
    # Clean each chunk read to the 36-char protocol BEFORE stitching, exactly
    # as the 2D-RoPE-STR checkpoints' native charset already guarantees.
    return [conv.clean(t) for t in labels], xs, w


@torch.no_grad()
def chunked_read_parseq(model, conv, img_path: str, chunk_w: int, stride: int,
                        amp: bool, phase: int = 0):
    img = Image.open(img_path).convert("RGB")
    texts, xs, _ = decode_chunks_parseq(model, conv, img, chunk_w, stride, amp,
                                        phase=phase)
    return stitch(texts, xs, chunk_w), texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltb_root", default=os.path.expanduser(
        "~/Documents/RESEARCH/STR-NEXT/data/ltb"))
    ap.add_argument("--buckets", nargs="+", default=["26-35", "36-55", "56+"],
                    choices=list(LTB_BUCKETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk_w", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True,
                           trust_repo=True).to(DEVICE).eval()
    conv = LabelConverter()  # 36-char cleaning only; no encode/decode used
    print(f"model=parseq(torch.hub) chunk_w={args.chunk_w} "
          f"stride={args.stride} phase={args.phase}")
    print(f"\n=== axis: both [chunked w{args.chunk_w} s{args.stride} "
          f"p{args.phase}] ===")
    accs = []
    for bucket in args.buckets:
        samples = load_bucket(args.ltb_root, bucket)
        if args.limit:
            samples = samples[: args.limit]
        correct = edit_total = ref_total = pred_len_total = 0
        for i, (img_path, label) in enumerate(samples):
            pred, chunk_texts = chunked_read_parseq(
                model, conv, img_path, args.chunk_w, args.stride, args.amp,
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
        accs.append((acc, len(samples)))
        print(f"  {bucket:6s} (n={len(samples):4d}): acc={acc:5.2f}% "
              f"cer={cer:5.2f}% mean_pred_len={pred_len_total / n:.1f}")
    if len(accs) == len(LTB_BUCKETS):
        a_avg = sum(a for a, _ in accs) / len(accs)
        w_avg = sum(a * n for a, n in accs) / sum(n for _, n in accs)
        print(f"  A-Avg={a_avg:5.2f}%  W-Avg={w_avg:5.2f}%")


if __name__ == "__main__":
    main()
