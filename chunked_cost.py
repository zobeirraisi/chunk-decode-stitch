"""Compute-cost accounting for chunked decoding vs. plain decoding (Paper C
pilot, docs/paperc-chunked-pilot.md caveat 4).

Chunking trades accuracy for extra forward passes: every image becomes N
crops, each run through the full encoder-decoder, vs. one encoder pass +
one (longer) decode for plain `both`-axis decoding. This measures wall-clock
directly rather than a FLOPs estimate, since the two paths have different
decode-length distributions (chunked: short, fixed budget per crop; plain:
up to MAX_DECODE_LEN, but usually shorter after autoregressive drift/EOS)
that make a clean analytic FLOPs comparison misleading -- wall-clock captures
the real cost including any early-stopping.

Usage:
  .venv/bin/python chunked_cost.py --checkpoint checkpoints/base_union14m_grid_fs8_best.pt
"""
import argparse
import os
import time

import torch
from PIL import Image

from chunked_decode import chunk_offsets, decode_chunks, stitch
from eval_ltb import extend_decoder_pe, MAX_DECODE_LEN
from eval_u14m import load
from ltb_data import load_bucket, resize_aspect_preserving

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def time_plain(model, conv, img_path, amp):
    img = Image.open(img_path).convert("RGB")
    image = resize_aspect_preserving(img)
    from dataset import IMAGENET_NORM
    from torchvision import transforms as T
    to_tensor = T.Compose([T.ToTensor(), IMAGENET_NORM])
    tensor = to_tensor(image).unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", enabled=amp and DEVICE.type == "cuda"):
        pred = model.greedy_decode(tensor, conv.SOS, conv.EOS, max_len=MAX_DECODE_LEN)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, pred.shape[1]


@torch.no_grad()
def time_chunked(model, conv, img_path, chunk_w, stride, amp):
    img = Image.open(img_path).convert("RGB")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    texts, xs, w = decode_chunks(model, conv, img, chunk_w, stride, amp)
    stitch(texts, xs, chunk_w)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ltb_root", default=os.path.expanduser(
        "~/Documents/RESEARCH/STR-NEXT/data/ltb"))
    ap.add_argument("--n", type=int, default=150,
                     help="samples per bucket to time")
    ap.add_argument("--chunk_w", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--warmup", type=int, default=5,
                     help="untimed warmup calls (CUDA context / cudnn autotune)")
    args = ap.parse_args()

    model, conv, pe_type = load(args.checkpoint)
    model.to(DEVICE).eval()
    extend_decoder_pe(model, MAX_DECODE_LEN, train_len=conv.max_len + 2,
                      scale_mode="none")

    for bucket in ("26-35", "36-55", "56+"):
        samples = load_bucket(args.ltb_root, bucket)[: args.n + args.warmup]
        warm, timed = samples[: args.warmup], samples[args.warmup:]
        for img_path, _ in warm:
            time_plain(model, conv, img_path, args.amp)
            time_chunked(model, conv, img_path, args.chunk_w, args.stride, args.amp)

        plain_t = plain_steps = chunk_t = chunk_n = 0.0
        for img_path, _ in timed:
            t, steps = time_plain(model, conv, img_path, args.amp)
            plain_t += t; plain_steps += steps
            t, n = time_chunked(model, conv, img_path, args.chunk_w, args.stride, args.amp)
            chunk_t += t; chunk_n += n
        n = max(len(timed), 1)
        print(f"{bucket:6s} (n={n}): "
              f"plain={1000*plain_t/n:6.1f}ms/img (avg {plain_steps/n:.1f} decode steps)  "
              f"chunked={1000*chunk_t/n:6.1f}ms/img (avg {chunk_n/n:.1f} crops)  "
              f"ratio={chunk_t/plain_t:.2f}x")


if __name__ == "__main__":
    main()
