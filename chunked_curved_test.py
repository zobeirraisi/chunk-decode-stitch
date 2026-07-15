"""The genuinely open question from docs/paperc-chunked-pilot.md: does chunk
STITCHING degrade on long text that bends across chunk boundaries, as opposed
to LTB's overwhelmingly straight horizontal signage? Union14M-B's Curve/
Multi-Oriented subsets couldn't test this (short captions, almost always a
single chunk -- see chunked_u14m_nonhorizontal.py). This uses
curved_longtext_synth.py's long+curved real-word-phrase generator instead:
multiple chunks are guaranteed (fixed-pixel-wavelength curvature, long text),
and ground truth is exact (synthetic), so word accuracy is a clean,
noise-free signal at every (length, curve strength) combination.

Usage:
  .venv/bin/python chunked_curved_test.py \
      --checkpoint checkpoints/base_union14m_grid_fs8_best.pt --amp
"""
import argparse

import torch

from chunked_decode import decode_chunks, stitch
from curved_longtext_synth import generate_curved_length_samples
from dataset import IMAGENET_NORM
from eval_ltb import extend_decoder_pe, levenshtein, MAX_DECODE_LEN
from eval_u14m import load
from ltb_data import resize_aspect_preserving
from torchvision import transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TO_TENSOR = T.Compose([T.ToTensor(), IMAGENET_NORM])
BEND_LEVELS = [("straight", 0.0), ("mild", 0.15), ("strong", 0.28)]
LENGTHS = [30, 50, 70]  # spans LTB's 26-35/36-55/56+ bucket range


@torch.no_grad()
def plain_read(model, conv, img, amp):
    image = resize_aspect_preserving(img)
    tensor = TO_TENSOR(image).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", enabled=amp and DEVICE.type == "cuda"):
        pred = model.greedy_decode(tensor, conv.SOS, conv.EOS, max_len=MAX_DECODE_LEN)
    return conv.decode(pred[0].tolist())


@torch.no_grad()
def chunked_read_img(model, conv, img, chunk_w, stride, amp):
    texts, xs, _ = decode_chunks(model, conv, img, chunk_w, stride, amp)
    return stitch(texts, xs, chunk_w), len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=100, help="samples per (length, bend) cell")
    ap.add_argument("--chunk_w", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    model, conv, pe_type = load(args.checkpoint)
    model.to(DEVICE).eval()
    extend_decoder_pe(model, MAX_DECODE_LEN, train_len=conv.max_len + 2,
                      scale_mode="none")
    print(f"checkpoint={args.checkpoint} pe_type={pe_type} n={args.n} "
          f"chunk_w={args.chunk_w} stride={args.stride}")
    print(f"{'length':>6} {'bend':>9} | {'plain_acc':>9} {'plain_cer':>9} | "
          f"{'chunk_acc':>9} {'chunk_cer':>9} | {'delta_acc':>9}")
    for length in LENGTHS:
        for label, bend in BEND_LEVELS:
            samples = generate_curved_length_samples(length, args.n, bend, seed=0)
            p_correct = c_correct = 0
            p_edit = p_ref = c_edit = c_ref = 0
            for img, phrase in samples:
                gt = conv.clean(phrase)
                p_pred = plain_read(model, conv, img, args.amp)
                c_pred, _ = chunked_read_img(model, conv, img, args.chunk_w,
                                             args.stride, args.amp)
                p_correct += p_pred == gt
                c_correct += c_pred == gt
                p_edit += levenshtein(p_pred, gt); p_ref += len(gt)
                c_edit += levenshtein(c_pred, gt); c_ref += len(gt)
            n = len(samples)
            p_acc, c_acc = 100 * p_correct / n, 100 * c_correct / n
            p_cer = 100 * p_edit / max(p_ref, 1)
            c_cer = 100 * c_edit / max(c_ref, 1)
            print(f"{length:>6} {label:>9} | {p_acc:9.2f} {p_cer:9.2f} | "
                  f"{c_acc:9.2f} {c_cer:9.2f} | {c_acc - p_acc:+9.2f}")


if __name__ == "__main__":
    main()
