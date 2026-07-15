"""Paper B Phase 1 (deferred item) -- controlled-length synthetic eval.

Same three-axis decomposition as eval_ltb.py (decoder / width / both), but
on synth_controlled_length.py's generated phrases instead of the 4,789 real
LTB images. Purpose: a continuous, low-noise length curve (many samples per
length point, finely spaced) to complement LTB's three coarse, real-image
buckets -- the proposal's risk mitigation for "LTB too small/quirky"
(PROPOSAL.md, Risks table).

Run from code/:
  .venv/bin/python eval_synth_length.py --checkpoint checkpoints/verify_grid_hard_best.pt
"""
import argparse

import torch
from torchvision import transforms as T
from PIL import Image

from dataset import IMAGENET_NORM
from eval_ltb import MAX_DECODE_LEN, extend_decoder_pe, levenshtein, ntk_scale_for_width
from eval_u14m import load
from ltb_data import resize_aspect_preserving
from synth_controlled_length import generate_length_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TO_TENSOR = T.Compose([T.ToTensor(), IMAGENET_NORM])
LENGTHS = [26, 30, 35, 40, 45, 50, 56, 65, 75, 90, 100]


@torch.no_grad()
def eval_sample(model, conv, device, img, label: str, axis: str,
                 max_decode_len: int, amp: bool, use_ntk: bool = False):
    ntk_width_scale = None
    if axis == "decoder":
        image = img.resize((128, 32), Image.BILINEAR)
        decode_len = max_decode_len
    elif axis in ("width", "both"):
        image = resize_aspect_preserving(img)
        decode_len = max_decode_len if axis == "both" else conv.max_len + 2
        if use_ntk:
            ntk_width_scale = ntk_scale_for_width(image.size[0] // 8)
    else:
        raise ValueError(f"unknown axis {axis!r}")

    tensor = TO_TENSOR(image).unsqueeze(0).to(device)
    with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
        pred = model.greedy_decode(tensor, conv.SOS, conv.EOS, max_len=decode_len,
                                    ntk_width_scale=ntk_width_scale)
    pred_str = conv.decode(pred[0].tolist())
    gt_str = conv.clean(label)
    if axis == "width":
        gt_str = gt_str[: conv.max_len]
    return (pred_str == gt_str, levenshtein(pred_str, gt_str), len(gt_str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--axis", choices=["decoder", "width", "both", "all"], default="all")
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--n", type=int, default=150, help="samples per length point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ntk_width", action="store_true")
    args = ap.parse_args()

    model, conv, pe_type = load(args.checkpoint)
    model.to(DEVICE).eval()
    axes = ["decoder", "width", "both"] if args.axis == "all" else [args.axis]
    if "decoder" in axes or "both" in axes:
        extend_decoder_pe(model, MAX_DECODE_LEN)

    print(f"checkpoint={args.checkpoint} pe_type={pe_type}")
    for axis in axes:
        tag = " [ntk]" if args.ntk_width else ""
        print(f"\n=== axis: {axis}{tag} ===")
        for length in args.lengths:
            samples = generate_length_samples(length, args.n, seed=args.seed)
            results = [
                eval_sample(model, conv, DEVICE, img, label, axis,
                            MAX_DECODE_LEN, args.amp, use_ntk=args.ntk_width)
                for img, label in samples
            ]
            correct = sum(r[0] for r in results)
            acc = 100.0 * correct / max(len(results), 1)
            cer = 100.0 * sum(r[1] for r in results) / max(sum(r[2] for r in results), 1)
            print(f"  len={length:3d} (n={len(results):4d}): acc={acc:5.2f}% cer={cer:5.2f}%")


if __name__ == "__main__":
    main()
