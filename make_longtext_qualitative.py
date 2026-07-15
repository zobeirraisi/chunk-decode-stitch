"""
Generate a REAL qualitative figure for the Neurocomputing long-text paper
(no fabrication). For a handful of LTB long-text images it runs the
full-convergence Union14M-L base and the recommended 80/20 fine-tune
(v6at_long) on the *both* axis -- the real end-to-end failure mode -- using
the identical resize / decode pipeline as eval_ltb.py, and renders the input
crop with ground truth, baseline prediction, and fine-tuned prediction plus
per-example CER.

Run from code/:  .venv/bin/python make_longtext_qualitative.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

from eval_ltb import (load, TO_TENSOR, MAX_DECODE_LEN, TRAIN_W_TOKENS,
                      extend_decoder_pe, levenshtein)
from ltb_data import load_bucket, resize_aspect_preserving

LTB_ROOT = os.path.expanduser("~/Documents/RESEARCH/STR-NEXT/data/ltb")
BASE_CKPT = "checkpoints/base_union14m_grid_fs8_best.pt"
FT_CKPT = "checkpoints/finetune_u14mbase_v6at_long.pt"
OUT = os.path.expanduser(
    "~/Documents/RESEARCH/STR-NEXT/paper2-longtext/neurocomputing/figures/"
    "qualitative_ltb.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt):
    model, conv, pe_type = load(ckpt)
    model.to(DEVICE).eval()
    # both checkpoints use the additive decoder table -> rebuild it to the
    # raised decode length exactly as eval_ltb.py does (scale_mode none)
    extend_decoder_pe(model, MAX_DECODE_LEN, train_len=conv.max_len + 2,
                      scale_mode="none")
    return model, conv


@torch.no_grad()
def predict_both(model, conv, img_path):
    img = Image.open(img_path).convert("RGB")
    image = resize_aspect_preserving(img)          # more encoder tokens
    tensor = TO_TENSOR(image).unsqueeze(0).to(DEVICE)
    pred = model.greedy_decode(tensor, conv.SOS, conv.EOS,
                               max_len=MAX_DECODE_LEN,
                               ntk_width_scale=None, scale_mode="ntk",
                               train_w_tokens=TRAIN_W_TOKENS,
                               dec_scale_mode="none", dec_scale=1.0)
    return conv.decode(pred[0].tolist())


def cer(pred, gt):
    return 100.0 * levenshtein(pred, gt) / max(len(gt), 1)


def main():
    print(f"device={DEVICE}")
    base, conv = load_model(BASE_CKPT)
    ft, conv_ft = load_model(FT_CKPT)

    rows = []  # (img_path, gt, pred_base, pred_ft, cer_base, cer_ft, bucket)
    for bucket in ("26-35", "36-55", "56+"):
        samples = load_bucket(LTB_ROOT, bucket)[:60]  # scan a slice per bucket
        for img_path, label in samples:
            gt = conv.clean(label)
            pb = predict_both(base, conv, img_path)
            pf = predict_both(ft, conv_ft, img_path)
            cb, cf = cer(pb, gt), cer(pf, gt)
            rows.append((img_path, gt, pb, pf, cb, cf, bucket))
        print(f"scanned bucket {bucket}")

    # pick illustrative examples: fine-tune clearly closer to GT, still legible,
    # one per bucket where possible
    picks = []
    for bucket in ("26-35", "36-55", "56+"):
        cand = [r for r in rows if r[6] == bucket and r[4] - r[5] >= 8
                and r[5] <= 60 and len(r[1]) <= 70]
        cand.sort(key=lambda r: (r[5], -(r[4] - r[5])))  # low ft-CER, big gain
        if cand:
            picks.append(cand[0])
    # top up to 5 with the largest improvements overall (dedup by path)
    extra = sorted([r for r in rows if r not in picks and r[5] <= 60],
                   key=lambda r: -(r[4] - r[5]))
    for r in extra:
        if len(picks) >= 5:
            break
        if r[0] not in {p[0] for p in picks}:
            picks.append(r)

    print(f"\nselected {len(picks)} examples:")
    for r in picks:
        print(f"  [{r[6]}] gt={r[1]!r}\n      base={r[2]!r} ({r[4]:.0f})"
              f"\n      ours={r[3]!r} ({r[5]:.0f})")

    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.72 * n))
    if n == 1:
        axes = [axes]
    for ax, (img_path, gt, pb, pf, cb, cf, bucket) in zip(axes, picks):
        img = resize_aspect_preserving(Image.open(img_path).convert("RGB"))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb")
        txt = (f"GT:   {gt}\n"
               f"Base: {pb}   (CER {cb:.0f}%)\n"
               f"Ours: {pf}   (CER {cf:.0f}%)")
        ax.text(0, -0.34, txt, transform=ax.transAxes, family="monospace",
                fontsize=7.4, va="top", ha="left", linespacing=1.35)
    fig.subplots_adjust(hspace=1.15, left=0.02, right=0.98, top=0.99,
                        bottom=0.02)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
