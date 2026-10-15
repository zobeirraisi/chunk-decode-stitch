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
OUT = os.environ.get("QUAL_OUT", os.path.expanduser(
    "~/Documents/RESEARCH/STR-NEXT/paper2-longtext/neurocomputing/figures/"
    "qualitative_ltb.png"))
WIDTH = float(os.environ.get("QUAL_WIDTH", "4.79"))  # inches; supplement text width
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

    # print-size rendering: drawn at the target text width so the
    # monospace text is ~7 pt on the page; long lines wrap
    fs, fig_w = 7.0, WIDTH
    line_h, pitch = 1.3 * fs / 72.0, 0.6 * fs / 72.0
    max_chars = int(fig_w / pitch)
    blocks = []
    for (img_path, gt, pb, pf, cb, cf, bucket) in picks:
        img = resize_aspect_preserving(Image.open(img_path).convert("RGB"))
        img_h = min(fig_w * 32 / img.size[0], 0.45)
        lines = []
        for tag, text in (("GT:   ", gt), ("Base: ", pb), ("Ours: ", pf)):
            body = max_chars - len(tag)
            chunks = [text[k:k + body] for k in range(0, max(len(text), 1), body)]
            lines += [(tag if i == 0 else " " * len(tag)) + c for i, c in enumerate(chunks)]
        title = f"{len(gt)} characters: CER {cb:.0f}% (base) \u2192 {cf:.0f}% (fine-tuned)"
        blocks.append((img, img_h, title, lines, 1.4 * line_h + img_h + 0.05 + line_h * len(lines) + 0.5 * line_h))
    H = sum(b[-1] for b in blocks)
    fig = plt.figure(figsize=(fig_w, H))
    top = H
    for img, img_h, title, lines, h in blocks:
        fig.text(0, top / H, title, fontsize=fs + 0.5, fontweight="bold", va="top")
        y = top - 1.4 * line_h
        ax = fig.add_axes([0, (y - img_h) / H, (img_h * img.size[0] / 32) / fig_w, img_h / H])
        ax.imshow(img, aspect="auto"); ax.set_axis_off()
        y -= img_h + 0.05
        for ln in lines:
            fig.text(0, y / H, ln, family="monospace", fontsize=fs, va="top")
            y -= line_h
        top -= h
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
