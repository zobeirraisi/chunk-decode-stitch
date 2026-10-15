"""Failure-case figure for chunk-decode-stitch (paper: failure-case figure).

One block per example: the aspect-preserved LTB image with the 128px crop grid
drawn on it, each crop's own read (same color as its box), then the stitched
prediction aligned character by character to the ground truth (red =
substituted or inserted, orange = missing from the prediction). Examples are
taken from nc_error_taxonomy.py's output, a fixed number per error class, in
the order the taxonomy listed them (the dump order, so selection is not
cherry-picked).

The figure is drawn at the paper's text width (5.4 in = 390 pt, elsarticle
preprint 12pt) so font sizes are print sizes; long aligned lines wrap.
render_examples() is shared with make_nc_figures.py.

Usage (from code/):
  .venv/bin/python nc_failure_figure.py --tag base --per_class 2
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from ltb_data import resize_aspect_preserving
from nc_common import CHUNK_W, align, dump_path, read_jsonl, stitch_variant

LTB_ROOT = os.path.expanduser("~/Documents/RESEARCH/STR-NEXT/data/ltb")
CLASSES = ["over-merge", "duplication", "substitution", "deletion", "mixed"]
CROP_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#e377c2"]
TEXT_W = 5.4       # inches: \textwidth of the manuscript (390 pt)
MAX_IMG_H = 0.55   # inches; narrow images are not blown up past this
PITCH_EM = 0.6     # DejaVu Sans Mono advance width, in em


def diff_runs(gt, pred):
    """Aligned prediction / ground-truth character cells; gaps shown as '-'."""
    p_line, g_line = [], []
    for op, i, j in align(gt, pred):
        if op == "=":
            p_line.append((pred[j], "black", False)); g_line.append((gt[i], "#555555", False))
        elif op == "S":
            p_line.append((pred[j], "#d62728", True)); g_line.append((gt[i], "#555555", False))
        elif op == "I":
            p_line.append((pred[j], "#d62728", True)); g_line.append(("-", "#bbbbbb", False))
        else:
            p_line.append(("-", "#ff7f0e", True)); g_line.append((gt[i], "#ff7f0e", True))
    return p_line, g_line


def render_examples(examples, out, fig_w=TEXT_W, fontsize=8.0):
    """examples: [(title, record, prediction)]. Writes out (.pdf) and a .png."""
    line_h = 1.3 * fontsize / 72.0
    pitch = PITCH_EM * fontsize / 72.0               # inches per character
    max_chars = int(fig_w / pitch)
    prefix_w = 6                                     # "pred: " / "gt:   "

    blocks = []
    for title, r, pred in examples:
        img = resize_aspect_preserving(Image.open(os.path.join(LTB_ROOT, r["id"])).convert("RGB"))
        img_h = min(fig_w * 32 / img.size[0], MAX_IMG_H)
        p_line, g_line = diff_runs(r["gt"], pred)
        step = max_chars - prefix_w
        wraps = [(p_line[k:k + step], g_line[k:k + step]) for k in range(0, len(p_line), step)]
        n_lines = len(r["xs"]) + 2 * len(wraps)
        h = 1.5 * line_h + img_h + 0.05 + line_h * n_lines + 0.6 * line_h
        blocks.append((title, r, img, img_h, wraps, h))
    H = sum(b[-1] for b in blocks)
    fig = plt.figure(figsize=(fig_w, H))

    def mono(x_chars, y_in, cells):
        x = x_chars
        for ch, color, bold in cells:
            fig.text(x * pitch / fig_w, y_in / H, ch, color=color, family="monospace",
                     fontsize=fontsize, fontweight="bold" if bold else "normal", va="top")
            x += 1

    top = H
    for title, r, img, img_h, wraps, h in blocks:
        w = img.size[0]
        fig.text(0, top / H, title, fontsize=fontsize + 0.5, fontweight="bold", va="top")
        y = top - 1.5 * line_h
        img_w = img_h * w / 32
        ax = fig.add_axes([0, (y - img_h) / H, img_w / fig_w, img_h / H])
        ax.imshow(img, extent=(0, w, 32, 0), aspect="auto")
        for k, x in enumerate(r["xs"]):
            col = CROP_COLORS[k % len(CROP_COLORS)]
            off = 1 + (k % 2) * 2  # stagger overlapping boxes
            ax.add_patch(Rectangle((x + off, off), CHUNK_W - 2 * off, 32 - 2 * off,
                                   fill=False, lw=1.0, ec=col))
        ax.set_xlim(0, w); ax.set_ylim(32, 0); ax.set_axis_off()
        y -= img_h + 0.05
        for k, (txt, _) in enumerate(r["chunks"]):
            col = CROP_COLORS[k % len(CROP_COLORS)]
            mono(0, y, [(c, col, False) for c in f"crop {k + 1}: {txt}"])
            y -= line_h
        for n, (pl, gl) in enumerate(wraps):
            mono(0, y, [(c, "black", True) for c in ("pred: " if n == 0 else "")])
            mono(prefix_w, y, pl)
            y -= line_h
            mono(0, y, [(c, "#555555", True) for c in ("gt:   " if n == 0 else "")])
            mono(prefix_w, y, gl)
            y -= line_h
        top -= h
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--stitcher", default="full", choices=["full", "conf"])
    ap.add_argument("--per_class", type=int, default=2)
    ap.add_argument("--dumps", default="results/nc/dumps")
    ap.add_argument("--taxonomy", default="results/nc/error_taxonomy_{tag}.json")
    ap.add_argument("--out", default="results/nc/figures/failure_cases_{tag}_{stitcher}.pdf")
    ap.add_argument("--fontsize", type=float, default=8.0)
    args = ap.parse_args()

    with open(args.taxonomy.format(tag=args.tag)) as f:
        tax = json.load(f)[args.stitcher]
    recs = {r["id"]: r for r in read_jsonl(dump_path(args.dumps, args.tag, "chunks", "ltb", 64))}
    examples, seen = [], set()
    for c in CLASSES:  # each image appears once, under its first class
        for ex in [e for e in tax["examples"].get(c, []) if e["id"] not in seen][: args.per_class]:
            r = recs[ex["id"]]
            n = len(r["xs"])
            title = f"{c} ({len(r['gt'])} chars, {n} crop{'s' if n > 1 else ''})"
            examples.append((title, r, stitch_variant(r["chunks"], r["xs"], **tax["kw"])[0]))
            seen.add(ex["id"])
    if not examples:
        print("no examples"); return
    out = args.out.format(tag=args.tag, stitcher=args.stitcher)
    render_examples(examples, out, fontsize=args.fontsize)
    print(f"wrote {out} ({len(examples)} examples)")


if __name__ == "__main__":
    main()
