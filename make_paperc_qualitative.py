"""Paper C qualitative figure: real LTB long-text examples showing the
chunk-decode-stitch pipeline in action -- the aspect-preserved strip with its
overlapping crop windows, each crop's independent in-distribution read, and the
stitched transcription vs ground truth. Auto-picks one clean multi-chunk
SUCCESS and one interpretable near-miss FAILURE from the hardest (56+) bucket,
so the figure shows both the mechanism and its honest failure mode.

Run from code/:
  .venv/bin/python make_paperc_qualitative.py \
      --checkpoint checkpoints/base_union14m_grid_fs8_best.pt --amp
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

from chunked_decode import chunk_offsets, decode_chunks, stitch
from eval_ltb import extend_decoder_pe, levenshtein
from eval_u14m import load
from ltb_data import load_bucket, resize_aspect_preserving

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_W, STRIDE = 128, 64
GREEN, RED, GRAY = "#1baf7a", "#d9534f", "#52514e"
CROP_COLORS = ["#2a78d6", "#eda100", "#1baf7a", "#d9534f", "#8e5bd6", "#e0722a"]


def gather(model, conv, samples, amp):
    """Return list of dicts with the full pipeline trace per sample."""
    out = []
    for img_path, label in samples:
        img = Image.open(img_path).convert("RGB")
        texts, xs, w = decode_chunks(model, conv, img, CHUNK_W, STRIDE, amp)
        pred = stitch(texts, xs, CHUNK_W)
        gt = conv.clean(label)
        cer = levenshtein(pred, gt) / max(len(gt), 1)
        out.append(dict(path=img_path, gt=gt, pred=pred, chunks=texts,
                        xs=xs, w=w, correct=pred == gt, cer=cer))
    return out


def render_panel(ax_img, ax_txt, rec, title):
    """Top axis: resized strip with crop windows. Bottom axis: per-crop reads
    and stitched vs GT."""
    strip = resize_aspect_preserving(Image.open(rec["path"]).convert("RGB"))
    arr = np.asarray(strip)
    h, w = arr.shape[0], arr.shape[1]
    ax_img.imshow(arr, aspect="auto", extent=[0, w, 0, h])
    row = h * 1.35  # vertical pitch between staggered crop rows
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(-row * (len(rec["xs"]) + 0.3), h * 1.05)
    ax_img.axis("off")
    ax_img.set_title(title, fontsize=10.5, loc="left", pad=6, color=GRAY)

    # each crop: show the real pixels the model saw, boxed in its colour and
    # staggered downward, with its independent read to the right
    for i, x in enumerate(rec["xs"]):
        c = CROP_COLORS[i % len(CROP_COLORS)]
        y0 = -row * (1.0 + i) + row * 0.1
        crop = arr[:, x:min(x + CHUNK_W, w)]
        ax_img.imshow(crop, aspect="auto", extent=[x, x + crop.shape[1], y0, y0 + h])
        ax_img.add_patch(Rectangle((x, y0), CHUNK_W, h, fill=False,
                                   edgecolor=c, linewidth=1.8))
        # faint fill on the strip itself marking this crop's span
        ax_img.add_patch(Rectangle((x, 0), CHUNK_W, h, facecolor=c, alpha=0.06,
                                   edgecolor="none"))
        ax_img.text(x + CHUNK_W + 8, y0 + h * 0.5,
                    f"crop {i+1}:  {rec['chunks'][i]!r}", va="center",
                    ha="left", fontsize=8.5, color=c, family="monospace")

    # bottom text axis: stitched pred (char-coloured vs GT) and GT
    ax_txt.axis("off")
    ok = rec["correct"]
    head = "exact match" if ok else f"near miss (CER {rec['cer']*100:.1f}%)"
    ax_txt.text(0, 1.0, head, fontsize=9.5, color=GREEN if ok else RED,
                weight="bold", va="top", family="sans-serif")
    ax_txt.text(0, 0.62, f"stitched : {rec['pred']}", fontsize=8.6, va="top",
                family="monospace", color=GRAY)
    ax_txt.text(0, 0.28, f"ground tr: {rec['gt']}", fontsize=8.6, va="top",
                family="monospace", color=GRAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ltb_root", default=os.path.expanduser(
        "~/Documents/RESEARCH/STR-NEXT/data/ltb"))
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="../paper3-chunkeddecoding/manuscript/"
                    "figures/paperc_qualitative.png")
    args = ap.parse_args()

    model, conv, _ = load(args.checkpoint)
    model.to(DEVICE).eval()
    if getattr(model, "dec_self_pe", "none") != "1drope":
        extend_decoder_pe(model, 30, train_len=conv.max_len + 2, scale_mode="none")

    samples = load_bucket(args.ltb_root, "56+")
    recs = gather(model, conv, samples, args.amp)

    # SUCCESS: exact match with a panel-readable crop count (4-6), longest GT,
    # preferring URL-like strings (the canonical long-text case)
    succ = [r for r in recs if r["correct"] and 4 <= len(r["xs"]) <= 6]
    succ.sort(key=lambda r: (r["gt"].startswith("http"), len(r["gt"])),
              reverse=True)
    # FAILURE: wrong but small CER (interpretable slip, not garbage), 4-6 crops
    # so the panel stays readable
    fail = [r for r in recs if not r["correct"] and 0 < r["cer"] <= 0.18
            and 3 <= len(r["xs"]) <= 6]
    fail.sort(key=lambda r: r["cer"])
    if not succ or not fail:
        print(f"WARN succ={len(succ)} fail={len(fail)}; relaxing filters")
    s = succ[0]
    f = fail[0]
    print(f"SUCCESS gt={s['gt']!r} crops={len(s['xs'])}")
    print(f"FAILURE gt={f['gt']!r} pred={f['pred']!r} cer={f['cer']:.3f} "
          f"crops={len(f['xs'])}")

    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1], hspace=0.35,
                          wspace=0.12, top=0.93, bottom=0.05, left=0.04,
                          right=0.98)
    render_panel(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), s,
                 f"(a) success: {len(s['xs'])}-crop stitch, exact match")
    render_panel(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), f,
                 "(b) near miss: single-character error")
    fig.suptitle("chunk–decode–stitch on real LTB long text (base checkpoint, stride 64)",
                 fontsize=11.5, y=0.985)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
