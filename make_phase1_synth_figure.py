"""Paper B Phase 1 (deferred item) headline figure: continuous CER-vs-length
curve on the controlled synthetic set (synth_controlled_length.py /
eval_synth_length.py), baseline vs the Phase 3 fine-tuned checkpoint
(finetune_longtext_grid_v6.pt), on all three axes. Complements the coarse
3-bucket LTB tables (make_phase1_figure.py, make_phase3_figure.py) with a
low-noise (n=150/point), finely-spaced (11 points, 26-100 chars) curve.
Numbers are read straight from logs/eval_synth_length_base.log and
logs/eval_synth_length_v6.log -- this script only plots them, it does not
recompute anything.

This is NOT a strict positive-news figure: the decoder axis shows the
fine-tuned checkpoint is BETTER near the training cap (26-35 chars) but
WORSE than baseline in the 40-75 char range (peaking at CER > 100% at
length 56, i.e. more inserted/substituted characters than the reference is
long -- repetition-loop territory). This is reported as-is, not smoothed
over; it sharpens (does not contradict) Phase 3's existing "decoder axis
does not improve under any variant tried" finding into "and can actively
regress in the middle of the tested range."

Run from code/:  .venv/bin/python make_phase1_synth_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LENGTHS = [26, 30, 35, 40, 45, 50, 56, 65, 75, 90, 100]
BASE_COLOR = "#52514e"  # neutral gray: baseline, matches phase2/phase3 convention
V6_COLOR = "#2a78d6"    # first categorical slot: fine-tuned (80/20) variant

# axis -> (baseline CER, v6 CER) per length -- from logs/eval_synth_length_{base,v6}.log
PANELS = {
    "decoder axis": (
        [42.62, 56.04, 65.57, 74.76, 75.69, 76.13, 79.01, 79.89, 79.99, 83.11, 83.62],
        [32.13, 47.52, 65.36, 79.69, 86.66, 90.98, 101.93, 97.36, 90.73, 85.54, 83.44],
    ),
    "width axis": (
        [63.79, 65.47, 67.44, 68.19, 70.27, 69.79, 69.23, 72.45, 73.55, 73.68, 74.03],
        [51.14, 48.39, 56.96, 65.39, 61.91, 65.94, 64.73, 72.23, 71.19, 74.38, 76.60],
    ),
    "both axes": (
        [64.46, 70.34, 75.00, 78.13, 81.14, 82.74, 84.53, 86.59, 88.34, 90.07, 90.97],
        [51.76, 48.74, 57.71, 66.56, 62.60, 66.98, 65.26, 73.21, 71.44, 74.53, 76.63],
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0), sharex=True)

for ax, (title, (base, v6)) in zip(axes, PANELS.items()):
    ax.plot(LENGTHS, base, color=BASE_COLOR, linewidth=2, marker="o", markersize=5,
            markerfacecolor=BASE_COLOR, markeredgecolor="white", markeredgewidth=1.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=3, label="baseline")
    ax.plot(LENGTHS, v6, color=V6_COLOR, linewidth=2, marker="o", markersize=5,
            markerfacecolor=V6_COLOR, markeredgecolor="white", markeredgewidth=1.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=3, label="fine-tuned (80/20)")
    ax.axvline(25, color="#c3c2b7", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlabel("transcription length (chars)")
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)

axes[0].set_ylabel("CER (%)")
axes[0].annotate("training cap", (25, 1.0), xycoords=("data", "axes fraction"),
                  fontsize=8, color="#52514e", ha="right", va="top",
                  rotation=90, xytext=(-4, -4), textcoords="offset points")
axes[2].legend(loc="lower right", frameon=False, fontsize=9, labelcolor="linecolor")
fig.suptitle("Controlled synthetic length curve: baseline vs fine-tuned", fontsize=11, y=1.02)

fig.tight_layout()
fig.savefig("logs/phase1_synth_length_curve.png", dpi=220, bbox_inches="tight")
fig.savefig("logs/phase1_synth_length_curve.pdf", dpi=220, bbox_inches="tight")
print("wrote logs/phase1_synth_length_curve.png")
