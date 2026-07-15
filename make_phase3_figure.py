"""Paper B Phase 3 headline figure: LTB CER vs length bucket, on both the
width and both-axis conditions, across the short/long fine-tuning weight
sweep (baseline, 50/50, 70/30, 80/20). Numbers are the final, cross-checked
results from docs/paperb-phase1-3-results.md -- this script only plots
them, it does not recompute anything.

Run from code/:  .venv/bin/python make_phase3_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKETS = ["26-35", "36-55", "56+"]

# panel title -> [(series name, CER per bucket, color), ...]
# fixed categorical slot order: baseline=neutral gray, then blue/aqua/yellow
PANELS = {
    "width axis": [
        ("baseline",   [72.90, 73.08, 76.15], "#52514e"),
        ("50/50 (v1)", [63.04, 70.57, 81.63], "#2a78d6"),
        ("70/30 (v5)", [66.51, 72.70, 81.62], "#1baf7a"),
        ("80/20 (v6)", [69.79, 76.32, 81.74], "#eda100"),
    ],
    "both axes": [
        ("baseline",   [76.53, 81.57, 89.18], "#52514e"),
        ("50/50 (v1)", [63.42, 70.89, 82.02], "#2a78d6"),
        ("70/30 (v5)", [66.96, 73.14, 81.91], "#1baf7a"),
        ("80/20 (v6)", [70.34, 76.87, 82.18], "#eda100"),
    ],
}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
x = range(len(BUCKETS))

# The three fine-tuned lines converge near bucket 56+, so end-of-line labels
# there collide -- a standard legend (on the width panel, more separated) is
# the reliable choice regardless.
for ax, (title, series) in zip(axes, PANELS.items()):
    for name, ys, color in series:
        ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=6,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
                solid_capstyle="round", solid_joinstyle="round", zorder=3, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKETS)
    ax.set_xlabel("transcription length (chars)")
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlim(-0.15, len(BUCKETS) - 1 + 0.15)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)

axes[0].set_ylabel("CER (%)")
axes[0].set_ylim(55, 95)
axes[0].legend(loc="lower right", frameon=False, fontsize=9, labelcolor="linecolor")
fig.suptitle("LTB CER: fine-tuning weight sweep", fontsize=11, y=1.02)

fig.tight_layout()
fig.savefig("logs/phase3_ltb_cer.png", dpi=220, bbox_inches="tight")
fig.savefig("logs/phase3_ltb_cer.pdf", dpi=220, bbox_inches="tight")
print("wrote logs/phase3_ltb_cer.png")
