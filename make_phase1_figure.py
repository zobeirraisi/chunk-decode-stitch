"""Paper B Phase 1 headline figure: the LTB length-extrapolation "cliff",
decomposed into decoder-axis / width-axis / both-axis CER across length
buckets. Numbers are the final, cross-checked results from
docs/paperb-phase1-3-results.md (Phase 1 table) -- this script only plots
them, it does not recompute anything.

Run from code/:  .venv/bin/python make_phase1_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKETS = ["26-35", "36-55", "56+"]

# name -> (CER per bucket, color, linewidth) -- fixed categorical slot order
SERIES = [
    ("decoder axis", [70.63, 82.73, 84.89], "#2a78d6", 2.0),
    ("width axis",   [72.90, 73.08, 76.15], "#1baf7a", 2.0),
    ("both axes (end-to-end)", [76.53, 81.57, 89.18], "#eda100", 2.6),
]

fig, ax = plt.subplots(figsize=(5.8, 3.8))
x = range(len(BUCKETS))

for name, ys, color, lw in SERIES:
    ax.plot(x, ys, color=color, linewidth=lw, marker="o", markersize=6,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
            solid_capstyle="round", solid_joinstyle="round", zorder=3, label=name)

ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor="linecolor")

ax.set_xticks(list(x))
ax.set_xticklabels(BUCKETS)
ax.set_xlabel("transcription length (chars)")
ax.set_ylabel("CER (%)")
ax.set_title("LTB failure decomposition: width dominates decoder", fontsize=11, pad=12)
ax.set_xlim(-0.15, len(BUCKETS) - 1 + 0.15)
ax.set_ylim(65, 92)
ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#c3c2b7")
ax.tick_params(colors="#52514e", labelsize=9)

fig.tight_layout()
fig.savefig("logs/phase1_ltb_cliff.png", dpi=220)
fig.savefig("logs/phase1_ltb_cliff.pdf", dpi=220)
print("wrote logs/phase1_ltb_cliff.png")
