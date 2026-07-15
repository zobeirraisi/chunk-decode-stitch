"""Paper C figures: robustness (stride + crop-grid phase) and the long+curved
stitching test. Numbers are the final, cross-checked results from
docs/paperc-chunked-pilot.md -- this script only plots them, it does not
recompute anything (mirrors make_phase*_figure.py for Paper B).

Run from code/:  .venv/bin/python make_paperc_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../paper3-chunkeddecoding/manuscript/figures"

# shared palette / styling with Paper B's figures
GRAY = "#52514e"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
GOLD = "#eda100"
GRID = "#e6e5e0"
SPINE = "#c3c2b7"


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=GRAY, labelsize=9)


# ============================================================
# Figure 1: robustness -- stride sweep (with a real plateau) + phase sweep
# (a tight band). Two panels sharing the "A-Avg (%)" axis idea but on
# different x scales, so separate y-lims chosen to make each readable.
# ============================================================
STRIDE_PX = [32, 48, 64, 80, 96, 112]
STRIDE_OV = [75, 62.5, 50, 37.5, 25, 12.5]
STRIDE_AAVG = [43.66, 42.82, 42.79, 40.35, 35.73, 21.46]

PHASE_PX = [0, 16, 32, 48]
PHASE_AAVG = [42.79, 41.85, 42.90, 43.16]
SMTR = 41.57  # published SOTA reference line

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.7))

# -- stride panel --
axL.axhline(SMTR, color=GOLD, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
axL.text(112, SMTR + 1.0, "SMTR 41.57", color=GOLD, fontsize=8.5,
         ha="right", va="bottom")
# shade the >=50%-overlap plateau (strides <= 64)
axL.axvspan(30, 66, color=BLUE, alpha=0.07, zorder=0)
axL.plot(STRIDE_PX, STRIDE_AAVG, color=BLUE, linewidth=2, marker="o",
         markersize=6, markerfacecolor=BLUE, markeredgecolor="white",
         markeredgewidth=1.2, solid_capstyle="round", zorder=3)
for x, y in zip(STRIDE_PX, STRIDE_AAVG):
    axL.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8, color=GRAY)
axL.set_xticks(STRIDE_PX)
axL.set_xticklabels([f"{p}\n{o:g}%" for p, o in zip(STRIDE_PX, STRIDE_OV)])
axL.set_xlabel("crop stride (px)  /  overlap (%)")
axL.set_ylabel("LTB both-axis A-Avg (%)")
axL.set_title("stride sweep: a real plateau", fontsize=10, pad=8)
axL.set_ylim(18, 47)
axL.set_xlim(24, 120)
style(axL)

# -- phase panel: tight band, zoomed y --
axR.axhline(SMTR, color=GOLD, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
axR.text(48, SMTR + 0.08, "SMTR 41.57", color=GOLD, fontsize=8.5,
         ha="right", va="bottom")
axR.plot(PHASE_PX, PHASE_AAVG, color=AQUA, linewidth=2, marker="s",
         markersize=6, markerfacecolor=AQUA, markeredgecolor="white",
         markeredgewidth=1.2, solid_capstyle="round", zorder=3)
for x, y in zip(PHASE_PX, PHASE_AAVG):
    axR.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color=GRAY)
axR.set_xticks(PHASE_PX)
axR.set_xlabel("crop-grid phase (px), stride 64")
axR.set_title("phase sweep: 1.3-pt band", fontsize=10, pad=8)
axR.set_ylim(41.2, 43.9)
axR.set_xlim(-6, 54)
style(axR)

fig.suptitle("Training-free chunked decoding is robust to its only design choices",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/paperc_robustness.png", dpi=220, bbox_inches="tight")
fig.savefig(f"{OUT}/paperc_robustness.pdf", dpi=220, bbox_inches="tight")
print(f"wrote {OUT}/paperc_robustness.png")
plt.close(fig)


# ============================================================
# Figure 2: long+curved stitching. chunked acc vs length, one line per bend;
# plain decode is a flat 0.00% at every cell (shown as a reference band).
# ============================================================
LENGTHS = [30, 50, 70]
CURVED = {  # bend -> chunked acc per length
    "straight": ([94.67, 87.33, 90.67], GRAY),
    "mild":     ([92.67, 90.00, 86.00], BLUE),
    "strong":   ([92.67, 86.00, 84.00], GOLD),
}

fig, ax = plt.subplots(figsize=(5.4, 4.0))
x = range(len(LENGTHS))
# plain-decode reference (0.00% everywhere)
ax.axhspan(-2, 2, color="#d9534f", alpha=0.08, zorder=0)
ax.axhline(0, color="#d9534f", linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
ax.text(len(LENGTHS) - 1, 3.0, "plain decode 0.00% (every cell)",
        color="#b5433f", fontsize=8.5, ha="right", va="bottom")

for name, (ys, color) in CURVED.items():
    ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=6,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2,
            solid_capstyle="round", zorder=3, label=name)
ax.set_xticks(list(x))
ax.set_xticklabels([str(v) for v in LENGTHS])
ax.set_xlabel("transcription length (chars)")
ax.set_ylabel("word accuracy (%)")
ax.set_title("Chunked decoding survives long, curved text", fontsize=10.5, pad=10)
ax.set_ylim(-4, 100)
ax.set_xlim(-0.15, len(LENGTHS) - 1 + 0.15)
ax.legend(title="sine bend", loc="lower left", frameon=False, fontsize=9,
          title_fontsize=9, labelcolor="linecolor")
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/paperc_curved.png", dpi=220, bbox_inches="tight")
fig.savefig(f"{OUT}/paperc_curved.pdf", dpi=220, bbox_inches="tight")
print(f"wrote {OUT}/paperc_curved.png")
plt.close(fig)
