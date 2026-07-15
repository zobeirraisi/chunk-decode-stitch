"""Paper B Phase 2 headline figure: training-free rescaling on the width and
both-axis LTB conditions, baseline vs. NTK-aware / position-interpolation
(PI) / YaRN rescaling. NTK stays within 1pp of baseline everywhere (a null
result); PI and YaRN -- which, unlike NTK, also compress the local/
high-frequency rotary dimensions -- give a modest, consistent 2-4pp CER
reduction on the both-axis condition, smaller than fine-tuning's 6-13pp
(Phase 3) but clearly non-null. Numbers are the final, cross-checked results
from docs/paperb-phase1-3-results.md (Phase 2 table) -- this script only
plots them, it does not recompute anything.

Run from code/:  .venv/bin/python make_phase2_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKETS = ["26-35", "36-55", "56+"]

# panel title -> [(series name, CER per bucket, color), ...]
# fixed categorical slot order, consistent with make_phase3_figure.py:
# baseline=neutral gray, then blue/aqua/yellow for the three variants
PANELS = {
    "width axis": [
        ("baseline", [72.90, 73.08, 76.15], "#52514e"),
        ("+NTK",     [73.06, 73.56, 76.90], "#2a78d6"),
        ("+PI",      [71.52, 74.83, 77.47], "#1baf7a"),
        ("+YaRN",    [71.55, 72.95, 75.23], "#eda100"),
    ],
    "both axes": [
        ("baseline", [76.53, 81.57, 89.18], "#52514e"),
        ("+NTK",     [76.32, 81.16, 88.63], "#2a78d6"),
        ("+PI",      [72.68, 78.72, 86.99], "#1baf7a"),
        ("+YaRN",    [73.02, 78.84, 87.12], "#eda100"),
    ],
}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
x = range(len(BUCKETS))

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
axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor="linecolor")
fig.suptitle("Training-free rescaling: NTK null, PI/YaRN modest both-axis gain", fontsize=11, y=1.02)

fig.tight_layout()
fig.savefig("logs/phase2_ntk_null.png", dpi=220, bbox_inches="tight")
fig.savefig("logs/phase2_ntk_null.pdf", dpi=220, bbox_inches="tight")
print("wrote logs/phase2_ntk_null.png")
