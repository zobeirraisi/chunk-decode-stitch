"""Print-size supplement figures.

Redraws Fig. S1 (controlled synthetic length curve) and Fig. S2 (stride and
crop-phase robustness) at the supplement's text width (345 pt = 4.79 in,
article class, 10pt) with 7.5-8.5 pt text, so nothing is shrunk on inclusion.
Every number is parsed from the original evaluation logs, not retyped:
  S1: logs/eval_synth_length_{base,v6}.log
  S2: logs/chunked_pilot/base_s{64,96}.log, logs/chunked_hardening/{stride,phase}_*.log

Run from code/:  .venv/bin/python make_nc_supp_figures.py
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
W = 4.79  # inches
GRAY, BLUE, AQUA, GOLD = "#52514e", "#2a78d6", "#1baf7a", "#eda100"
GRID, SPINE = "#e6e5e0", "#c3c2b7"
SMTR = 41.57  # published A-Avg (Du et al.)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=GRAY)


def synth_curve(path):
    """{axis: {length: cer}} from an eval_synth_length.py log."""
    out, axis = {}, None
    for line in open(path):
        m = re.match(r"=== axis: (\w+)", line)
        if m:
            axis = m.group(1); out[axis] = {}
        m = re.search(r"len=\s*(\d+).*cer=\s*([\d.]+)%", line)
        if m and axis:
            out[axis][int(m.group(1))] = float(m.group(2))
    return out


def a_avg(path):
    accs = [float(x) for x in re.findall(r"acc=\s*([\d.]+)%", open(path).read())[:3]]
    assert len(accs) == 3, path
    return sum(accs) / 3


def fig_s1():
    base = synth_curve("logs/eval_synth_length_base.log")
    ft = synth_curve("logs/eval_synth_length_v6.log")
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.75), sharex=True)
    for ax, (axis, title) in zip(axes, [("decoder", "decoder axis"), ("width", "width axis"),
                                         ("both", "both axes")]):
        L = sorted(base[axis])
        ax.plot(L, [base[axis][l] for l in L], color=GRAY, lw=1.3, marker="o", ms=2.6,
                label="baseline", zorder=3)
        ax.plot(L, [ft[axis][l] for l in L], color=BLUE, lw=1.3, marker="o", ms=2.6,
                label="fine-tuned (80/20)", zorder=3)
        ax.axvline(25, color=SPINE, lw=0.9, ls="--", zorder=1)
        ax.set_title(title, pad=3)
        ax.set_xlabel("length (chars)")
        ax.set_xticks([25, 50, 75, 100])
        style(ax)
    axes[0].set_ylabel("CER (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, labelcolor="linecolor",
               handlelength=1.5, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/phase1_synth_length_curve_nc.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)


def fig_s2():
    strides = {32: "logs/chunked_hardening/stride_32.log", 48: "logs/chunked_hardening/stride_48.log",
               64: "logs/chunked_pilot/base_s64.log", 80: "logs/chunked_hardening/stride_80.log",
               96: "logs/chunked_pilot/base_s96.log", 112: "logs/chunked_hardening/stride_112.log"}
    phases = {0: "logs/chunked_pilot/base_s64.log", 16: "logs/chunked_hardening/phase_16.log",
              32: "logs/chunked_hardening/phase_32.log", 48: "logs/chunked_hardening/phase_48.log"}
    sx = sorted(strides); sy = [a_avg(strides[s]) for s in sx]
    px = sorted(phases); py = [a_avg(phases[p]) for p in px]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(W, 1.9))
    axL.axvspan(28, 64, color=BLUE, alpha=0.07, lw=0)
    axL.plot(sx, sy, color=BLUE, lw=1.3, marker="o", ms=3, zorder=3)
    axL.axhline(SMTR, color=GOLD, lw=0.9, ls="--")
    axL.text(30, SMTR - 1.2, "SMTR 41.57", color=GOLD, ha="left", va="top", fontsize=7)
    for x, y in zip(sx, sy):
        below = SMTR - 2.5 < y < SMTR  # label would sit on the SMTR line
        axL.annotate(f"{y:.1f}", (x, y), xytext=(-4 if below else 0, -10 if below else 4),
                     textcoords="offset points", ha="right" if below else "center",
                     fontsize=6.5, color=GRAY)
    axL.set_xticks(sx)
    axL.set_xlabel("crop stride (px)")
    axL.set_ylabel("LTB A-Avg (%)")
    axL.set_title("stride sweep", pad=3)
    axR.plot(px, py, color=AQUA, lw=1.3, marker="o", ms=3, zorder=3)
    axR.axhline(SMTR, color=GOLD, lw=0.9, ls="--")
    axR.text(48, SMTR - 0.05, "SMTR 41.57", color=GOLD, ha="right", va="top", fontsize=7)
    for x, y in zip(px, py):
        below = y == min(py)
        first = x == px[0]
        axR.annotate(f"{y:.2f}", (x, y), xytext=(5 if first else 0, -10 if below else 4),
                     textcoords="offset points", ha="left" if first else "center",
                     fontsize=6.5, color=GRAY)
    axR.set_xticks(px)
    axR.set_xlabel("crop-grid phase (px), stride 64")
    axR.set_title("phase sweep", pad=3)
    axR.set_ylim(min(py + [SMTR]) - 0.35, max(py) + 0.35)
    axL.set_ylim(min(sy) - 3, max(sy) + 4)
    for ax in (axL, axR):
        style(ax)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/paperc_robustness_nc.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print("stride A-Avg", dict(zip(sx, [round(v, 2) for v in sy])))
    print("phase  A-Avg", dict(zip(px, [round(v, 2) for v in py])))


if __name__ == "__main__":
    fig_s1()
    fig_s2()
    print(f"wrote {OUT}/phase1_synth_length_curve_nc.pdf, {OUT}/paperc_robustness_nc.pdf")
