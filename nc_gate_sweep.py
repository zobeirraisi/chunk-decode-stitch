"""Width-gate threshold sweep (paper: Width-Gate Sweep).

Deployment rule: read an image plainly (training-matched 128x32 squash) when
its aspect-preserved width at height 32 is <= T, otherwise chunk-decode-stitch
it. The paper uses T = 128 (the training width). This sweep shows how the
rule behaves for T from "always chunk" to "never chunk" on
  std6   the six standard benchmarks (mean word accuracy over the six)
  u14m   Union14M-B, seven subsets (mean over subsets)
  LTB    both-axis, W-Avg and A-Avg (plain branch = squeezed decoder-axis read)
using only dumps (nc_dump.py --mode gate / chunks / decoder). If the curve is
flat around T = 128, the gate is a fixed property of the model (its training
width), not a tuned hyperparameter.

Usage (from code/):
  .venv/bin/python nc_gate_sweep.py --tag base
"""
import argparse
import json
import os
from collections import defaultdict

from nc_common import dump_path, read_jsonl, score, stitch_variant

THRESHOLDS = [0, 96, 128, 160, 192, 256, 384, 10**9]  # 0 = always chunk


def label(t):
    return "always chunk" if t == 0 else ("never chunk" if t >= 10**9 else f"T={t}")


def mean_acc(recs, preds):
    by = defaultdict(lambda: [0, 0])
    for r, p in zip(recs, preds):
        by[r["bucket"]][0] += p == r["gt"]
        by[r["bucket"]][1] += 1
    per = {b: 100.0 * c / n for b, (c, n) in by.items()}
    return sum(per.values()) / max(len(per), 1), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--dumps", default="results/nc/dumps")
    ap.add_argument("--stitch", default="full", help="stitcher mode for the chunk branch")
    ap.add_argument("--out", default="results/nc/gate_sweep_{tag}.json")
    args = ap.parse_args()

    short = read_jsonl(dump_path(args.dumps, args.tag, "gate", "all", 64))
    ltb = read_jsonl(dump_path(args.dumps, args.tag, "chunks", "ltb", 64))
    plain_ltb = {r["id"]: r["pred"][0] for r in
                 read_jsonl(dump_path(args.dumps, args.tag, "decoder", "ltb"))}
    missing = sum(r["id"] not in plain_ltb for r in ltb)
    if missing:
        print(f"[warn] {missing} LTB records lack a plain read; scoring the intersection")
        ltb = [r for r in ltb if r["id"] in plain_ltb]
    for r in short:
        r["stitched"] = stitch_variant(r["chunks"], r["xs"], mode=args.stitch)[0]
    for r in ltb:
        r["stitched"] = stitch_variant(r["chunks"], r["xs"], mode=args.stitch)[0]
    std6 = [r for r in short if r["src"] == "std6"]
    u14m = [r for r in short if r["src"] == "u14m"]

    rows = {}
    print(f"=== {args.tag}: width-gate sweep (chunk if width > T; stitcher={args.stitch}) ===")
    print(f"{'rule':14s} {'std6 avg':>9s} {'u14m avg':>9s} {'LTB W-Avg':>10s} {'LTB A-Avg':>10s}"
          f" {'%std6 chunked':>14s}")
    for t in THRESHOLDS:
        pick = lambda r, plain: r["stitched"] if r["w"] > t else plain
        s6, s6_per = mean_acc(std6, [pick(r, r["plain"][0]) for r in std6])
        su, su_per = mean_acc(u14m, [pick(r, r["plain"][0]) for r in u14m])
        sl = score(ltb, [pick(r, plain_ltb[r["id"]]) for r in ltb])
        frac = 100.0 * sum(r["w"] > t for r in std6) / max(len(std6), 1)
        rows[label(t)] = {"std6": s6, "std6_per": s6_per, "u14m": su, "u14m_per": su_per,
                          "ltb": sl, "std6_chunked_pct": frac}
        print(f"{label(t):14s} {s6:9.2f} {su:9.2f} {sl['W-Avg']:10.2f} {sl['A-Avg']:10.2f} {frac:14.1f}")
    out = args.out.format(tag=args.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
