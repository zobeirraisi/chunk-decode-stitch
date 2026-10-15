"""Score the portrait-rotation rule (dumps from nc_rotate.py).

For each portrait image, three candidate reads: unrotated (from the existing
nc_dump.py dump, read exactly as the reported pipeline reads it), rotated
clockwise, rotated counter-clockwise (published stitcher on the rotated crop
grid). Rule: keep the candidate with the highest mean per-character
confidence (no labels). Landscape images keep their reported read. Also
printed: the oracle (correct if any candidate is correct), as an upper bound.

  LTB:        reported read = stitched chunks at stride 64
  std6/u14m:  reported read = gated pipeline at T=128 (plain squash read if
              aspect-preserved width <= 128, else stitched chunks)

Usage (from code/):
  .venv/bin/python nc_rotate_eval.py --tag base --src ltb
  .venv/bin/python nc_rotate_eval.py --tag base --src all
"""
import argparse
import json
from collections import defaultdict

from nc_common import LTB_BUCKETS3, dump_path, read_jsonl, stitch_variant

GATE_T = 128


def mean_conf(conf):
    return sum(conf) / len(conf) if conf else 0.0


def stitched(chunks, xs):
    t, c = stitch_variant(chunks, xs, mode="full")
    return t, c


def reported(r, src):
    if src == "ltb" or r["w"] > GATE_T:
        return stitched(r["chunks"], r["xs"])
    return r["plain"][0], r["plain"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--src", default="ltb", help="ltb | all (std6 + u14m)")
    ap.add_argument("--dumps", default="results/nc/dumps")
    ap.add_argument("--out", default="results/nc/rotate_{tag}_{src}.json")
    args = ap.parse_args()

    if args.src == "ltb":
        base = read_jsonl(dump_path(args.dumps, args.tag, "chunks", "ltb", 64))
    else:
        base = read_jsonl(f"{args.dumps}/{args.tag}_gate_all_s64.jsonl.gz")
    rot = {(r["src"], r["bucket"], r["id"]): r
           for r in read_jsonl(f"{args.dumps}/{args.tag}_rotate_{args.src}.jsonl.gz")}

    # group key: LTB bucket, or the benchmark family (std6 / u14m) and subset
    stats = defaultdict(lambda: defaultdict(int))
    for r in base:
        key = (r["src"], r["bucket"])
        s = stats[key]
        t0, c0 = reported(r, args.src)
        s["n"] += 1
        s["base"] += t0 == r["gt"]
        q = rot.get((r["src"], r["bucket"], r["id"]))
        if q is None:
            s["rule"] += t0 == r["gt"]
            s["oracle"] += t0 == r["gt"]
            continue
        cands = [(t0, c0, "none"),
                 (*stitched(q["chunks_cw"], q["xs_cw"]), "cw"),
                 (*stitched(q["chunks_ccw"], q["xs_ccw"]), "ccw")]
        best = max(cands, key=lambda x: mean_conf(x[1]))
        s["portrait"] += 1
        s["portrait_base"] += t0 == r["gt"]
        s["rule"] += best[0] == r["gt"]
        s["portrait_rule"] += best[0] == r["gt"]
        s["oracle"] += any(c[0] == r["gt"] for c in cands)
        s["chose_" + best[2]] += 1

    def acc(s, k):
        return 100.0 * s[k] / max(s["n"], 1)

    out = {}
    print(f"=== {args.tag} / {args.src}: portrait-rotation rule (mean-confidence choice) ===")
    print(f"{'group':28s} {'n':>7s} {'portr':>6s} {'base':>7s} {'rule':>7s} {'oracle':>7s}"
          f"  portrait: base->rule   chose none/cw/ccw")
    for key in sorted(stats):
        s = stats[key]
        out["/".join(key)] = dict(s)
        print(f"{'/'.join(key):28s} {s['n']:7d} {s['portrait']:6d} {acc(s, 'base'):7.2f} "
              f"{acc(s, 'rule'):7.2f} {acc(s, 'oracle'):7.2f}  {s['portrait_base']:5d}->"
              f"{s['portrait_rule']:<5d}  {s['chose_none']}/{s['chose_cw']}/{s['chose_ccw']}")

    def summary(keys, label):
        ks = [k for k in keys if k in stats]
        if not ks:
            return
        tot = {m: [acc(stats[k], m) for k in ks] for m in ("base", "rule", "oracle")}
        ns = [stats[k]["n"] for k in ks]
        line = f"{label:28s}"
        for m in ("base", "rule", "oracle"):
            w = sum(a * n for a, n in zip(tot[m], ns)) / sum(ns)
            a = sum(tot[m]) / len(tot[m])
            line += f"  {m} W {w:6.2f} A {a:6.2f}"
            out[f"{label}/{m}"] = {"W": w, "A": a}
        print(line)

    if args.src == "ltb":
        summary([("ltb", b) for b in LTB_BUCKETS3], "LTB")
    else:
        summary([k for k in stats if k[0] == "std6"], "std6 (A = benchmark avg)")
        summary([k for k in stats if k[0] == "u14m"], "u14m (A = subset avg)")
    path = args.out.format(tag=args.tag, src=args.src)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
