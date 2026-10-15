"""Parse an eval_ltb.py log and emit the SMTR-format word-accuracy row
(per-bucket acc + W-Avg + A-Avg) for a chosen axis, so results drop straight
into the Du et al. (SMTR, AAAI'25) Table-4 comparison. W-Avg is sample-weighted
by the true full-LTB bucket sizes; A-Avg is the unweighted 3-bucket mean. Both
formulas reproduce the published FocalSVTR row of that table.

Usage:
  .venv/bin/python ltb_wavg.py logs/run3_u14mbase/v6_ltb.log --axis both
"""
import argparse
import re

# true full-LTB bucket sizes (data/ltb/*.txt); sum = 4789 = published LTB size
BUCKET_N = {"26-35": 3376, "36-55": 1147, "56+": 266}


def parse(path, axis):
    """Return {bucket: acc} for the requested axis section of an eval_ltb log."""
    accs, cur = {}, None
    for line in open(path):
        m = re.search(r"=== axis:\s*(\w+)", line)
        if m:
            cur = m.group(1)
            continue
        if cur != axis:
            continue
        # e.g. "  26-35  (n= 600): acc=13.00% cer=45.45% mean_pred_len=19.9"
        m = re.search(r"(\S+)\s+\(n=\s*\d+\):\s*acc=\s*([\d.]+)%", line)
        if m:
            accs[m.group(1)] = float(m.group(2))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--axis", default="both", choices=["decoder", "width", "both"])
    args = ap.parse_args()
    accs = parse(args.log, args.axis)
    if not accs:
        print(f"no '{args.axis}' axis buckets found in {args.log}")
        return
    order = [b for b in ("26-35", "36-55", "56+") if b in accs]
    a_avg = sum(accs[b] for b in order) / len(order)
    tot_n = sum(BUCKET_N[b] for b in order)
    w_avg = sum(accs[b] * BUCKET_N[b] for b in order) / tot_n
    cells = " | ".join(f"{accs[b]:.2f}" for b in order)
    print(f"axis={args.axis}  buckets({'/'.join(order)}) = {cells}")
    print(f"  W-Avg = {w_avg:.2f}   A-Avg = {a_avg:.2f}")
    print(f"  markdown row: | {cells} | {w_avg:.2f} | {a_avg:.2f} |")


if __name__ == "__main__":
    main()
