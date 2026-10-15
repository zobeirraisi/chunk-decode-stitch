"""Resolution-controlled decoder-axis analysis (paper: decoder positional encoding section).

The confound: the decoder-axis condition squeezes a long line into 128px,
so its CER mixes decoder-length extrapolation with lost per-character
legibility. The control used here holds resolution fixed WITHIN each image:
every character of a squeezed line has the same nominal pixels-per-character
(128/L), but only characters at ground-truth positions >= the training cap
(25) require decoding past the trained horizon. Splitting each image's edits
by ground-truth position therefore gives, at identical resolution,
  CER_in      positions 0..24   (in-horizon: legibility + ordinary errors)
  CER_beyond  positions >= 25   (beyond-horizon: legibility + length)
and CER_beyond - CER_in estimates the decoder-length effect alone.
CER_in rising with L (as px/char falls) measures the legibility effect alone.

Edits are attributed by alignment (nc_common.align): substitutions and
deletions to their ground-truth index, insertions to the decode step that
emitted them (their prediction index). Caveat reported with the numbers: later
positions also sit further right in the image and later in the autoregressive
sequence (error propagation), so the difference is an upper bound on the pure
position-index effect, not a lower one.

Usage (from code/):
  .venv/bin/python nc_decoder_control.py --tags base seed2 ft
"""
import argparse
import json
import os
from collections import defaultdict

from nc_common import align, dump_path, read_jsonl

CAP = 25
BANDS = [(0, 13), (13, 25), (25, 38), (38, 50), (50, 75), (75, 200)]


def split_edits(gt: str, pred: str):
    """Per position: number of edits attributed to it. Substitutions and
    deletions go to their ground-truth index; an insertion goes to the decoder
    step that emitted it (its prediction index), since the training horizon
    is a property of the decode step. Insertions past the end of the ground
    truth are kept (in an overflow list) so no edit is dropped."""
    per = [0] * len(gt)
    overflow = []
    for op, i, j in align(gt, pred):
        if op in ("S", "D"):
            per[i] += 1
        elif op == "I":
            if j < len(gt):
                per[j] += 1
            else:
                overflow.append(j)
    return per, overflow


def analyze(recs):
    agg = defaultdict(lambda: {"in": [0, 0], "beyond": [0, 0],
                               "bands": [[0, 0] for _ in BANDS], "n": 0, "pxpc": 0.0})
    for r in recs:
        gt, pred = r["gt"], r["pred"][0]
        if not gt:
            continue
        per, overflow = split_edits(gt, pred)
        a = agg[r["bucket"]]
        for j in overflow:  # steps past the reference length are all >= CAP here
            key = "in" if j < CAP else "beyond"
            a[key][0] += 1
            for b, (lo, hi) in enumerate(BANDS):
                if lo <= j < hi:
                    a["bands"][b][0] += 1
        a["n"] += 1
        a["pxpc"] += 128.0 / len(gt)
        for pos, e in enumerate(per):
            key = "in" if pos < CAP else "beyond"
            a[key][0] += e; a[key][1] += 1
            for b, (lo, hi) in enumerate(BANDS):
                if lo <= pos < hi:
                    a["bands"][b][0] += e; a["bands"][b][1] += 1
    out = {}
    for bucket, a in agg.items():
        cer = lambda x: 100.0 * x[0] / x[1] if x[1] else None
        out[bucket] = {
            "n": a["n"], "mean_px_per_char": a["pxpc"] / max(a["n"], 1),
            "cer_in": cer(a["in"]), "cer_beyond": cer(a["beyond"]),
            "bands": {f"{lo}-{hi - 1}": cer(x) for (lo, hi), x in zip(BANDS, a["bands"])},
        }
        if out[bucket]["cer_beyond"] is not None:
            out[bucket]["length_effect"] = out[bucket]["cer_beyond"] - out[bucket]["cer_in"]
    return out


def order(keys):
    return sorted(keys, key=lambda k: int(k.rstrip("+").split("-")[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["base"])
    ap.add_argument("--srcs", nargs="+", default=["ltb", "synth0"])
    ap.add_argument("--dumps", default="results/nc/dumps")
    ap.add_argument("--out", default="results/nc/decoder_control.json")
    args = ap.parse_args()

    result = {}
    for tag in args.tags:
        for src in args.srcs:
            path = dump_path(args.dumps, tag, "decoder", src)
            if not os.path.exists(path):
                print(f"[skip] {path}"); continue
            res = analyze(read_jsonl(path))
            result[f"{tag}/{src}"] = res
            print(f"\n=== {tag} / {src}: decoder axis (squeezed to 128px, budget 110) ===")
            print(f"{'bucket':7s} {'n':>5s} {'px/char':>8s} {'CER<25':>8s} {'CER>=25':>8s} {'diff':>7s}   bands "
                  + " ".join(f"{lo}-{hi - 1}" for lo, hi in BANDS))
            for b in order(res):
                x = res[b]
                bands = " ".join(f"{v:6.1f}" if v is not None else "     -" for v in x["bands"].values())
                beyond = f"{x['cer_beyond']:8.2f}" if x["cer_beyond"] is not None else "       -"
                diff = f"{x['length_effect']:7.2f}" if "length_effect" in x else "      -"
                print(f"{b:7s} {x['n']:5d} {x['mean_px_per_char']:8.2f} {x['cer_in']:8.2f} {beyond} {diff}   {bands}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
