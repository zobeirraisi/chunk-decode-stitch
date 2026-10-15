"""Quantitative failure taxonomy for chunk-decode-stitch on LTB (paper: Where the
Remaining Errors Come From).

Every wrong prediction is aligned to its ground truth (nc_common.align);
consecutive non-match operations form an error run, and each run gets one
label:
  over-merge      only deletions, and the deleted string also occurs in the
                  ground truth within REPEAT_WINDOW characters (a repeated
                  substring collapsed by the overlap alignment)
  duplication     only insertions, and the inserted string also occurs in the
                  prediction within REPEAT_WINDOW characters (overlap not removed)
  substitution    only substitutions (a misread character)
  deletion        other deletions (characters dropped)
  insertion       other insertions (spurious characters)
  mixed           anything else
A sample can carry several labels. Reported per bucket: number of wrong
samples, share containing each label, character edits per label, and the
share of near misses (edit distance 1). Run for the published stitcher and
the confidence-aware one, so the effect of confidence stitching on each error class is visible.

Usage (from code/):
  .venv/bin/python nc_error_taxonomy.py --tag base
"""
import argparse
import json
import os
from collections import Counter, defaultdict

from nc_common import LTB_BUCKETS3, align, dump_path, read_jsonl, stitch_variant

REPEAT_WINDOW = 15  # chars; about two crops' worth of overlap at stride 64
LABELS = ["over-merge", "duplication", "substitution", "deletion", "insertion", "mixed"]


def occurs_nearby(s: str, text: str, start: int, end: int) -> bool:
    """Does s occur in text within REPEAT_WINDOW chars of [start, end),
    not counting the span itself?"""
    lo = max(0, start - REPEAT_WINDOW - len(s))
    hi = min(len(text), end + REPEAT_WINDOW + len(s))
    for k in range(lo, hi - len(s) + 1):
        if k != start and text[k:k + len(s)] == s:
            return True
    return False


def error_runs(gt: str, pred: str):
    """Yield (label, n_edits, gt_start) for each error run."""
    ops = align(gt, pred)
    run, gpos = [], 0
    for op in ops + [("=", None, None)]:
        if op[0] != "=":
            run.append(op)
            continue
        if run:
            kinds = {o[0] for o in run}
            gi = [o[1] for o in run if o[1] is not None]
            pj = [o[2] for o in run if o[2] is not None]
            start = gi[0] if gi else gpos
            if kinds == {"D"}:
                s = "".join(gt[i] for i in gi)
                lab = "over-merge" if occurs_nearby(s, gt, gi[0], gi[-1] + 1) else "deletion"
            elif kinds == {"I"}:
                s = "".join(pred[j] for j in pj)
                lab = "duplication" if occurs_nearby(s, pred, pj[0], pj[-1] + 1) else "insertion"
            elif kinds == {"S"}:
                lab = "substitution"
            else:
                lab = "mixed"
            yield lab, len(run), start
            run = []
        if op[1] is not None:
            gpos = op[1] + 1


def taxonomy(recs, preds):
    out, examples = {}, defaultdict(list)
    for bucket in LTB_BUCKETS3:
        n = wrong = near = 0
        has, edits = Counter(), Counter()
        for r, p in zip(recs, preds):
            if r["bucket"] != bucket:
                continue
            n += 1
            if p == r["gt"]:
                continue
            wrong += 1
            labs = set()
            ed = 0
            for lab, k, _ in error_runs(r["gt"], p):
                labs.add(lab); edits[lab] += k; ed += k
            near += ed == 1
            for lab in labs:
                has[lab] += 1
                if len(examples[lab]) < 12:
                    examples[lab].append({"id": r["id"], "bucket": bucket,
                                          "gt": r["gt"], "pred": p})
        tot = max(sum(edits.values()), 1)
        out[bucket] = {
            "n": n, "wrong": wrong,
            "near_miss_share": 100.0 * near / max(wrong, 1),
            "samples_with": {l: 100.0 * has[l] / max(wrong, 1) for l in LABELS},
            "edit_share": {l: 100.0 * edits[l] / tot for l in LABELS},
        }
    return out, dict(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--dumps", default="results/nc/dumps")
    ap.add_argument("--ablation", default="results/nc/stitch_ablation.json",
                    help="read the dev-selected confidence-aware settings from here")
    ap.add_argument("--out", default="results/nc/error_taxonomy_{tag}.json")
    args = ap.parse_args()

    recs = read_jsonl(dump_path(args.dumps, args.tag, "chunks", "ltb", args.stride))
    variants = {"full": {"mode": "full"}}
    if os.path.exists(args.ablation):
        with open(args.ablation) as f:
            variants["conf"] = json.load(f)["dev_selection"]["conf"]["kw"]
    result = {}
    for name, kw in variants.items():
        preds = [stitch_variant(r["chunks"], r["xs"], **kw)[0] for r in recs]
        tax, ex = taxonomy(recs, preds)
        result[name] = {"kw": kw, "taxonomy": tax, "examples": ex}
        print(f"\n=== {args.tag} / {name} stitcher: share of WRONG samples containing each error class (%) ===")
        print(f"{'bucket':8s} {'wrong':>6s} {'ED=1':>6s} " + " ".join(f"{l:>12s}" for l in LABELS))
        for b in LTB_BUCKETS3:
            t = tax[b]
            print(f"{b:8s} {t['wrong']:6d} {t['near_miss_share']:6.1f} "
                  + " ".join(f"{t['samples_with'][l]:12.1f}" for l in LABELS))
    out = args.out.format(tag=args.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
