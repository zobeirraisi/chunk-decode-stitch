"""Stitcher ablation, confidence-aware stitching, adaptive stride
and cross-model generality -- CPU only, from nc_dump.py dumps.

Hyperparameters of the new variants (geometric-prior strength, confidence
threshold tau, adaptive-stride rule) are selected on the SYNTHETIC dev set
(synth1, disjoint from every reported test set) and then applied unchanged to
LTB. LTB is never used for selection.

Sanity check: the "full" stitcher must reproduce chunked_decode.stitch exactly
on every record (asserted), and therefore the paper's main LTB table rows.

Usage (from code/):
  .venv/bin/python nc_stitch_ablation.py --tags base seed2 ft parseq crnn vitstr trba abinet
"""
import argparse
import itertools
import json
import os

from nc_common import (LTB_BUCKETS3, dump_path, fmt_row, read_jsonl, score,
                       stitch_reference, stitch_variant)

PRIORS = [0.01, 0.03, 0.05, 0.1]
TAUS = [0.3, 0.5, 0.7, 0.85]
ADAPT_W0 = [192, 256, 320, 384]   # images narrower than this use the small stride
ADAPT_SMALL = [32, 48]
# published main-table rows (stride 64, "full" stitcher), per bucket
PAPER = {"base": (48.28, 43.24, 36.84), "seed2": (48.76, 42.81, 37.59),
         "ft": (47.27, 41.76, 36.09), "parseq": (52.67, 47.69, 40.98)}


def dev_acc(recs, preds) -> float:
    """Synthetic dev objective: plain word accuracy over all lengths."""
    return 100.0 * sum(p == r["gt"] for r, p in zip(recs, preds)) / max(len(recs), 1)


def run_mode(recs, **kw):
    return [stitch_variant(r["chunks"], r["xs"], **kw)[0] for r in recs]


def check_reference(recs, label):
    bad = sum(stitch_variant(r["chunks"], r["xs"], mode="full")[0]
              != stitch_reference([c[0] for c in r["chunks"]], r["xs"], 128)
              for r in recs)
    assert bad == 0, f"{label}: full stitcher deviates from chunked_decode.stitch on {bad} records"


def select_on_dev(dev):
    """Grid-search prior (all search-based modes) and tau (conf) on synth dev."""
    best = {}
    for mode in ["full", "conf"]:
        grid = itertools.product(PRIORS, TAUS if mode == "conf" else [None])
        scored = []
        for prior, tau in grid:
            kw = {"mode": mode, "prior": prior}
            if tau is not None:
                kw["tau"] = tau
            scored.append((dev_acc(dev, run_mode(dev, **kw)), kw))
        scored.sort(key=lambda t: -t[0])
        best[mode] = {"dev_acc": scored[0][0], "kw": scored[0][1],
                      "grid": [(a, k) for a, k in scored]}
    return best


def adaptive(recs_by_stride, w0, small, kw):
    """Per image: stride `small` if the aspect-preserved width < w0 (strings
    just past the training cap span few crops, so boundary errors dominate),
    else the default 64. Width is known at test time; no label is used."""
    base = recs_by_stride[64]
    alt = {r["id"]: r for r in recs_by_stride[small]}
    preds = []
    for r in base:
        src = alt[r["id"]] if r["w"] < w0 else r
        preds.append(stitch_variant(src["chunks"], src["xs"], **kw)[0])
    return base, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["base", "seed2", "ft"])
    ap.add_argument("--dev_tag", default="base",
                    help="model whose synthetic dev dump selects hyperparameters")
    ap.add_argument("--dumps", default="results/nc/dumps")
    ap.add_argument("--out", default="results/nc/stitch_ablation.json")
    args = ap.parse_args()

    out = {}
    dev_path = dump_path(args.dumps, args.dev_tag, "chunks", "synth1", 64)
    dev = read_jsonl(dev_path)
    check_reference(dev, "dev")
    sel = select_on_dev(dev)
    out["dev_selection"] = {m: {"dev_acc": v["dev_acc"], "kw": v["kw"]} for m, v in sel.items()}
    print(f"dev ({dev_path}, n={len(dev)}):")
    for m, v in sel.items():
        print(f"  best {m:5s}: {v['kw']}  dev acc {v['dev_acc']:.2f}")

    # adaptive-stride rule, also selected on dev (needs dev dumps at 32/48/64)
    dev_by_s = {}
    for s in [32, 48, 64]:
        p = dump_path(args.dumps, args.dev_tag, "chunks", "synth1", s)
        if os.path.exists(p):
            dev_by_s[s] = read_jsonl(p)
    adapt_rule = None
    if len(dev_by_s) == 3:
        cands = []
        for w0, small in itertools.product(ADAPT_W0, ADAPT_SMALL):
            base, preds = adaptive(dev_by_s, w0, small, sel["conf"]["kw"])
            cands.append((dev_acc(base, preds), w0, small))
        cands.sort(reverse=True)
        adapt_rule = {"dev_acc": cands[0][0], "w0": cands[0][1], "small": cands[0][2]}
        out["adaptive_rule"] = adapt_rule
        print(f"  best adaptive stride: {adapt_rule}")

    for tag in args.tags:
        path = dump_path(args.dumps, tag, "chunks", "ltb", 64)
        if not os.path.exists(path):
            print(f"[skip] {tag}: {path} missing"); continue
        recs = read_jsonl(path)
        check_reference(recs, tag)
        rows = {
            "concat (no overlap removal)": {"mode": "concat"},
            "geometric overlap only": {"mode": "geometric"},
            "alignment, no fallback": {"mode": "align"},
            "full (published)": {"mode": "full"},
            "full, dev-selected prior": sel["full"]["kw"],
            "confidence-aware (dev-selected)": sel["conf"]["kw"],
        }
        print(f"\n=== {tag}: LTB both-axis word accuracy, stride 64 "
              f"({' / '.join(LTB_BUCKETS3)}) ===")
        out[tag] = {}
        for name, kw in rows.items():
            s = score(recs, run_mode(recs, **kw))
            out[tag][name] = s
            print(fmt_row(name, s))
            if name == "full (published)" and tag in PAPER:
                # cross-image batching under AMP can flip rare near-tie argmaxes
                d = [s[b]["acc"] - p for b, p in zip(LTB_BUCKETS3, PAPER[tag])]
                out[tag]["repro_delta_vs_paper"] = d
                print(f"{'  -> delta vs paper':34s} " + " ".join(f"{x:+6.2f}" for x in d))
        if adapt_rule:
            by_s = {s: read_jsonl(dump_path(args.dumps, tag, "chunks", "ltb", s))
                    for s in [64, adapt_rule["small"]]
                    if os.path.exists(dump_path(args.dumps, tag, "chunks", "ltb", s))}
            if len(by_s) == 2:
                base, preds = adaptive(by_s, adapt_rule["w0"], adapt_rule["small"],
                                       sel["conf"]["kw"])
                s = score(base, preds)
                out[tag]["confidence-aware + adaptive stride"] = s
                print(fmt_row("confidence-aware + adaptive stride", s))
        # per-stride sweep with the published stitcher (all strides dumped)
        for st in [32, 48, 80, 96]:
            p = dump_path(args.dumps, tag, "chunks", "ltb", st)
            if os.path.exists(p):
                r2 = read_jsonl(p)
                s = score(r2, run_mode(r2, mode="full"))
                out[tag][f"full, stride {st}"] = s
                print(fmt_row(f"full, stride {st}", s))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
