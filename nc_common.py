"""Shared helpers for the analysis experiments (nc_*.py).

Design: every GPU run (nc_dump.py) writes per-crop reads WITH per-character
confidences to JSONL. All stitcher variants, the error taxonomy, the gate
sweep and the decoder-axis control then run on CPU from those dumps, so the
GPU is needed exactly once per (model, data, stride).

Dump record formats (one JSON object per line, gzip-compressed):
  chunks : {"src", "bucket", "id", "gt", "w", "xs", "chunks": [[text, [p...]], ...]}
  plain  : {"src", "bucket", "id", "gt", "w", "pred": [text, [p...]]}
  gate   : chunks record + "plain": [text, [p...]]   (short-text benchmarks)
`gt` is already cleaned to the 36-char protocol; `w` is the aspect-preserved
width at height 32; `xs` are crop left edges; `p` are per-character max
softmax probabilities aligned 1:1 with `text`.
"""
import gzip
import json
import os

from chunked_decode import merge_pair, stitch as stitch_reference  # noqa: F401
from eval_ltb import levenshtein

CHUNK_W = 128
LTB_BUCKETS3 = ["26-35", "36-55", "56+"]


# --------------------------------------------------------------------- I/O
def write_jsonl(path: str, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_path(out_dir: str, tag: str, kind: str, src: str, stride=None) -> str:
    s = f"_s{stride}" if stride is not None else ""
    return os.path.join(out_dir, f"{tag}_{kind}_{src}{s}.jsonl.gz")


# --------------------------------------------------------------- alignment
def align(a: str, b: str):
    """Levenshtein alignment a -> b with backtrace.

    Returns a list of ops (op, i, j): op in {"=", "S", "D", "I"}; i indexes a
    (None for "I"), j indexes b (None for "D"). "D" = char of a missing in b,
    "I" = char of b not in a. Ties prefer match/substitution, then deletion."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (ai != b[j - 1]), prev[j] + 1, row[j - 1] + 1)
    ops, i, j, last = [], n, m, None
    while i > 0 or j > 0:
        diag = i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1])
        dele = i > 0 and d[i][j] == d[i - 1][j] + 1
        ins = j > 0 and d[i][j] == d[i][j - 1] + 1
        # on ties, continue an open gap so each indel run stays contiguous
        if last == "I" and ins:
            op = "I"
        elif last == "D" and dele:
            op = "D"
        else:
            op = "M" if diag else ("D" if dele else "I")
        if op == "M":
            ops.append(("=" if a[i - 1] == b[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op == "D":
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
        last = op
    return ops[::-1]


# ---------------------------------------------------------------- stitchers
# Each stitcher maps (chunks=[(text, conf)], xs, chunk_w) -> (text, conf).
# "full" is the published algorithm; it is re-implemented here on
# (text, conf) pairs and checked against chunked_decode.stitch on text.

def _k_search(acc: str, nxt: str, ov_px: int, chunk_w: int, prior: float,
              min_score: float, slack: int = 3, fallback: bool = True):
    """Overlap-length search of chunked_decode.merge_pair, with the strength
    of the geometric prior (`prior`, 0.01 in the paper) exposed."""
    frac = ov_px / chunk_w
    k_exp = round(frac * len(nxt))
    lo = max(0, k_exp - slack - int(0.25 * k_exp))
    hi = min(len(acc), len(nxt), k_exp + slack + int(0.25 * k_exp))
    best_k, best_score = None, -1.0
    for k in range(lo, hi + 1):
        if k == 0:
            continue
        score = 1.0 - levenshtein(acc[-k:], nxt[:k]) / k
        score -= prior * abs(k - k_exp)
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None or (fallback and best_score < min_score):
        best_k = min(k_exp, len(acc), len(nxt))
    return best_k, k_exp


def _half_merge(acc, nxt, k):
    """Published overlap rule: left half from acc, right half from nxt."""
    (at, ac), (nt, nc) = acc, nxt
    half = k // 2
    cut = len(at) - k + half
    return at[:cut] + nt[half:], ac[:cut] + nc[half:]


def _conf_merge(acc, nxt, k, tau: float):
    """Confidence-resolved overlap: align acc's last k chars with nxt's first
    k chars; on a match keep the char (max confidence), on a substitution keep
    the more confident read, and keep an unmatched char only if its own
    confidence is at least tau."""
    (at, ac), (nt, nc) = acc, nxt
    a, b = at[len(at) - k:], nt[:k]
    a_c, b_c = ac[len(ac) - k:], nc[:k]
    mt, mc = [], []
    for op, i, j in align(a, b):
        if op == "=":
            mt.append(a[i]); mc.append(max(a_c[i], b_c[j]))
        elif op == "S":
            if a_c[i] >= b_c[j]:
                mt.append(a[i]); mc.append(a_c[i])
            else:
                mt.append(b[j]); mc.append(b_c[j])
        elif op == "D":
            if a_c[i] >= tau:
                mt.append(a[i]); mc.append(a_c[i])
        else:  # "I"
            if b_c[j] >= tau:
                mt.append(b[j]); mc.append(b_c[j])
    head_t, head_c = at[: len(at) - k], ac[: len(ac) - k]
    return head_t + "".join(mt) + nt[k:], head_c + mc + nc[k:]


def stitch_variant(chunks, xs, chunk_w: int = CHUNK_W, mode: str = "full",
                   prior: float = 0.01, min_score: float = 0.4, tau: float = 0.5):
    """Stitch per-crop reads. Modes (stitcher ablation):
      concat     no overlap removal (lower bound)
      geometric  overlap length from pixel geometry only, no text alignment
      align      edit-distance search, no geometric fallback
      full       published method (search + fallback + half/half merge)
      conf       full search, confidence-resolved overlap
    `prior` scales the pull toward the geometric expectation (0.01 = paper)."""
    acc = (chunks[0][0], list(chunks[0][1]))
    for i in range(1, len(chunks)):
        nxt = (chunks[i][0], list(chunks[i][1]))
        if not nxt[0]:
            continue
        if not acc[0]:
            acc = nxt
            continue
        ov_px = max(xs[i - 1] + chunk_w - xs[i], 0)
        if mode == "concat":
            acc = (acc[0] + nxt[0], acc[1] + nxt[1])
            continue
        if mode == "geometric":
            k_exp = round(ov_px / chunk_w * len(nxt[0]))
            k = min(k_exp, len(acc[0]), len(nxt[0]))
        else:
            k, _ = _k_search(acc[0], nxt[0], ov_px, chunk_w, prior, min_score,
                             fallback=(mode != "align"))
        if mode == "conf":
            acc = _conf_merge(acc, nxt, k, tau)
        else:
            acc = _half_merge(acc, nxt, k)
    return acc


# ------------------------------------------------------------------ scoring
def score(records, preds):
    """Per-bucket word accuracy / CER plus W-Avg and A-Avg over LTB buckets.
    `preds` is aligned with `records`."""
    by = {}
    for r, p in zip(records, preds):
        b = by.setdefault(r["bucket"], [0, 0, 0, 0])  # n, correct, edits, ref
        b[0] += 1
        b[1] += p == r["gt"]
        b[2] += levenshtein(p, r["gt"])
        b[3] += len(r["gt"])
    out = {k: {"n": v[0], "acc": 100.0 * v[1] / max(v[0], 1),
               "cer": 100.0 * v[2] / max(v[3], 1)} for k, v in by.items()}
    ltb = [out[b] for b in LTB_BUCKETS3 if b in out]
    if len(ltb) == 3:
        out["A-Avg"] = sum(x["acc"] for x in ltb) / 3
        out["W-Avg"] = sum(x["acc"] * x["n"] for x in ltb) / sum(x["n"] for x in ltb)
    return out


def fmt_row(name: str, s: dict) -> str:
    cells = " ".join(f"{s[b]['acc']:6.2f}" for b in LTB_BUCKETS3 if b in s)
    tail = f"  W-Avg {s['W-Avg']:6.2f}  A-Avg {s['A-Avg']:6.2f}" if "A-Avg" in s else ""
    return f"{name:34s} {cells}{tail}"
