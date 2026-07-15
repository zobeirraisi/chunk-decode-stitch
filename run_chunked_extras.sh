#!/usr/bin/env bash
# Compute-cost accounting + non-horizontal-layout stress test for the chunked
# decoding pilot (docs/paperc-chunked-pilot.md caveats 2 and 4), on the
# headline (untouched base) checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=logs/chunked_extras
mkdir -p "$OUT"
BASE=checkpoints/base_union14m_grid_fs8_best.pt

echo "########## [$(date)] compute cost (n=150/bucket) ##########"
$PY chunked_cost.py --checkpoint "$BASE" --n 150 --amp > "$OUT/cost.log" 2>&1
cat "$OUT/cost.log"

echo "########## [$(date)] non-horizontal stress test (full subsets) ##########"
$PY chunked_u14m_nonhorizontal.py --checkpoint "$BASE" --amp > "$OUT/nonhorizontal.log" 2>&1
cat "$OUT/nonhorizontal.log"

echo "EXTRAS DONE"
