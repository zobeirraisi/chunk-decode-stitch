#!/usr/bin/env bash
# Paper C pilot: training-free chunked decoding on LTB, full buckets, both-axis
# protocol (comparable to manuscript Table `sota`). 2 checkpoints x 2 strides:
#  - fine-tuned (finetune_u14mbase_v6at_long): the paper's recommended model;
#    chunking is the decoding-side fix layered on the representation-side fix.
#  - baseline strong base: does chunking alone work, i.e. do the two fixes
#    compose (Paper C thesis) or does chunking subsume the fine-tune?
#  - stride 64 (50% overlap) vs 96 (25% overlap): overlap ablation.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=logs/chunked_pilot
mkdir -p "$OUT"

declare -A CKPTS=(
  [ft]=checkpoints/finetune_u14mbase_v6at_long.pt
  [base]=checkpoints/base_union14m_grid_fs8_best.pt
)

for tag in ft base; do
  for stride in 64 96; do
    echo "=== $tag stride$stride [$(date +%H:%M)] ==="
    $PY chunked_decode.py --checkpoint "${CKPTS[$tag]}" --stride "$stride" \
        --amp > "$OUT/${tag}_s${stride}.log" 2>&1
    grep -E "axis:|acc=" "$OUT/${tag}_s${stride}.log"
  done
done
echo "CHUNKED PILOT DONE"
