#!/usr/bin/env bash
# Hardening sweep for the chunked-decoding pilot (docs/paperc-chunked-pilot.md).
# Three axes of robustness, all full-bucket LTB `both`-axis eval:
#   1. Stride sweep (chunk_w=128 fixed): is stride=64 actually near-optimal, or
#      did the first pilot get lucky with two arbitrary points?
#   2. Crop-grid phase shift ("seed" for an otherwise fully deterministic
#      algorithm): does the arbitrary pixel phase of the crop grid matter?
#      base checkpoint, stride=64 (winner so far).
#   3. Fine-tune weight seed variance: v6at / v6at_seed1 / v6at_seed2 (same
#      recipe, 5k iters, differ only by training seed) chunked at stride=64 --
#      does the "fine-tuning doesn't help once chunked" finding hold across
#      fine-tune seeds, or was seed 0 a fluke?
# stride=64/phase=0 base-checkpoint result already exists (logs/chunked_pilot/
# base_s64.log) and is reused rather than re-run (fully deterministic).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=logs/chunked_hardening
mkdir -p "$OUT"
BASE=checkpoints/base_union14m_grid_fs8_best.pt

echo "########## [$(date)] 1/3 STRIDE SWEEP (base ckpt, chunk_w=128) ##########"
for stride in 32 48 80 112; do
  echo "=== stride$stride [$(date +%H:%M)] ==="
  $PY chunked_decode.py --checkpoint "$BASE" --stride "$stride" --amp \
      > "$OUT/stride_${stride}.log" 2>&1
  grep -E "axis:|acc=" "$OUT/stride_${stride}.log"
done

echo "########## [$(date)] 2/3 PHASE SWEEP (base ckpt, stride=64) ##########"
for phase in 16 32 48; do
  echo "=== phase$phase [$(date +%H:%M)] ==="
  $PY chunked_decode.py --checkpoint "$BASE" --stride 64 --phase "$phase" --amp \
      > "$OUT/phase_${phase}.log" 2>&1
  grep -E "axis:|acc=" "$OUT/phase_${phase}.log"
done

echo "########## [$(date)] 3/3 FINE-TUNE SEED VARIANCE (stride=64) ##########"
for seed_ckpt in v6at v6at_seed1 v6at_seed2; do
  echo "=== $seed_ckpt [$(date +%H:%M)] ==="
  $PY chunked_decode.py --checkpoint "checkpoints/finetune_u14mbase_${seed_ckpt}.pt" \
      --stride 64 --amp > "$OUT/seed_${seed_ckpt}.log" 2>&1
  grep -E "axis:|acc=" "$OUT/seed_${seed_ckpt}.log"
done

echo "########## [$(date)] HARDENING SWEEP DONE ##########"
echo "=== SMTR-format rows ==="
for f in "$OUT"/stride_*.log "$OUT"/phase_*.log "$OUT"/seed_*.log; do
  echo -n "$(basename "$f" .log): "
  $PY ltb_wavg.py "$f" --axis both | grep markdown
done
