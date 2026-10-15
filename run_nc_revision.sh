#!/usr/bin/env bash
# Analysis experiments: stitcher ablation, error taxonomy, decoder-axis control,
# width-gate sweep, public hub models, Union14M-B.
#
# Stage 1 (GPU, inference only, ~1-2 h on one 24 GB card): dump per-crop reads
# with per-character confidences for every model/dataset/stride needed.
# Stage 2 (CPU, minutes): stitcher ablation, confidence-aware stitching,
# adaptive stride, error taxonomy, decoder-axis control, gate sweep, figures.
# Stage 3 (GPU): Union14M-B re-runs under the corrected evaluation protocol
# (invalid labels excluded, not resampled).
#
# Every step skips work whose output already exists, so the script can be
# re-run after an interruption. Run from code/:
#   bash run_nc_revision.sh            # all stages
#   bash run_nc_revision.sh gpu        # stage 1 + 3 only
#   bash run_nc_revision.sh cpu        # stage 2 only (needs the dumps)
# Pick the GPU with CUDA_VISIBLE_DEVICES=<id> if the machine has several.
# Smoke test (CPU, a few samples per subset, separate output dir, no u14m):
#   CUDA_VISIBLE_DEVICES= NC_LIMIT=2 NC_OUT=/tmp/nc_smoke bash run_nc_revision.sh smoke
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
OUT=${NC_OUT:-results/nc}
DUMPS=$OUT/dumps
LOG=${NC_OUT:+$NC_OUT/logs}
LOG=${LOG:-logs/nc_revision}
LIMIT=()
[[ -n "${NC_LIMIT:-}" ]] && LIMIT=(--limit "$NC_LIMIT")
mkdir -p "$DUMPS" "$LOG"
STAGE=${1:-all}

declare -A CKPT=(
  [base]=checkpoints/base_union14m_grid_fs8_best.pt
  [seed2]=checkpoints/base_union14m_grid_fs8_seed2_best.pt
  [ft]=checkpoints/finetune_u14mbase_v6at_long.pt
  [decrope]=checkpoints/base_union14m_decrope_fs8_best.pt
)
HUB_MODELS=(parseq crnn vitstr trba abinet)

have() { [[ -s "$1" ]]; }

dump() {  # dump <expected-output> <log-name> <nc_dump.py args...>
  local target=$1 name=$2; shift 2
  if have "$target"; then echo "[skip] $name"; return; fi
  echo "[run ] $name  ($(date +%H:%M:%S))"
  $PY nc_dump.py --amp --out_dir "$DUMPS" "${LIMIT[@]}" "$@" > "$LOG/$name.log" 2>&1
}

stage_gpu() {
  # Stitcher ablation / confidence stitching / adaptive stride: LTB crop reads. Base gets the full stride sweep; the other two
  # checkpoints get the strides the adaptive rule can pick.
  dump "$DUMPS/base_chunks_ltb_s96.jsonl.gz" base_chunks_ltb \
       --model "ours:${CKPT[base]}" --tag base --mode chunks --src ltb --strides 32 48 64 80 96
  for t in seed2 ft; do
    dump "$DUMPS/${t}_chunks_ltb_s64.jsonl.gz" "${t}_chunks_ltb" \
         --model "ours:${CKPT[$t]}" --tag "$t" --mode chunks --src ltb --strides 32 48 64
  done
  # synthetic DEV set (seed 1, disjoint from the paper's curve): selects the
  # stitcher hyperparameters and the adaptive-stride rule
  dump "$DUMPS/base_chunks_synth1_s64.jsonl.gz" base_chunks_synth1 \
       --model "ours:${CKPT[base]}" --tag base --mode chunks --src synth --synth_seed 1 --strides 32 48 64
  # Decoder-axis control: decoder-axis reads (squeezed, budget 110) on LTB and the paper's synthetic curve
  for t in base seed2 ft decrope; do
    dump "$DUMPS/${t}_decoder_ltb.jsonl.gz" "${t}_decoder_ltb" \
         --model "ours:${CKPT[$t]}" --tag "$t" --mode decoder --src ltb
  done
  dump "$DUMPS/base_decoder_synth0.jsonl.gz" base_decoder_synth0 \
       --model "ours:${CKPT[base]}" --tag base --mode decoder --src synth --synth_seed 0
  # Width-gate sweep: plain + chunked reads on the six standard benchmarks and Union14M-B
  dump "$DUMPS/base_gate_all_s64.jsonl.gz" base_gate_all \
       --model "ours:${CKPT[base]}" --tag base --mode gate --src all --strides 64
  # Generality: public torch.hub checkpoints (PARSeq reproduces the main table; CRNN is CTC)
  for m in "${HUB_MODELS[@]}"; do
    dump "$DUMPS/${m}_chunks_ltb_s64.jsonl.gz" "${m}_chunks_ltb" \
         --model "hub:$m" --tag "$m" --mode chunks --src ltb --strides 32 48 64
  done
}

stage_u14m() {
  # every Union14M-B number printed in the paper
  for ck in base_union14m_grid_fs8_best finetune_u14mbase_v6at finetune_u14mbase_v6at_long \
            verify_grid_hard_best finetune_longtext_grid_v6; do
    if have "$LOG/u14m_$ck.log"; then echo "[skip] u14m $ck"; continue; fi
    echo "[run ] u14m $ck  ($(date +%H:%M:%S))"
    $PY eval_u14m.py --checkpoint "checkpoints/$ck.pt" > "$LOG/u14m_$ck.log" 2>&1
  done
}

stage_cpu() {
  $PY nc_stitch_ablation.py --tags base seed2 ft "${HUB_MODELS[@]}" \
      --dumps "$DUMPS" --out "$OUT/stitch_ablation.json" | tee "$LOG/stitch_ablation.txt"
  for t in base seed2 ft parseq; do
    $PY nc_error_taxonomy.py --tag "$t" --dumps "$DUMPS" \
        --ablation "$OUT/stitch_ablation.json" --out "$OUT/error_taxonomy_{tag}.json" \
        | tee "$LOG/error_taxonomy_$t.txt"
  done
  $PY nc_decoder_control.py --tags base seed2 ft decrope --srcs ltb synth0 \
      --dumps "$DUMPS" --out "$OUT/decoder_control.json" | tee "$LOG/decoder_control.txt"
  for s in full conf; do
    $PY nc_gate_sweep.py --tag base --stitch "$s" --dumps "$DUMPS" \
        --out "$OUT/gate_sweep_{tag}_$s.json" | tee "$LOG/gate_sweep_$s.txt"
  done
  for s in full conf; do
    $PY nc_failure_figure.py --tag base --stitcher "$s" --per_class 2 --dumps "$DUMPS" \
        --taxonomy "$OUT/error_taxonomy_{tag}.json" \
        --out "$OUT/figures/failure_cases_{tag}_{stitcher}.pdf"
  done
  echo "summaries: $LOG/*.txt   json: $OUT/*.json   figures: $OUT/figures/"
}

case "$STAGE" in
  gpu) stage_gpu; stage_u14m ;;
  cpu) stage_cpu ;;
  all) stage_gpu; stage_u14m; stage_cpu ;;
  smoke) stage_gpu; stage_cpu ;;
  *) echo "usage: $0 [all|gpu|cpu|smoke]"; exit 1 ;;
esac
echo "done ($(date))"
