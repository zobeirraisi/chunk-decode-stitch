# Out-of-Length Scene Text Recognition

Code for **"Out-of-Length Scene Text Recognition: A Two-Axis Diagnosis and a
Training-Free Fix"** (under review).

Two parts, matching the paper:

- **Part I — diagnosis + representation-side fixes**: decompose out-of-length
  failure into the encoder-width and decoder-time axes; training-free rotary
  rescaling (NTK / PI / YaRN) and a lightweight weighted fine-tuning recipe.
- **Part II — `chunk-decode-stitch`**: a training-free decoding algorithm that
  turns any word-level autoregressive STR checkpoint into a long-text reader
  (overlapping training-width crops, independent in-distribution decodes,
  geometry-anchored edit-distance stitching).

## Layout

Section numbers refer to the current manuscript.

| Paper section | Script(s) |
|---|---|
| Base architecture (2D-RoPE-STR) | `model.py`, `dataset.py`, `train_2drope_str.py`, `evaluate.py` (`chart.py`, `warp_consistency.py` are imported by them; their code paths are not used in the paper) |
| §4.3 Failure characterization (LTB, 3 axes) | `eval_ltb.py`, `ltb_data.py` |
| §4.3.1 Controlled synthetic length curve | `synth_controlled_length.py`, `eval_synth_length.py` |
| §4.4 Training-free rescaling (NTK/PI/YaRN) | `eval_ltb.py --scale_mode {ntk,pi,yarn}` (implementations in `model.py`) |
| §4.5 Fine-tuning recipe | `run_finetune_longtext.py`, `longtext_synth.py` |
| §4.6 Full-convergence base, Union14M-B | `train_2drope_str.py` (command below), `eval_u14m.py` |
| §4.7 LTB W-Avg / A-Avg rows | `ltb_wavg.py` |
| §4.8 Decoder positional encoding | `train_2drope_str.py --dec_self_pe 1drope`, `eval_ltb.py --axis decoder --decoder_scale_mode {none,ntk,pi,yarn}`, `nc_decoder_control.py` |
| §4.9–4.10 Chunked decoding on LTB | `chunked_decode.py`, `run_chunked_pilot.sh`; public PARSeq: `chunked_parseq.py`; other `torch.hub` models (incl. CTC CRNN): `nc_dump.py --model hub:<name>` + `nc_stitch_ablation.py` |
| §4.11 Stitcher ablation, confidence stitching, adaptive stride | `nc_stitch_ablation.py` |
| §4.12 Error taxonomy and failure figure | `nc_error_taxonomy.py`, `nc_failure_figure.py` |
| §4.13 Portrait rule (vertical text) | `nc_rotate.py`, `nc_rotate_eval.py` |
| §4.14 Robustness (stride/phase/seed) and compute cost | `run_chunked_hardening.sh`, `chunked_cost.py` |
| §4.15 Non-horizontal and long-curved text | `chunked_u14m_nonhorizontal.py`, `curved_longtext_synth.py`, `chunked_curved_test.py`, `run_chunked_extras.sh` |
| §4.16 Width-gate sweep | `nc_gate_sweep.py` |
| Figures | `make_phase*_figure.py`, `make_paperc_figures.py`, `make_*_qualitative.py`, `make_nc_figures.py`, `make_nc_supp_figures.py` |

The analyses of §4.8 and §4.10–4.16 share one pipeline: a GPU pass
(`nc_dump.py`) writes every crop read with per-character confidences, and
CPU scripts (`nc_*.py`, helpers in `nc_common.py`) compute the tables from
these dumps. `run_nc_revision.sh` runs the whole pipeline (resumable;
`bash run_nc_revision.sh gpu|cpu|all`). The stitcher-ablation script asserts
that its "full" stitcher reproduces `chunked_decode.stitch` on every record.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Datasets (downloaded separately, paths are script arguments):

- **Union14M-L / Union14M-B**: https://github.com/Mountchicken/Union14M
- **LTB** (Long Text Benchmark, 4,789 images): released with SMTR,
  https://github.com/Topdu/OpenOCR (Du et al., AAAI 2025)
- Standard six benchmarks (IIIT5K, SVT, IC13, IC15, CUTE80, SVTP): standard
  LMDB releases.

## Quick start: chunked decoding

```bash
# our checkpoint
python chunked_decode.py --checkpoint <ckpt.pt> --stride 64 --amp

# any public PARSeq checkpoint via torch.hub (no training, no our-weights)
python chunked_parseq.py --stride 64 --amp
```

## Training the base checkpoints

Full-convergence base (§4.6; the second seed uses the same command with a
different `--run_name`; the rotary-decoder variant of §4.8 adds
`--dec_self_pe 1drope`):

```bash
python train_2drope_str.py \
  --union_root <Union14M-L-LMDB-Filtered> \
  --train_sources easy normal medium hard challenging \
  --data_root <standard-benchmark LMDB root> \
  --pe_type 2drope --feature_stride 8 --dim_split 0.5 \
  --num_encoder_layers 6 --num_decoder_layers 6 --max_len 25 \
  --rope_base_h 10000 --rope_base_w 10000 \
  --iters 200000 --batch_size 256 --lr 3e-4 --warmup 5000 \
  --weight_decay 0.01 --label_smoothing 0.1 --grad_clip 5.0 \
  --num_workers 8 --amp \
  --eval_every 10000 --save_every 10000 --log_every 200 \
  --run_name base_union14m_grid_fs8
```

## Portrait rule

```bash
python nc_rotate.py --model ours:<ckpt.pt> --tag base --src ltb --amp   # rotated reads
python nc_rotate_eval.py --tag base --src ltb                            # scoring
```

## Citation

Citation entry will be added on acceptance; until then please cite the
manuscript by title.
