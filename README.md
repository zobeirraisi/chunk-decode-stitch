# Out-of-Length Scene Text Recognition

Code for **"Out-of-Length Scene Text Recognition: A Two-Axis Diagnosis and a
Training-Free Fix"** (under review, Pattern Recognition).

Two parts, matching the paper:

- **Part I — diagnosis + representation-side fixes**: decompose out-of-length
  failure into the encoder-width and decoder-time axes; training-free rotary
  rescaling (NTK / PI / YaRN) and a lightweight weighted fine-tuning recipe.
- **Part II — `chunk-decode-stitch`**: a training-free decoding algorithm that
  turns any word-level autoregressive STR checkpoint into a long-text reader
  (overlapping training-width crops, independent in-distribution decodes,
  geometry-anchored edit-distance stitching).

## Layout

| Paper section | Script(s) |
|---|---|
| Base architecture (2D-RoPE-STR) | `model.py`, `dataset.py`, `train_2drope_str.py`, `evaluate.py` |
| §4.4 Failure characterization (LTB, 3 axes) | `eval_ltb.py`, `ltb_data.py` |
| §4.5 Training-free rescaling (NTK/PI/YaRN) | `eval_ltb.py --scale_mode {ntk,pi,yarn}` (implementations in `model.py`) |
| §4.6 Fine-tuning recipe | `run_finetune_longtext.py`, `longtext_synth.py` |
| §4.7 Synthetic length curve / Multi-Words | `synth_controlled_length.py`, `eval_synth_length.py`, `eval_u14m.py` |
| §4.11–4.12 Chunked decoding on LTB | `chunked_decode.py`, `run_chunked_pilot.sh` |
| §4.13 Robustness hardening (stride/phase/seed) | `run_chunked_hardening.sh` |
| §4.14 Compute cost | `chunked_cost.py` |
| §4.15 Non-horizontal / width gating | `chunked_u14m_nonhorizontal.py` |
| §4.16 Long + curved synthetic benchmark | `curved_longtext_synth.py`, `chunked_curved_test.py`, `run_chunked_extras.sh` |
| Architecture generality (public PARSeq) | `chunked_parseq.py` |
| Figures | `make_phase*_figure.py`, `make_paperc_figures.py`, `make_*_qualitative.py` |

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

## Citation

Citation entry will be added on acceptance; until then please cite the
manuscript by title.
