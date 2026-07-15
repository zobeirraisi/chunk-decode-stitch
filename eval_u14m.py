"""Evaluate a checkpoint on the Union14M-B hard-generalization subsets.

This is the curvature go/no-go: does the *learned* CurviRoPE chart help on real
curved / multi-oriented text (the method's actual arena, thousands of samples
per subset) vs the grid baseline? Uses each model's own predicted chart (no
oracle). Run for both pilot checkpoints and diff.

  .venv/bin/python eval_u14m.py --checkpoint checkpoints/pilot_curvi_best.pt
"""
import argparse, os
import torch

from dataset import LabelConverter, LmdbDataset, U14M_BENCHMARKS
from evaluate import evaluate_dataset
from model import STRModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
U14M_ROOT = os.path.expanduser("~/Documents/RESEARCH/STR-NEXT/data/u14m")


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck.get("config", {})
    conv = LabelConverter(max_len=cfg.get("max_len", 25))
    model = STRModel(
        vocab_size=conv.vocab_size, pe_type=cfg.get("pe_type", "2drope"),
        learnable_freq=cfg.get("learnable_freq", False),
        dim_split=cfg.get("dim_split", 0.5), n_ctrl=cfg.get("n_ctrl", 8),
        feature_stride=cfg.get("feature_stride", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 6),
        num_decoder_layers=cfg.get("num_decoder_layers", 6),
        rope_base_h=cfg.get("rope_base_h", 10000.0),
        rope_base_w=cfg.get("rope_base_w", 10000.0),
        cross_pe=cfg.get("cross_pe", None), pretrained_backbone=False,
        dec_self_pe=cfg.get("dec_self_pe", "none"),
        max_len=conv.max_len,
    ).to(DEVICE)
    model.load_state_dict(ck["model"])
    return model, conv, cfg.get("pe_type")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    model, conv, pe = load(args.checkpoint)
    print(f"Loaded {args.checkpoint} (pe_type={pe})", flush=True)
    res = {}
    for d, name in U14M_BENCHMARKS.items():
        path = os.path.join(U14M_ROOT, d)
        if not os.path.isdir(path):
            print(f"  [skip] {name}: {path}"); continue
        ds = LmdbDataset(path, conv)
        acc = evaluate_dataset(model, ds, conv, DEVICE, batch_size=args.batch_size,
                               num_workers=args.num_workers, max_len=conv.max_len + 2)
        res[name] = acc
        print(f"  {name:14s}: {acc:5.2f}%  (n={len(ds)})", flush=True)
    res["Avg"] = sum(res.values()) / len(res)
    print(f"  {'Avg':14s}: {res['Avg']:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
