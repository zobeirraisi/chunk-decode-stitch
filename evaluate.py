"""
Word-accuracy evaluation over the standard 6 STR benchmarks.

Word accuracy = fraction of images whose full predicted transcription matches
the ground truth after case-insensitive, alphanumeric-only normalisation
(the protocol used by the DTRB benchmark and the STR literature).
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LabelConverter, LmdbDataset, EVAL_BENCHMARKS
from model import STRModel


def normalize(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


@torch.no_grad()
def evaluate_dataset(model, dataset, converter, device, batch_size=256,
                     num_workers=4, max_len=27, amp=True):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    correct, total = 0, 0
    for images, _, tgt_out in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model.greedy_decode(images, converter.SOS, converter.EOS, max_len)
        for p, gt in zip(pred.cpu(), tgt_out):
            pred_str = normalize(converter.decode(p.tolist()))
            gt_str = normalize(converter.decode(gt.tolist()))
            correct += int(pred_str == gt_str)
            total += 1
    return 100.0 * correct / max(total, 1)


def evaluate_all(model, data_root, converter, device, **kw):
    results = {}
    eval_root = os.path.join(data_root, "evaluation")
    for dirname, name in EVAL_BENCHMARKS.items():
        path = os.path.join(eval_root, dirname)
        if not os.path.isdir(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        ds = LmdbDataset(path, converter)
        acc = evaluate_dataset(model, ds, converter, device, **kw)
        results[name] = acc
        print(f"  {name:8s}: {acc:5.2f}%  (n={len(ds)})")
    if results:
        results["Avg"] = sum(results.values()) / len(results)
        print(f"  {'Avg':8s}: {results['Avg']:5.2f}%")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_lmdb_release")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", default=None, help="optional path to write results JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    converter = LabelConverter(max_len=cfg.get("max_len", 25))
    model = STRModel(
        vocab_size=converter.vocab_size,
        pe_type=cfg.get("pe_type", "2drope"),
        learnable_freq=cfg.get("learnable_freq", False),
        dim_split=cfg.get("dim_split", 0.5),
        feature_stride=cfg.get("feature_stride", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 6),
        num_decoder_layers=cfg.get("num_decoder_layers", 6),
        rope_base_h=cfg.get("rope_base_h", 10000.0),
        rope_base_w=cfg.get("rope_base_w", 10000.0),
        cross_pe=cfg.get("cross_pe", None),
        pretrained_backbone=False,
        max_len=converter.max_len,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.checkpoint} (pe_type={cfg.get('pe_type')}, "
          f"iter={ckpt.get('iteration', '?')})")

    results = evaluate_all(
        model, args.data_root, converter, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_len=converter.max_len + 2, amp=not args.no_amp,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
