"""Phase 3 -- fine-tune an existing trained checkpoint on mixed-length data
to test whether weight adaptation recovers the width/decoder-axis CER
degradation that Phase 1 found and Phase 2's training-free NTK scaling did
not fix.

v1 alternated 2 streams (narrow+short / wide+long) 50/50 and found a real
improvement on width/both axes, but a regression on the decoder axis
(narrow+long, never trained on). v2 added a 3rd stream (long text squashed
to 128px) at an equal 1/3 share to cover that combination -- this made
EVERY axis worse, including width/both, which don't even touch that
stream's image pipeline. Diagnosis: long text rendered then squashed 4x
down to 128px is well under 1.5px/char, close to physically unlearnable;
forcing the model to predict exact labels from near-noise input injects
harmful gradient noise broadly, and 1/3 of every batch being this condition
was enough to drag down overall training quality, not just fail to help
the one axis it targeted.

v3 (this version) keeps the 3rd stream but at a much smaller default share
(--stream_weights, default 0.45/0.45/0.10) -- light exposure to the
narrow+long combination without letting a near-unlearnable signal dominate
gradient updates. Streams (same 3 as v2, now weighted-random instead of a
fixed round-robin):
  short         : 128px squash (build_union14m_train, base checkpoint's tiers)
  long_wide     : 512px wide canvas (LongTextSynthDataset)
  long_squashed : 128px squash of long text (LongTextSynthDataset, squash_w=128)

Run from code/:
  .venv/bin/python run_finetune_longtext.py \
    --resume checkpoints/verify_grid_hard_best.pt \
    --union_root ../data/Union14M-L-LMDB-Filtered \
    --iters 20000 --batch_size 64 --lr 1e-4 --amp
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LabelConverter, build_union14m_train
from eval_ltb import extend_decoder_pe
from longtext_synth import LongTextSynthDataset
from model import STRModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * min(p, 1.0)))


def infinite(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--union_root", required=True)
    ap.add_argument("--train_sources", nargs="+", default=["hard", "challenging"])
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--long_min_len", type=int, default=26,
                     help="min synthetic long-text length; 26 (default) avoids "
                          "overlapping the short stream's 0-25 coverage so the "
                          "whole synthetic budget targets the extrapolation regime")
    ap.add_argument("--stream_weights", type=float, nargs=3, default=[0.45, 0.45, 0.10],
                     metavar=("SHORT", "LONG_WIDE", "LONG_SQUASHED"),
                     help="per-iteration sampling probabilities for the 3 "
                          "streams; need not be pre-normalized")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--run_name", default="finetune_longtext_grid")
    ap.add_argument("--seed", type=int, default=0,
                     help="seed for torch/random and the synthetic-stream RNGs; "
                          "vary it for a seed-variance study")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Build at the checkpoint's original max_len=25 for a strict state_dict
    # load, then grow the decoder table afterward -- extend_decoder_pe is a
    # deterministic buffer swap (no weights lost), same mechanism eval_ltb.py
    # already uses training-free at eval time.
    load_conv = LabelConverter(max_len=25)
    ck = torch.load(args.resume, map_location=DEVICE, weights_only=False)
    cfg = ck.get("config", {})
    model = STRModel(
        vocab_size=load_conv.vocab_size, pe_type=cfg.get("pe_type", "2drope"),
        d_model=cfg.get("d_model", 512), n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 2048),
        learnable_freq=cfg.get("learnable_freq", False),
        dim_split=cfg.get("dim_split", 0.5), n_ctrl=cfg.get("n_ctrl", 8),
        feature_stride=cfg.get("feature_stride", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 6),
        num_decoder_layers=cfg.get("num_decoder_layers", 6),
        rope_base_h=cfg.get("rope_base_h", 10000.0),
        rope_base_w=cfg.get("rope_base_w", 10000.0),
        cross_pe=cfg.get("cross_pe", None), pretrained_backbone=False,
        dec_self_pe=cfg.get("dec_self_pe", "none"),
        max_len=load_conv.max_len,
    ).to(DEVICE)
    model.load_state_dict(ck["model"])
    if cfg.get("dec_self_pe", "none") == "1drope":
        # rotary decoder carries decode-time position in-attention -> no additive
        # table to grow; the long stream trains those rotations directly
        print(f"Resumed from {args.resume}, rotary decoder (no PE table to extend)")
    else:
        extend_decoder_pe(model, args.max_len)
        print(f"Resumed from {args.resume}, extended decoder to max_len={args.max_len}")

    conv = LabelConverter(max_len=args.max_len)
    short_set = build_union14m_train(args.union_root, conv,
                                     tiers=tuple(args.train_sources))
    long_set = LongTextSynthDataset(conv, n_samples=200_000, seed=args.seed + 1,
                                    min_len=args.long_min_len)
    squashed_long_set = LongTextSynthDataset(conv, n_samples=200_000, seed=args.seed + 2,
                                             min_len=args.long_min_len, squash_w=128)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=args.num_workers > 0)

    short_loader = make_loader(short_set, shuffle=True)
    long_loader = make_loader(long_set, shuffle=False)
    squashed_long_loader = make_loader(squashed_long_set, shuffle=False)
    short_it = infinite(short_loader)
    long_it = infinite(long_loader)
    squashed_long_it = infinite(squashed_long_loader)
    streams = [("short", short_it), ("long_wide", long_it),
              ("long_squashed", squashed_long_it)]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=conv.PAD, label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    model.train()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(it, args.iters, args.lr, args.warmup)
        stream, it_source = random.choices(streams, weights=args.stream_weights, k=1)[0]
        img, ti, to = next(it_source)
        img = img.to(DEVICE, non_blocking=True)
        ti, to = ti.to(DEVICE), to.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=args.amp):
            logits = model(img, ti)
            loss = crit(logits.reshape(-1, logits.size(-1)), to.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        if it % 100 == 0 or it == args.iters:
            print(f"  it {it}/{args.iters} [{stream}] loss {loss.item():.3f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    save_cfg = {
        "pe_type": cfg.get("pe_type", "2drope"),
        "d_model": cfg.get("d_model", 512), "n_heads": cfg.get("n_heads", 8),
        "d_ff": cfg.get("d_ff", 2048),
        "learnable_freq": cfg.get("learnable_freq", False),
        "dim_split": cfg.get("dim_split", 0.5),
        "n_ctrl": cfg.get("n_ctrl", 8),
        "feature_stride": cfg.get("feature_stride", 8),
        "num_encoder_layers": cfg.get("num_encoder_layers", 6),
        "num_decoder_layers": cfg.get("num_decoder_layers", 6),
        "rope_base_h": cfg.get("rope_base_h", 10000.0),
        "rope_base_w": cfg.get("rope_base_w", 10000.0),
        "cross_pe": cfg.get("cross_pe", None),
        "dec_self_pe": cfg.get("dec_self_pe", "none"),
        "max_len": args.max_len,
    }
    save_path = os.path.join(args.save_dir, f"{args.run_name}.pt")
    torch.save({"model": model.state_dict(), "config": save_cfg,
                "iteration": args.iters}, save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
