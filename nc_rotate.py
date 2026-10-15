"""Rotation rule for portrait images (follow-up to the single-crop failure of
the error taxonomy): every LTB image narrow enough to fit into one crop is read
wrong, and about half of them are vertical text. Label-free rule: for an image
whose original height exceeds its width, also read it rotated by 90 degrees
clockwise and counter-clockwise (aspect-preserving resize, crop grid, stride
64), and keep whichever of the three reads has the highest mean per-character
decoder confidence. The unrotated read comes from the existing nc_dump.py dumps.

This script dumps the two rotated reads for every portrait image; the scoring
is in nc_rotate_eval.py. Run from code/:
  .venv/bin/python nc_rotate.py --model ours:checkpoints/X.pt --tag base --src ltb --amp
  .venv/bin/python nc_rotate.py --model ours:checkpoints/X.pt --tag base --src all --amp
"""
import argparse
import time

from PIL import Image

from ltb_data import resize_aspect_preserving
from dataset import LabelConverter
from nc_common import write_jsonl
from nc_dump import (CHUNK_DECODE_LEN, crops_for, iter_ltb, iter_short,
                     make_reader, run_batched)

STRIDE = 64


def portrait_only(items):
    """Load each image once and keep only portrait ones (height > width)."""
    for meta, loader in items:
        img = loader()
        if img.height > img.width:
            yield meta, (lambda im=img: im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", default="ltb", help="ltb | std6 | u14m | all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--crop_batch", type=int, default=256)
    ap.add_argument("--out_dir", default="results/nc/dumps")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    reader = make_reader(args.model, args.amp)
    conv = LabelConverter()
    t0 = time.time()
    items = iter_ltb(conv, args.limit) if args.src == "ltb" else \
        iter_short(conv, args.limit, args.src)

    def build(meta, img):
        crops_all = []
        for name, angle in (("cw", -90), ("ccw", 90)):
            rot = img.rotate(angle, expand=True)
            resized = resize_aspect_preserving(rot)
            xs, crops = crops_for(resized, STRIDE)
            meta[f"xs_{name}"], meta[f"w_{name}"] = xs, resized.size[0]
            crops_all += crops
        meta["n_cw"] = len(meta["xs_cw"])
        return meta, crops_all

    recs = []
    for meta, reads in run_batched(reader, portrait_only(items), args.crop_batch,
                                   build, CHUNK_DECODE_LEN):
        n = meta.pop("n_cw")
        meta["chunks_cw"] = [list(r) for r in reads[:n]]
        meta["chunks_ccw"] = [list(r) for r in reads[n:]]
        recs.append(meta)
    path = f"{args.out_dir}/{args.tag}_rotate_{args.src}.jsonl.gz"
    write_jsonl(path, recs)
    print(f"wrote {len(recs)} portrait records -> {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
