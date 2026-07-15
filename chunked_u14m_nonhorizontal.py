"""Non-horizontal-layout stress test for chunked decoding (Paper C pilot,
docs/paperc-chunked-pilot.md caveat 2: LTB is mostly regular horizontal text
-- a horizontal sliding window is a strong prior for it specifically; this
was flagged untested).

Union14M-B's Curve and Multi-Oriented subsets exercise exactly the layout the
chunker assumes least (text that bends or is rotated within the crop). Their
labels are capped at the training max_len (<=25 chars, `LmdbDataset` filters
out anything longer) and their images are short-caption crops -- almost every
sample fits in a single 128px chunk (confirmed below), so this is NOT a test
of multi-chunk stitching under curved layouts (no long+curved dataset exists
to test that; would need a new synthetic generator, out of scope for this
pilot). What it DOES test, honestly: does the chunker's own resize policy
(aspect-preserving resize, then upscale the crop to 128px if narrower) ever
*hurt* accuracy relative to the standard training-matched squash-to-128x32
resize, when the text is curved/rotated rather than straight. `plain_read`
therefore uses the same squash transform as `dataset.build_transform`
(non-augmented), matching `eval_u14m.py`'s own evaluation protocol exactly,
not `eval_ltb.py`'s aspect-preserving one (which is designed for wide/long
text and is a poor match for these short, often-narrow images).

Usage:
  .venv/bin/python chunked_u14m_nonhorizontal.py \
      --checkpoint checkpoints/base_union14m_grid_fs8_best.pt
"""
import argparse
import io
import os

import lmdb
import torch
from PIL import Image

from chunked_decode import decode_chunks, stitch
from dataset import IMAGENET_NORM, LabelConverter
from eval_ltb import extend_decoder_pe, levenshtein, MAX_DECODE_LEN
from eval_u14m import load
from torchvision import transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TO_TENSOR = T.Compose([T.ToTensor(), IMAGENET_NORM])
U14M_ROOT = os.path.expanduser("~/Documents/RESEARCH/STR-NEXT/data/u14m")
SUBSETS = ["curve", "multi_oriented"]  # data/u14m/<name> directory names


def iter_raw_lmdb(root: str, conv: LabelConverter, limit: int = 0):
    """Yield (PIL image, cleaned label), bypassing the squash transform."""
    env = lmdb.open(root, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n = int(txn.get(b"num-samples"))
        n = min(n, limit) if limit else n
        for i in range(1, n + 1):
            label = txn.get(f"label-{i:09d}".encode())
            imgbuf = txn.get(f"image-{i:09d}".encode())
            if label is None or imgbuf is None:
                continue
            label = label.decode("utf-8", "ignore")
            cleaned = conv.clean(label)
            if not cleaned or len(cleaned) > conv.max_len:
                continue
            try:
                img = Image.open(io.BytesIO(imgbuf)).convert("RGB")
            except Exception:
                continue
            yield img, cleaned
    env.close()


@torch.no_grad()
def plain_read(model, conv, img, amp):
    # standard training-matched squash resize (dataset.build_transform,
    # non-augmented), NOT eval_ltb's aspect-preserving resize -- these are
    # short single-word/phrase images, not the wide long-text case that
    # resize_aspect_preserving is designed for.
    image = img.resize((128, 32), Image.BILINEAR)
    tensor = TO_TENSOR(image).unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", enabled=amp and DEVICE.type == "cuda"):
        pred = model.greedy_decode(tensor, conv.SOS, conv.EOS, max_len=conv.max_len + 2)
    return conv.decode(pred[0].tolist())


@torch.no_grad()
def chunked_read_img(model, conv, img, chunk_w, stride, amp):
    texts, xs, _ = decode_chunks(model, conv, img, chunk_w, stride, amp)
    return stitch(texts, xs, chunk_w), len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--u14m_root", default=U14M_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="0 = full subset")
    ap.add_argument("--chunk_w", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    model, conv, pe_type = load(args.checkpoint)
    model.to(DEVICE).eval()
    extend_decoder_pe(model, MAX_DECODE_LEN, train_len=conv.max_len + 2,
                      scale_mode="none")

    print(f"checkpoint={args.checkpoint} pe_type={pe_type}")
    for dirname in SUBSETS:
        path = os.path.join(args.u14m_root, dirname)
        if not os.path.isdir(path):
            print(f"  [skip] {dirname}: {path} not found"); continue
        plain_correct = chunk_correct = 0
        plain_cer_n = plain_cer_d = chunk_cer_n = chunk_cer_d = 0
        multi_chunk_n = 0
        n = 0
        for img, gt in iter_raw_lmdb(path, conv, args.limit):
            n += 1
            p_pred = plain_read(model, conv, img, args.amp)
            c_pred, n_chunks = chunked_read_img(model, conv, img, args.chunk_w,
                                                args.stride, args.amp)
            multi_chunk_n += n_chunks > 1
            plain_correct += p_pred == gt
            chunk_correct += c_pred == gt
            plain_cer_n += levenshtein(p_pred, gt); plain_cer_d += len(gt)
            chunk_cer_n += levenshtein(c_pred, gt); chunk_cer_d += len(gt)
        if n == 0:
            print(f"  [empty] {dirname}"); continue
        print(f"  {dirname:14s} (n={n}, {multi_chunk_n} multi-chunk / "
              f"{n - multi_chunk_n} single-chunk):")
        print(f"    plain:   acc={100*plain_correct/n:5.2f}%  "
              f"cer={100*plain_cer_n/max(plain_cer_d,1):5.2f}%")
        print(f"    chunked: acc={100*chunk_correct/n:5.2f}%  "
              f"cer={100*chunk_cer_n/max(chunk_cer_d,1):5.2f}%")


if __name__ == "__main__":
    main()
