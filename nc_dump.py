"""GPU pass of the analysis pipeline: dump reads + per-char confidences.

One run per model writes JSONL dumps (format in nc_common.py) that every
CPU-side analysis reuses:
  --mode chunks   LTB (or the synthetic dev set) cropped at each --strides;
                  stitcher ablation, confidence stitching,
                  error taxonomy, adaptive stride, model generality
  --mode decoder  LTB / synthetic decoder axis: squash to 128x32, decode budget
                  110, plain read; resolution-controlled decoder analysis
                  and the plain branch of the LTB gate sweep
  --mode gate     standard-6 + Union14M-B: plain squash read AND stride-64 crop
                  reads for every image; width-gate threshold sweep

Models:  --model ours:<checkpoint.pt>   our 2D-RoPE-STR checkpoints
         --model hub:<name>             baudm/parseq torch.hub models
                                        (parseq, abinet, trba, vitstr, crnn),
                                        loaded from the local hub cache

Crops from many images are decoded in one batch (--crop_batch), unlike
chunked_decode.py's one-image-per-batch loop; greedy decoding is per-crop
deterministic, so the reads are identical to that script's (checked by
nc_stitch_ablation.py, which re-scores the published "full" stitcher).

Usage (from code/):
  .venv/bin/python nc_dump.py --model ours:checkpoints/base_union14m_grid_fs8_best.pt \
      --tag base --mode chunks --src ltb --strides 32 48 64 80 96 --amp
"""
import argparse
import io
import os
import time

import lmdb
import torch
from PIL import Image
from torchvision import transforms as T

from chunked_decode import chunk_offsets
from dataset import EVAL_BENCHMARKS, IMAGENET_NORM, LabelConverter, U14M_BENCHMARKS
from eval_ltb import MAX_DECODE_LEN, extend_decoder_pe
from eval_u14m import load
from ltb_data import load_bucket, resize_aspect_preserving
from nc_common import CHUNK_W, LTB_BUCKETS3, dump_path, write_jsonl
from synth_controlled_length import generate_length_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUB_DIR = os.path.expanduser("~/.cache/torch/hub/baudm_parseq_main")
LTB_ROOT = os.path.expanduser("~/Documents/RESEARCH/STR-NEXT/data/ltb")
U14M_ROOT = os.path.expanduser("~/Documents/RESEARCH/STR-NEXT/data/u14m")
STD_ROOT = "data_lmdb_release/evaluation"
SYNTH_LENGTHS = [26, 30, 35, 40, 45, 50, 56, 65, 75, 90, 100]
CHUNK_DECODE_LEN = 30  # per-crop budget, as in chunked_decode.py


# ------------------------------------------------------------------ models
class OursReader:
    """2D-RoPE-STR greedy decoding that also returns per-char confidences."""

    def __init__(self, ckpt: str, amp: bool):
        self.model, self.conv, _ = load(ckpt)
        self.model.to(DEVICE).eval()
        self.amp = amp
        self.tf = T.Compose([T.ToTensor(), IMAGENET_NORM])
        # grow the decoder table once to the largest budget used here
        # (lossless rebuild, scale_mode "none"; see chunked_decode.main)
        if getattr(self.model, "dec_self_pe", "none") != "1drope":
            extend_decoder_pe(self.model, MAX_DECODE_LEN,
                              train_len=self.conv.max_len + 2, scale_mode="none")

    @torch.no_grad()
    def read(self, images, max_len: int):
        batch = torch.stack([self.tf(im) for im in images]).to(DEVICE)
        conv, model = self.conv, self.model
        with torch.autocast("cuda", enabled=self.amp and DEVICE.type == "cuda"):
            memory = model.encode(batch)
            B = batch.size(0)
            seq = torch.full((B, 1), conv.SOS, dtype=torch.long, device=DEVICE)
            probs = []
            finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
            for _ in range(max_len):
                logits = model.decode(seq, memory)
                p, nxt = logits[:, -1, :].float().softmax(-1).max(-1)
                seq = torch.cat([seq, nxt[:, None]], dim=1)
                probs.append(p)
                finished |= nxt == conv.EOS
                if finished.all():
                    break
        ids = seq[:, 1:].tolist()
        probs = torch.stack(probs, 1).tolist()
        out = []
        for row, prow in zip(ids, probs):
            text, conf = [], []
            for idx, p in zip(row, prow):  # mirrors LabelConverter.decode
                if idx == conv.EOS:
                    break
                if idx in conv.idx_to_char:
                    text.append(conv.idx_to_char[idx]); conf.append(round(p, 4))
            out.append(("".join(text), conf))
        return out


class HubReader:
    """baudm/parseq torch.hub models; output cleaned to the 36-char protocol
    with confidences filtered in parallel. CTC (crnn): per-char confidence is
    the max frame probability over that character's best-path segment."""

    def __init__(self, name: str, amp: bool):
        self.model = torch.hub.load(HUB_DIR, name, source="local",
                                    pretrained=True).to(DEVICE).eval()
        self.tok = self.model.tokenizer
        self.is_ctc = type(self.tok).__name__ == "CTCTokenizer"
        self.conv = LabelConverter()
        self.amp = amp
        self.tf = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    def _clean(self, chars, conf):
        keep = [(c.lower(), p) for c, p in zip(chars, conf)
                if c.lower() in self.conv.char_to_idx]
        return "".join(c for c, _ in keep), [round(p, 4) for _, p in keep]

    @torch.no_grad()
    def read(self, images, max_len: int = 0):  # max_len: native cap
        batch = torch.stack([self.tf(im) for im in images]).to(DEVICE)
        with torch.autocast("cuda", enabled=self.amp and DEVICE.type == "cuda"):
            dist = self.model(batch).float().softmax(-1)
        out = []
        for d in dist:
            p, ids = d.max(-1)
            p, ids = p.tolist(), ids.tolist()
            chars, conf = [], []
            if self.is_ctc:
                prev = None
                for i, pi in zip(ids, p):
                    if i != prev and i != self.tok.blank_id:
                        chars.append(self.tok._itos[i]); conf.append(pi)
                    elif i == prev and i != self.tok.blank_id:
                        conf[-1] = max(conf[-1], pi)
                    prev = i
            else:
                for i, pi in zip(ids, p):
                    if i == self.tok.eos_id:
                        break
                    tok = self.tok._itos[i]
                    if len(tok) == 1:
                        chars.append(tok); conf.append(pi)
            out.append(self._clean(chars, conf))
        return out


def make_reader(spec: str, amp: bool):
    kind, _, arg = spec.partition(":")
    if kind == "ours":
        return OursReader(arg, amp)
    if kind == "hub":
        return HubReader(arg, amp)
    raise ValueError(f"--model must be ours:<ckpt> or hub:<name>, got {spec!r}")


# -------------------------------------------------------------------- data
def iter_ltb(conv, limit: int):
    for bucket in LTB_BUCKETS3:
        samples = load_bucket(LTB_ROOT, bucket)
        if limit:
            samples = samples[:limit]
        for path, label in samples:
            yield {"src": "ltb", "bucket": bucket, "id": os.path.relpath(path, LTB_ROOT),
                   "gt": conv.clean(label)}, (lambda p=path: Image.open(p).convert("RGB"))


def iter_synth(conv, limit: int, seed: int):
    """Synthetic dev set: seed 1 is disjoint from the paper's curve (seed 0),
    so stitcher hyperparameters are never tuned on any reported test set."""
    n = limit or 150
    for L in SYNTH_LENGTHS:
        for i, (img, phrase) in enumerate(generate_length_samples(L, n, seed=seed)):
            yield {"src": f"synth{seed}", "bucket": str(L), "id": f"{L}_{i}",
                   "gt": conv.clean(phrase)}, (lambda im=img: im)


def iter_lmdb(root: str, src: str, bucket: str, conv, limit: int):
    """Raw LMDB images; empty or >max_len labels are EXCLUDED (the standard
    evaluation protocol), never resampled."""
    env = lmdb.open(root, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n = int(txn.get(b"num-samples"))
        for i in range(1, n + 1):
            label = txn.get(f"label-{i:09d}".encode())
            buf = txn.get(f"image-{i:09d}".encode())
            if label is None or buf is None:
                continue
            gt = conv.clean(label.decode("utf-8", "ignore"))
            if not gt or len(gt) > conv.max_len:
                continue
            try:
                img = Image.open(io.BytesIO(buf)).convert("RGB")
            except Exception:
                continue
            yield {"src": src, "bucket": bucket, "id": i, "gt": gt}, (lambda im=img: im)
            limit -= 1
            if limit == 0:
                break
    env.close()


def iter_short(conv, limit: int, which: str):
    if which in ("std6", "all"):
        for d, name in EVAL_BENCHMARKS.items():
            yield from iter_lmdb(os.path.join(STD_ROOT, d), "std6", name, conv, limit)
    if which in ("u14m", "all"):
        for d, name in U14M_BENCHMARKS.items():
            yield from iter_lmdb(os.path.join(U14M_ROOT, d), "u14m", name, conv, limit)


# ------------------------------------------------------------------ passes
def crops_for(resized: Image.Image, stride: int):
    w = resized.size[0]
    xs = chunk_offsets(w, CHUNK_W, stride)
    crops = []
    for x in xs:
        c = resized.crop((x, 0, min(x + CHUNK_W, w), 32))
        if c.size[0] < CHUNK_W:  # single narrow image, as in chunked_decode
            c = c.resize((CHUNK_W, 32), Image.BILINEAR)
        crops.append(c)
    return xs, crops


def run_batched(reader, items, crop_batch: int, build, max_len: int):
    """items: iterable of (meta, loader). build(meta, img) -> (meta, [images]).
    Decodes images from many samples per forward pass; yields (meta, reads)."""
    pend, flat = [], []

    def flush():
        reads = reader.read([im for _, im in flat], max_len) if flat else []
        k = 0
        for meta, n in pend:
            yield meta, reads[k:k + n]
            k += n

    for meta, loader in items:
        meta, ims = build(meta, loader())
        pend.append((meta, len(ims)))
        flat.extend((None, im) for im in ims)
        if len(flat) >= crop_batch:
            yield from flush()
            pend, flat = [], []
    yield from flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short name used in dump file names")
    ap.add_argument("--mode", choices=["chunks", "decoder", "gate"], required=True)
    ap.add_argument("--src", default="ltb",
                    help="chunks/decoder: ltb | synth; gate: std6 | u14m | all")
    ap.add_argument("--strides", type=int, nargs="+", default=[64])
    ap.add_argument("--synth_seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="per bucket/subset (0 = all)")
    ap.add_argument("--crop_batch", type=int, default=256)
    ap.add_argument("--out_dir", default="results/nc/dumps")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    reader = make_reader(args.model, args.amp)
    conv = LabelConverter()
    t0 = time.time()

    def source():
        if args.mode == "gate":
            return iter_short(conv, args.limit, args.src)
        if args.src == "ltb":
            return iter_ltb(conv, args.limit)
        return iter_synth(conv, args.limit, args.synth_seed)

    src_name = args.src if args.src != "synth" else f"synth{args.synth_seed}"
    if args.mode == "decoder":
        # decoder axis exactly as eval_ltb.py: squash to 128x32, budget 110
        def build(meta, img):
            meta["w"] = resize_aspect_preserving(img).size[0]
            return meta, [img.resize((CHUNK_W, 32), Image.BILINEAR)]
        recs = [dict(m, pred=list(r[0])) for m, r in
                run_batched(reader, source(), args.crop_batch, build, MAX_DECODE_LEN)]
        path = dump_path(args.out_dir, args.tag, "decoder", src_name)
        write_jsonl(path, recs)
        print(f"wrote {len(recs)} records -> {path} ({time.time() - t0:.0f}s)")
        return

    for stride in args.strides:
        def build(meta, img, stride=stride):
            resized = resize_aspect_preserving(img)
            xs, crops = crops_for(resized, stride)
            meta = dict(meta, w=resized.size[0], xs=xs)
            if args.mode == "gate":  # plain squash read rides along as crop 0
                crops = [img.resize((CHUNK_W, 32), Image.BILINEAR)] + crops
            return meta, crops
        recs = []
        for meta, reads in run_batched(reader, source(), args.crop_batch, build,
                                       CHUNK_DECODE_LEN):
            if args.mode == "gate":
                # plain read uses the standard 27-step budget (conv.max_len+2,
                # as chunked_u14m_nonhorizontal.plain_read); greedy prefixes
                # are budget-independent, so truncating is exact
                t, c = reads[0]
                cap = conv.max_len + 2
                meta["plain"], reads = [t[:cap], c[:cap]], reads[1:]
            meta["chunks"] = [list(r) for r in reads]
            recs.append(meta)
        kind = "gate" if args.mode == "gate" else "chunks"
        path = dump_path(args.out_dir, args.tag, kind, src_name, stride)
        write_jsonl(path, recs)
        print(f"wrote {len(recs)} records -> {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
