"""
Data loading for 2D-RoPE-STR.

Reads the standard Deep-Text-Recognition-Benchmark (DTRB) LMDB datasets
(`data_lmdb_release`) whose keys are:
    num-samples            -> integer count
    image-%09d             -> raw (encoded) image bytes, 1-indexed
    label-%09d             -> ground-truth transcription (bytes)

The label converter uses an attention-decoder vocabulary:
    index 0            -> <PAD>
    index 1..C         -> characters of CHARSET
    index C+1          -> <SOS>  (start of sequence / [GO])
    index C+2          -> <EOS>  (end of sequence / [s])
so `vocab_size == len(CHARSET) + 3`.
"""

import io
import random

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as T

# Case-insensitive 36-symbol alphanumeric set (DTRB default).
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class LabelConverter:
    """Maps transcriptions <-> index tensors for an attention decoder."""

    def __init__(self, charset: str = CHARSET, max_len: int = 25):
        self.charset = charset
        self.max_len = max_len
        self.char_to_idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 == PAD
        self.idx_to_char = {i + 1: c for i, c in enumerate(charset)}
        self.PAD = 0
        self.SOS = len(charset) + 1
        self.EOS = len(charset) + 2
        self.vocab_size = len(charset) + 3

    def clean(self, text: str) -> str:
        """Lower-case and drop out-of-charset symbols (DTRB-style filtering)."""
        text = text.lower()
        return "".join(c for c in text if c in self.char_to_idx)

    def encode(self, text: str):
        """Return (tgt_input, tgt_output) index tensors of length max_len+2.

        tgt_input  = [SOS, c1, ..., cn,  PAD ...]
        tgt_output = [c1,  ..., cn, EOS, PAD ...]
        """
        text = self.clean(text)[: self.max_len]
        ids = [self.char_to_idx[c] for c in text]
        tgt_in = [self.SOS] + ids
        tgt_out = ids + [self.EOS]
        pad_len = self.max_len + 2
        tgt_in = tgt_in + [self.PAD] * (pad_len - len(tgt_in))
        tgt_out = tgt_out + [self.PAD] * (pad_len - len(tgt_out))
        return torch.tensor(tgt_in), torch.tensor(tgt_out)

    def decode(self, indices) -> str:
        """Greedy index sequence -> string, stopping at <EOS>."""
        chars = []
        for idx in indices:
            idx = int(idx)
            if idx == self.EOS:
                break
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return "".join(chars)


def build_transform(img_h: int = 32, img_w: int = 128, augment: bool = False):
    """Image preprocessing. With ``augment`` (training only) we add the
    geometric/photometric perturbations described in the paper, which are
    especially helpful for curved/perspective text (CUTE80, SVTP)."""
    norm = IMAGENET_NORM
    if augment:
        # Resize FIRST: some STR crops are only a few pixels wide, and
        # RandomPerspective's lstsq solve fails on such degenerate inputs.
        # Operating on the fixed 32x128 image keeps the warp well-conditioned.
        return T.Compose([
            T.Resize((img_h, img_w)),
            T.RandomApply([T.RandomRotation(15, expand=False)], p=0.5),
            T.RandomApply([T.RandomPerspective(distortion_scale=0.3, p=1.0)], p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize((img_h, img_w)), T.ToTensor(), norm])


class LmdbDataset(Dataset):
    """A single DTRB LMDB dataset.

    The LMDB environment is opened lazily inside each worker process because
    `lmdb.Environment` objects cannot be pickled across the DataLoader fork.
    Out-of-charset / over-length labels are handled on the fly by resampling,
    which avoids an expensive full-database scan at construction time.
    """

    def __init__(self, root: str, converter: LabelConverter,
                 img_h: int = 32, img_w: int = 128, augment: bool = False):
        self.root = root
        self.converter = converter
        self.transform = build_transform(img_h, img_w, augment)
        # Always-available clean transform, used as a fallback if a random
        # augmentation throws on a pathological sample (keeps long runs alive).
        self.clean_transform = build_transform(img_h, img_w, False)
        self._env = None
        env = lmdb.open(root, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.n_samples = int(txn.get(b"num-samples"))
        env.close()

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(self.root, readonly=True, lock=False,
                                  readahead=False, meminit=False)
        return self._env

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # LMDB keys are 1-indexed.
        for _ in range(10):
            key = index + 1
            with self.env.begin(write=False) as txn:
                label = txn.get(f"label-{key:09d}".encode())
                imgbuf = txn.get(f"image-{key:09d}".encode())
            if label is None or imgbuf is None:
                index = random.randint(0, self.n_samples - 1)
                continue
            label = label.decode("utf-8", "ignore")
            cleaned = self.converter.clean(label)
            if len(cleaned) == 0 or len(cleaned) > self.converter.max_len:
                index = random.randint(0, self.n_samples - 1)
                continue
            try:
                img = Image.open(io.BytesIO(imgbuf)).convert("RGB")
            except Exception:
                index = random.randint(0, self.n_samples - 1)
                continue
            try:
                image = self.transform(img)
            except Exception:
                image = self.clean_transform(img)
            tgt_in, tgt_out = self.converter.encode(label)
            return image, tgt_in, tgt_out
        # Fallback: blank sample (should be extremely rare).
        image = torch.zeros(3, 32, 128)
        tgt_in, tgt_out = self.converter.encode("")
        return image, tgt_in, tgt_out


def build_training_set(data_root: str, converter: LabelConverter,
                       sources=("MJ", "ST"), img_h=32, img_w=128, augment=False):
    """Concatenate the requested training LMDBs (MJ has train/val/test subdirs)."""
    import os
    subsets = []
    for src in sources:
        if src == "MJ":
            for split in ("train", "val", "test"):
                p = os.path.join(data_root, "training", "MJ", split)
                if os.path.isdir(p):
                    subsets.append(LmdbDataset(p, converter, img_h, img_w, augment))
        else:
            p = os.path.join(data_root, "training", src)
            if os.path.isdir(p):
                subsets.append(LmdbDataset(p, converter, img_h, img_w, augment))
    if not subsets:
        raise FileNotFoundError(f"No training LMDBs found under {data_root} for {sources}")
    return ConcatDataset(subsets) if len(subsets) > 1 else subsets[0]


UNION14M_TIERS = ("easy", "normal", "medium", "hard", "challenging")

def build_union14m_train(union_root, converter, tiers=UNION14M_TIERS,
                         img_h=32, img_w=128, augment=False):
    """Concatenate the Union14M-L filtered training tiers (each an LMDB)."""
    import os
    subsets = []
    for tier in tiers:
        p = os.path.join(union_root, f"filter_train_{tier}")
        if os.path.isdir(p):
            subsets.append(LmdbDataset(p, converter, img_h, img_w, augment))
    if not subsets:
        raise FileNotFoundError(f"No Union14M-L tiers found under {union_root} for {tiers}")
    return ConcatDataset(subsets) if len(subsets) > 1 else subsets[0]


# Standard 6-benchmark evaluation suite (directory name -> display name).
EVAL_BENCHMARKS = {
    "IIIT5k_3000": "IIIT5K",
    "SVT": "SVT",
    "IC13_1015": "IC13",
    "IC15_2077": "IC15",
    "CUTE80": "CUTE80",
    "SVTP": "SVTP",
}


# Union14M-B hard-generalization suite (dir name -> display name).
U14M_BENCHMARKS = {
    "artistic": "Artistic", "contextless": "Contextless", "curve": "Curve",
    "general": "General", "multi_oriented": "Multi-Oriented",
    "multi_words": "Multi-Words", "salient": "Salient",
}


class DummySTRDataset(Dataset):
    """Random images/labels for fast smoke-testing of the pipeline."""

    def __init__(self, converter: LabelConverter, num_samples=512):
        self.converter = converter
        self.num_samples = num_samples
        self.chars = converter.charset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 128)
        length = random.randint(3, self.converter.max_len)
        text = "".join(random.choice(self.chars) for _ in range(length))
        tgt_in, tgt_out = self.converter.encode(text)
        return image, tgt_in, tgt_out
