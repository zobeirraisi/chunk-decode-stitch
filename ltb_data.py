"""LTB (long-text benchmark) data access.

Bucket list files ship with the benchmark download (data/ltb/*.txt), one
line per sample as `relative/image/path.jpg\tlabel text`. They are already
disjoint by transcription length (26-35 / 36-55 / 56+, with 70+ as a further
subset of 56+ -- verified to sum to the full 4789-sample set), so we use
them as-is instead of re-bucketing from the LMDB copy of the same data.
"""
import os

from PIL import Image

LTB_BUCKETS = {
    "26-35": "ultra_long_26_35_list.txt",
    "36-55": "ultra_long_36_55_list.txt",
    "56+": "ultra_long_56_list.txt",
    "70+": "ultra_long_70_list.txt",
}


def load_bucket(ltb_root: str, bucket: str):
    """Return [(absolute_image_path, raw_label), ...] for one length bucket."""
    list_path = os.path.join(ltb_root, LTB_BUCKETS[bucket])
    samples = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            rel_path, label = line.rstrip("\n").split("\t", 1)
            samples.append((os.path.join(ltb_root, rel_path), label))
    return samples


def resize_aspect_preserving(img: Image.Image, img_h: int = 32,
                              feature_stride: int = 8,
                              max_w: int = 1600) -> Image.Image:
    """Resize to a fixed height with width scaled to preserve aspect ratio,
    rounded up to a multiple of feature_stride for clean CNN tokenization.
    Caps width at max_w to bound memory on pathological (very wide) inputs.
    This is the width-extrapolation input: no squashing to a fixed 128px."""
    w, h = img.size
    new_w = round(w * img_h / h)
    new_w = min(new_w, max_w)
    new_w = ((new_w + feature_stride - 1) // feature_stride) * feature_stride
    new_w = max(new_w, feature_stride)
    return img.resize((new_w, img_h), Image.BILINEAR)
