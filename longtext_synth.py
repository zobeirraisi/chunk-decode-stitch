"""Synthetic long text on a wide canvas, for Phase-3 fine-tuning.

Unlike curved_synth.py (fixed 128px canvas, short curved words, exact
oracle geometry for the CurviRoPE experiment), this renders STRAIGHT text
of RANDOM length (3-90 chars) onto a single fixed WIDE canvas (512px). A
fixed canvas -- rather than a canvas sized per-sample to the text -- means
every sample in this dataset has identical image shape, so it can be
batched with the default DataLoader collation, no custom padding/collate
function needed. Font size is chosen dynamically per sample so arbitrarily
long strings always fit within the canvas.
"""
import random
import string

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CHARS = string.ascii_lowercase + string.digits
CANVAS_W, CANVAS_H = 512, 32
FONT_SIZES = (22, 18, 14, 11, 9, 7)  # largest-first; first one that fits wins
MARGIN = 4
_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_font_cache = {}


def _font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def render_longtext(text: str) -> Image.Image:
    """White text on a dark canvas, left-aligned, vertically centered.
    Picks the largest font size (from FONT_SIZES) whose rendered width
    fits within CANVAS_W - 2*MARGIN. If none of FONT_SIZES fit (only
    possible for wide-character strings outside the dataset's actual
    uniform-random alphanumeric alphabet), keeps shrinking one point at a
    time down to size 1, which PIL can always render -- so this never
    overflows, for any input."""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (random.randint(0, 60),) * 3)
    d = ImageDraw.Draw(img)
    limit = CANVAS_W - 2 * MARGIN
    chosen_font = None
    for size in FONT_SIZES:
        f = _font(size)
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        if (r - l) <= limit:
            chosen_font = f
            break
    if chosen_font is None:
        for size in range(FONT_SIZES[-1] - 1, 0, -1):
            f = _font(size)
            l, t, r, b = d.textbbox((0, 0), text, font=f)
            chosen_font = f
            if (r - l) <= limit:
                break
    l, t, r, b = d.textbbox((0, 0), text, font=chosen_font)
    tw, th = r - l, b - t
    x = MARGIN - l
    y = (CANVAS_H - th) / 2 - t
    d.text((x, y), text, fill=(255, 255, 255), font=chosen_font)
    return img


class LongTextSynthDataset(torch.utils.data.Dataset):
    """Yields (normalized_img, tgt_in, tgt_out) -- plain 2drope fine-tuning
    samples, no chart/oracle fields.

    squash_w: if set (e.g. 128), the rendered wide canvas is resized down to
    this width before normalization -- i.e. long text on a NARROW image,
    matching eval_ltb.py's "decoder axis" condition. Without this, a
    fine-tuning recipe that only ever pairs wide images with long targets
    (and narrow images with short targets) teaches the model to use image
    width as a proxy for target length rather than reading content -- this
    is how that combination gets covered during training too."""

    def __init__(self, converter, n_samples=50000, min_len=3, max_len=90, seed=0,
                 squash_w=None):
        self.conv = converter
        self.n = n_samples
        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.squash_w = squash_w

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx)
        length = rng.randint(self.min_len, self.max_len)
        text = "".join(rng.choice(CHARS) for _ in range(length))
        random.seed(self.seed * 1_000_003 + idx)
        pil = render_longtext(text)
        if self.squash_w is not None:
            pil = pil.resize((self.squash_w, CANVAS_H), Image.BILINEAR)
        img = _NORM(T.ToTensor()(pil))
        tgt_in, tgt_out = self.conv.encode(text)
        return img, tgt_in, tgt_out
