"""Controlled-length synthetic eval set (Paper B Phase 1, deferred item).

LTB's 4,789 real images are bucketed only into 26-35/36-55/56+ -- coarse,
and quirky in ways real scraped text always is (OCR-adjacent labels,
non-Latin scripts cleaned away, uneven per-bucket counts). This module
generates a controlled complement: real dictionary-word phrases at exact,
finely-spaced target lengths (so the "cliff" can be read as a continuous
curve, not three points), with as many samples per length as we want (noise
goes down with N, unlike the fixed real-image counts).

Phrases are built from real words (not random character soup) so failure is
about length, not about the harder sub-problem of reading gibberish. Spaces
are kept in the rendered image for visual realism (real signage has them),
but the label used for scoring is the space-stripped, LabelConverter.clean()
form -- exactly how LTB's own space-containing raw labels are scored -- so
"target length" below always means the post-clean() character count.

Rendering: fixed font size (unlike longtext_synth.py's shrink-to-fit, which
is deliberately un-realistic for training-distribution coverage), image
width = actual rendered text width + margin. This makes image width scale
with text length the way a photographed sign does, so resize_aspect_
preserving (ltb_data.py) produces the same kind of width-axis extrapolation
LTB itself exhibits.
"""
import functools
import random

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 20
IMG_H = 32
MARGIN = 6
WORDLIST_PATH = "/usr/share/dict/words"
SEED_BASE = 999_000  # disjoint from longtext_synth.py's training seed (0)


@functools.lru_cache(maxsize=1)
def _wordlist():
    words = []
    with open(WORDLIST_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if 2 <= len(w) <= 10 and w.isalpha():
                words.append(w)
    return words


@functools.lru_cache(maxsize=1)
def _font():
    return ImageFont.truetype(FONT_PATH, FONT_SIZE)


def make_phrase(rng: random.Random, target_len: int) -> str:
    """Real-word phrase whose space-stripped length is exactly target_len."""
    words_pool = _wordlist()
    words, clean_len = [], 0
    while clean_len < target_len:
        w = rng.choice(words_pool)
        words.append(w)
        clean_len += len(w)
    overflow = clean_len - target_len
    if overflow > 0:
        last = words[-1]
        trimmed = last[: len(last) - overflow]
        words = words[:-1] + ([trimmed] if trimmed else [])
    return " ".join(words)


def render_controlled(text: str) -> Image.Image:
    """White text on a dark canvas, fixed font size, width fit to content."""
    font = _font()
    tmp = Image.new("RGB", (8, 8))
    l, t, r, b = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    img = Image.new("RGB", (tw + 2 * MARGIN, IMG_H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    x = MARGIN - l
    y = (IMG_H - th) / 2 - t
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    return img


def generate_length_samples(target_len: int, n: int, seed: int = 0):
    """Return [(PIL.Image, phrase), ...] of length n, deterministic per
    (target_len, seed), using a seed range disjoint from training data."""
    out = []
    for i in range(n):
        rng = random.Random(SEED_BASE + seed * 1_000_003 + target_len * 131 + i)
        phrase = make_phrase(rng, target_len)
        out.append((render_controlled(phrase), phrase))
    return out
