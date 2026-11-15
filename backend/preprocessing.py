"""Preprocessing inferensi. Port 1:1 dari notebook (bagian 2.9 Strip Processing)."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu

LINE_FILL = 0.65


def flatten_background(g):
    g = g.astype(np.uint8)
    k = max(15, (min(g.shape) // 6) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    out = np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255)
    lo, hi = np.percentile(out, 1), np.percentile(out, 50)
    return (np.clip((out - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def clean_gray(img):
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img.convert("RGBA"))
    g = flatten_background(np.array(img.convert("L")))
    ink = g < threshold_otsu(g)
    return g, ink


def find_lines(ink):
    prof = np.convolve(ink.sum(1).astype(np.float32), np.ones(5) / 5, mode="same")
    on = prof > max(0.2 * np.percentile(prof, 95), 1)
    edges = np.diff(np.r_[0, on.astype(int), 0])
    segs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    if not segs:
        return [(0, ink.shape[0])]
    med = np.median([e - s for s, e in segs])
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] < max(3, 0.2 * med):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    med = np.median([e - s for s, e in merged])
    return [(s, e) for s, e in merged if e - s >= 0.3 * med] or [(0, ink.shape[0])]


def ink_extent(ink):
    prof = ink.sum(1).astype(np.float32)
    rows = np.where(prof > 0.05 * np.percentile(prof, 95))[0]
    return rows.max() - rows.min() + 1 if rows.size else ink.shape[0]


def image_to_strip(img, height, width):
    g, ink = clean_gray(img)
    cores = find_lines(ink)
    scale = LINE_FILL * height / (ink_extent(ink) / len(cores))
    gap = np.full((height, height // 4), 255, np.uint8)

    parts = []
    for i, (s, e) in enumerate(cores):
        ext = 0.6 * (e - s)
        top = max(s - ext, 0 if i == 0 else (cores[i - 1][1] + s) / 2)
        bot = min(e + ext, g.shape[0] if i == len(cores) - 1 else (e + cores[i + 1][0]) / 2)
        top, bot = int(round(top)), int(round(bot))
        cols = np.where(ink[top:bot].any(0))[0]
        if cols.size == 0:
            continue
        band = Image.fromarray(g[top:bot, max(cols[0] - 4, 0):cols[-1] + 5])
        new_w, new_h = max(1, round(band.width * scale)), max(1, round(band.height * scale))
        band = np.array(band.resize((new_w, new_h), Image.LANCZOS))
        canvas = np.full((height, new_w), 255, np.uint8)
        if new_h <= height:
            canvas[(height - new_h) // 2:(height - new_h) // 2 + new_h] = band
        else:
            canvas[:] = band[(new_h - height) // 2:(new_h - height) // 2 + height]
        parts += [canvas, gap]

    if not parts:
        return np.full((height, width), 255, np.uint8)
    strip = np.hstack(parts[:-1])[:, :4 * width]
    while strip.shape[1] < width:
        strip = np.hstack([strip, gap, strip])
    return strip[:, :4 * width]


def windows_from_strip(strip, width, k):
    starts = np.unique(np.linspace(0, strip.shape[1] - width, k).round().astype(int))
    return np.stack([strip[:, s:s + width] for s in starts])[..., np.newaxis]


def shrink(img, max_side):
    """Opsional (MAX_SIDE > 0): kecilkan foto besar agar cepat. Dengan 0, gambar tidak disentuh sama sekali."""
    if not max_side or max(img.size) <= max_side:
        return img
    img = ImageOps.exif_transpose(img).copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def image_to_windows(data: bytes, height: int, width: int, k: int, max_side: int = 1600) -> np.ndarray:
    img = shrink(Image.open(io.BytesIO(data)), max_side)
    strip = image_to_strip(img, height, width)
    return windows_from_strip(strip, width, k).astype("float32")
