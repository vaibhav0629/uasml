"""Preprocessing inferensi. Port 1:1 dari notebook UAS_Handwriting_Classification.ipynb
(bagian 2.8-2.10: flatten_background, remove_rules, clean_gray, find_lines, image_to_strip, windows_from_strip).

Satu-satunya perbedaan: operasi close elips pada flatten_background memakai close_ellipse_exact,
yang hasilnya IDENTIK bit-per-bit dengan cv2.morphologyEx tetapi jauh lebih cepat pada foto besar.
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu

MAX_STRIP_WINDOWS = 4
LINE_FILL = 0.62
LINE_GAP_RATIO = 0.14


def close_ellipse_exact(g, k):
    """Hasil identik dengan cv2.morphologyEx(g, MORPH_CLOSE, ellipse(k,k)), tapi jauh lebih cepat.
    Elips dipecah per baris: dilasi = max atas dy dari (dilasi horizontal selebar w(dy)) yang digeser dy.
    Dilasi horizontal dibangun bertahap (w -> w+1) dari yang sempit ke lebar."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    r = k // 2
    # setengah lebar tiap baris, dibaca langsung dari SE OpenCV agar pasti sama
    hw = np.array([ (np.where(row)[0].max() - r) if row.any() else -1 for row in se ])
    for i, row in enumerate(se):  # pastikan tiap baris simetris & kontigu
        c = np.where(row)[0]
        if c.size: assert c.min() == r - hw[i] and c.size == 2 * hw[i] + 1
    return _erode(_dilate(g, hw, r), hw, r)

def _pass(g, hw, r, op):
    h, w = g.shape
    out = None
    cur = g.copy()            # cur = filter horizontal selebar wcur
    wcur = 0
    order = sorted(set(int(x) for x in hw if x >= 0))
    for wv in order:
        while wcur < wv:      # perlebar 1 piksel tiap sisi
            nxt = cur.copy()
            op(nxt[:, 1:], cur[:, :-1], out=nxt[:, 1:])
            op(nxt[:, :-1], cur[:, 1:], out=nxt[:, :-1])
            cur = nxt; wcur += 1
        for i in np.where(hw == wv)[0]:
            dy = int(i) - r   # baris SE ke-i menggeser sebesar dy
            if abs(dy) >= h: continue
            if out is None: out = None
            # out(y) = op(out(y), cur(y+dy))
            if dy >= 0:
                dst, src = slice(0, h - dy), slice(dy, h)
            else:
                dst, src = slice(-dy, h), slice(0, h + dy)
            if out is None:
                out = np.full_like(g, 0 if op is np.maximum else 255)
            op(out[dst], cur[src], out=out[dst])
    return out

def _dilate(g, hw, r): return _pass(g, hw, r, np.maximum)
def _erode(g, hw, r):  return _pass(g, hw, r, np.minimum)


def flatten_background(gray):
    gray = np.asarray(gray, dtype=np.uint8)
    size = max(15, (min(gray.shape) // 6) | 1)
    background = close_ellipse_exact(gray, size)  # identik dengan cv2.morphologyEx(MORPH_CLOSE, ellipse), jauh lebih cepat
    background = cv2.GaussianBlur(background, (0, 0), size / 3)
    corrected = np.clip(gray.astype(np.float32) / np.maximum(background, 1) * 255, 0, 255)
    low, high = np.percentile(corrected, [1, 50])
    return (np.clip((corrected - low) / max(high - low, 1), 0, 1) * 255).astype(np.uint8)


def remove_rules(ink, min_frac=0.5, max_thin=0.06, ratio=2.5):
    ink = np.asarray(ink, dtype=bool)
    height, width = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    rules = np.zeros_like(ink)
    if count <= 1:
        return ink, rules

    components = stats[1:]
    large = components[components[:, 4] >= max(20, int(0.00005 * height * width))]
    median_h = float(np.median(large[:, 3])) if len(large) else 1.0
    median_w = float(np.median(large[:, 2])) if len(large) else 1.0
    comp_w, comp_h, area = components[:, 2], components[:, 3], components[:, 4]
    fill = area / np.maximum(1, comp_w * comp_h)

    vertical = (comp_h >= min_frac * height) & (comp_w <= max_thin * width) & (fill >= 0.5) & (comp_h >= ratio * median_h)
    horizontal = (comp_w >= min_frac * width) & (comp_h <= max_thin * height) & (fill >= 0.5) & (comp_w >= ratio * median_w)
    rule_ids = np.where(vertical | horizontal)[0] + 1
    if rule_ids.size:
        rules = np.isin(labels, rule_ids)
    return ink & ~rules, rules


# Normalisasi latar dan segmentasi tinta.
def _adaptive_ink_mask(gray):
    short_side = min(gray.shape)
    block = min(51, max(15, (short_side // 8) | 1))
    if block >= short_side:
        block = max(3, (short_side - 1) | 1)
    block |= 1
    local = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 7
    ) > 0
    return local & (gray < np.percentile(gray, 45))


def clean_gray(img, return_rules=False):
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)

    gray = flatten_background(np.asarray(img.convert("L"), dtype=np.uint8))
    if np.ptp(gray) == 0:
        empty = np.zeros(gray.shape, dtype=bool)
        return (gray, empty, empty) if return_rules else (gray, empty)

    ink = gray < threshold_otsu(gray)
    if ink.sum() < max(3, int(gray.size * 0.00005)) and np.std(gray) > 2:
        ink = _adaptive_ink_mask(gray)

    ink, rules = remove_rules(ink)
    if rules.any():
        expanded = cv2.dilate(rules.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        gray = gray.copy()
        gray[expanded] = 255
        ink &= ~expanded
    return (gray, ink, rules) if return_rules else (gray, ink)


def _row_segments(active):
    edges = np.diff(np.r_[0, np.asarray(active, dtype=np.int8), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def find_lines(ink):
    ink = np.asarray(ink, dtype=bool)
    if not ink.size or not ink.any():
        return []

    profile = cv2.GaussianBlur(ink.sum(axis=1, dtype=np.float32).reshape(-1, 1), (1, 5), 0).ravel()
    nonzero = profile[profile > 0]
    if not nonzero.size:
        return []

    threshold = max(1.0, 0.04 * float(np.percentile(nonzero, 95)), 0.0015 * ink.shape[1])
    segments = _row_segments(profile >= threshold)
    if not segments:
        rows = np.where(ink.any(axis=1))[0]
        return [(int(rows.min()), int(rows.max()) + 1)]

    gap = max(2, int(round(0.32 * max(1.0, float(np.median([end - start for start, end in segments]))))))
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    if len(merged) > 20:
        rows = np.where(ink.any(axis=1))[0]
        return [(int(rows.min()), int(rows.max()) + 1)]
    return [(int(start), int(end)) for start, end in merged]


def ink_extent(ink):
    rows = np.where(np.asarray(ink, dtype=bool).any(axis=1))[0]
    return int(rows.max() - rows.min() + 1) if rows.size else ink.shape[0]


def _crop_line_bands(gray, ink, cores):
    line_data, ink_heights = [], []
    for index, (start, end) in enumerate(cores):
        top = 0 if index == 0 else (cores[index - 1][1] + start) // 2
        bottom = gray.shape[0] if index == len(cores) - 1 else (end + cores[index + 1][0]) // 2
        band_ink = ink[top:bottom]
        ys, xs = np.where(band_ink)
        if not xs.size:
            continue

        ink_h, ink_w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
        pad_y, pad_x = max(2, round(0.10 * ink_h)), max(4, round(0.025 * ink_w))
        y0, y1 = top + max(0, ys.min() - pad_y), top + min(band_ink.shape[0], ys.max() + 1 + pad_y)
        x0, x1 = max(0, xs.min() - pad_x), min(gray.shape[1], xs.max() + 1 + pad_x)
        crop, crop_ink = gray[y0:y1, x0:x1], ink[y0:y1, x0:x1]
        if crop.size and crop_ink.any():
            line_data.append((crop, crop_ink))
            ink_heights.append(ink_h)
    return line_data, ink_heights


def _compute_strip_scale(line_data, ink_heights, height, max_width, gap_width):
    scale = LINE_FILL * height / max(1.0, float(np.median(ink_heights)))
    scale = min(scale, 0.82 * height / max(1.0, max(ink_heights)))
    width_budget = max(1, max_width - gap_width * max(0, len(line_data) - 1))
    source_width = sum(crop.shape[1] for crop, _ in line_data)
    if source_width:
        scale = min(scale, width_budget / source_width)
    return max(scale, 0.05)


def _render_centered_line(crop, crop_ink, scale, height):
    new_w = max(1, round(crop.shape[1] * scale))
    new_h = max(1, round(crop.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    ink = cv2.resize(crop_ink.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
    rows = np.where(ink.any(axis=1))[0]
    if not rows.size:
        return None

    center = (float(rows.min()) + float(rows.max())) / 2
    offset = round(height / 2 - center)
    canvas = np.full((height, new_w), 255, dtype=np.uint8)
    src_top, dst_top = max(0, -offset), max(0, offset)
    copy_h = min(new_h - src_top, height - dst_top)
    if copy_h > 0:
        canvas[dst_top:dst_top + copy_h] = gray[src_top:src_top + copy_h]
    return canvas


# Konversi halaman menjadi strip horizontal.
def image_to_strip(img, height, width):
    height, width = int(height), int(width)
    max_width = MAX_STRIP_WINDOWS * width
    blank = np.full((height, width), 255, dtype=np.uint8)

    gray, ink = clean_gray(img)
    cores = find_lines(ink) if ink.any() else []
    line_data, ink_heights = _crop_line_bands(gray, ink, cores)
    if not line_data:
        return blank

    gap = max(8, round(height * LINE_GAP_RATIO))
    scale = _compute_strip_scale(line_data, ink_heights, height, max_width, gap)
    lines = [
        rendered for crop, crop_ink in line_data
        if (rendered := _render_centered_line(crop, crop_ink, scale, height)) is not None
    ]
    if not lines:
        return blank

    total_width = sum(line.shape[1] for line in lines) + gap * max(0, len(lines) - 1)
    strip = np.full((height, min(max(width, total_width), max_width)), 255, dtype=np.uint8)
    x = 0
    for index, line in enumerate(lines):
        if x >= strip.shape[1]:
            break
        copy_width = min(line.shape[1], strip.shape[1] - x)
        strip[:, x:x + copy_width] = line[:, :copy_width]
        x += copy_width + (gap if index < len(lines) - 1 else 0)
    return strip


def windows_from_strip(strip, width, k):
    strip = np.asarray(strip, dtype=np.uint8)
    if strip.shape[1] < width:
        padded = np.full((strip.shape[0], width), 255, dtype=np.uint8)
        padded[:, :strip.shape[1]] = strip
        strip = padded

    max_start = max(0, strip.shape[1] - width)
    starts = np.unique(np.linspace(0, max_start, k).round().astype(int))
    return np.stack([strip[:, start:start + width] for start in starts])[..., np.newaxis]


def shrink(img, max_side):
    """Opsional (MAX_SIDE > 0): kecilkan foto besar. Dengan 0, gambar tidak disentuh (sama dengan notebook)."""
    if not max_side or max(img.size) <= max_side:
        return img
    img = ImageOps.exif_transpose(img).copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img
