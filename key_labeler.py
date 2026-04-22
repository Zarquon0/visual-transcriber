"""Piano keyboard detection + labeling pipeline.

This module implements an alternative detection pipeline to the labeler
in ``key_extractor2.py``, tuned for *tight* keyboard crops (where the whole
warped image IS the keyboard region, no surrounding body/floor). The crop
is produced either by:

1. ``find_keyboard_bbox`` in this file (heuristic auto-crop), or
2. ``auto_calibrate.find_corners_auto`` (trapezoidal perspective warp), or
3. ``manual_calibrate`` (4-click manual corners).

The labeler then annotates the warped keys with:

- a red horizontal line at the dynamically-detected black/white boundary
- blue polygons around each detected black key (actual pixel shape, not
  axis-aligned rectangles, so perspective slant and chamfered fronts are
  preserved)
- yellow vertical lines at white-key seams (full-height at E-F / B-C
  gaps where no black key interrupts; below the red line elsewhere)

The black-key detection uses *column-brightness valleys* (dark columns =
black keys) instead of Otsu on the upper band, which was merging adjacent
keys into a single blob on angled photos.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from seg_to_keys import isolate_white, warp_key_lines


def load_image(path: str) -> np.ndarray:
    """Load an image by path. Falls back to PIL for formats OpenCV may not
    ship with (e.g. AVIF on some builds)."""
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image

    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


# ── Bbox-style coarse auto-crop ──────────────────────────────────────────────

def find_keyboard_bbox(
    frame: np.ndarray,
    pad: int = 10,
    debug: bool = False,
    min_aspect: float = 4.0,
):
    """Return padded (x0, y0, x1, y1) of the largest wide-and-short bright
    region (the keyboard).

    Uses ``isolate_white`` + pure-horizontal dilation so vertically-adjacent
    floor/body pixels can't link with the keys. Picks the keyboard-shaped
    (aspect >= min_aspect) contour with the largest area, then unions other
    contours in the same vertical band (hand-split pieces of the keyboard).
    Finally tightens the y-range to the longest contiguous run of
    high-white-pixel-density rows so body/floor rows drop out.

    Robust to hand occlusion (which splits the white blob horizontally but
    leaves multiple same-y-band pieces).
    """
    threshed = isolate_white(frame)
    gray = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)
    H, W = frame.shape[:2]
    kw = max(5, W // 150)
    smeared = cv2.dilate(gray, np.ones((1, kw), np.uint8), iterations=1)
    contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, gray, smeared) if debug else None

    keyboard_like = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 0 and (w / h) >= min_aspect:
            keyboard_like.append((x, y, w, h, c))

    if keyboard_like:
        px, py, pw, ph, _ = max(keyboard_like, key=lambda p: p[2] * p[3])
        xs0, ys0, xs1, ys1 = px, py, px + pw, py + ph
        band_top = py - ph * 0.2
        band_bottom = py + ph * 1.2
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or (w * h) < (W * H) * 0.002:
                continue
            if y >= band_top and (y + h) <= band_bottom:
                xs0 = min(xs0, x); ys0 = min(ys0, y)
                xs1 = max(xs1, x + w); ys1 = max(ys1, y + h)
    else:
        biggest = max(contours, key=cv2.contourArea)
        xs0, ys0, w, h = cv2.boundingRect(biggest)
        xs1, ys1 = xs0 + w, ys0 + h

    # Tighten y-range to longest contiguous run of dense rows.
    roi_mask = gray[ys0:ys1, xs0:xs1]
    if roi_mask.size > 0:
        per_row = (roi_mask > 0).mean(axis=1)
        if per_row.max() > 0:
            dense = per_row >= 0.5 * per_row.max()
            best_start, best_len = 0, 0
            cur_start, cur_len = 0, 0
            for i, d in enumerate(dense):
                if d:
                    if cur_len == 0:
                        cur_start = i
                    cur_len += 1
                    if cur_len > best_len:
                        best_start, best_len = cur_start, cur_len
                else:
                    cur_len = 0
            if best_len > 30:
                prev_ys0 = ys0
                ys0 = prev_ys0 + best_start
                ys1 = prev_ys0 + best_start + best_len

    x0 = max(0, xs0 - pad); y0 = max(0, ys0 - pad)
    x1 = min(W, xs1 + pad); y1 = min(H, ys1 + pad)
    bbox = (x0, y0, x1, y1)
    return (bbox, gray, smeared) if debug else bbox


def warp_to_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Axis-aligned perspective warp: bbox rectangle → output rectangle.

    This is the simple version that doesn't correct perspective on the sides.
    For trapezoidal perspective correction, use ``auto_calibrate.warp_from_corners``
    or ``manual_calibrate.warp_from_corners``.
    """
    x0, y0, x1, y1 = bbox
    top = np.array([x0, y0, x1, y0], dtype=float)
    bottom = np.array([x0, y1, x1, y1], dtype=float)
    return warp_key_lines(frame, top, bottom)


# ── Labeler on a tight warp ──────────────────────────────────────────────────

def draw_labels_tight_crop(warped: np.ndarray) -> np.ndarray:
    """Annotate a tight-keyboard-crop warped image with detected key features.

    Assumes the warped image height IS the keyboard (no body above, no floor
    below). Detects:

    - the black/white horizontal boundary (``y_black_bottom``) via the
      strongest dark→light horizontal edge (Sobel-y)
    - black keys as polygons via column-brightness valleys in the upper band
      (adjacent dark columns = one black key), then per-key Otsu within a
      narrow x-strip to trace the actual pixel contour
    - white-key seams via Sobel-x peaks in the lower band (drawn below the
      red line when a black key sits above, full-height otherwise for the
      E-F / B-C gaps)
    """
    if warped is None or warped.size == 0:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    out = warped.copy()
    h, w = out.shape[:2]
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # --- Dynamic y_black_bottom via strongest dark→light horizontal edge ---
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    row_score = np.clip(sobel_y, 0, None).sum(axis=1)
    search = row_score[: int(0.75 * h)].copy()
    search[: int(0.1 * h)] = 0
    y_black_bottom = int(np.argmax(search))
    if y_black_bottom < int(0.3 * h):
        y_black_bottom = int(0.55 * h)

    # --- Black keys via column-brightness valleys in the upper band ---
    upper = gray[:y_black_bottom, :]
    mid_top = int(0.2 * y_black_bottom)
    mid_bot = int(0.8 * y_black_bottom)
    strip = upper[mid_top:mid_bot, :]
    col_mean = strip.mean(axis=0)
    # Smooth ~1/3 of a black-key width.
    ksz = max(5, int(w / 180))
    if ksz % 2 == 0:
        ksz += 1
    sm = np.convolve(col_mean, np.ones(ksz) / ksz, mode="same")
    med = float(np.median(sm))
    p10 = float(np.percentile(sm, 10))
    dark_thr = p10 + 0.35 * (med - p10)
    dark = sm <= dark_thr

    # Merge small gaps (prevents splitting one key into multiple runs).
    gap_merge = max(2, int(0.005 * w))
    i = 0
    n = len(dark)
    while i < n:
        if not dark[i]:
            j = i
            while j < n and not dark[j]:
                j += 1
            if (j - i) <= gap_merge and i > 0 and j < n:
                dark[i:j] = True
            i = j
        else:
            i += 1

    # For each dark run, trace actual key polygon (supports chamfered shapes).
    expected_bk = w / 60
    min_w = max(4, int(0.5 * expected_bk))
    max_w = max(min_w + 1, int(1.6 * expected_bk))
    row_dark_thr = dark_thr + 0.3 * (med - dark_thr)
    black_rects: list[tuple[int, int, int, int]] = []
    black_polys: list[np.ndarray | None] = []
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    i = 0
    while i < n:
        if dark[i]:
            j = i
            while j < n and dark[j]:
                j += 1
            bw = j - i
            if min_w <= bw <= max_w:
                pad_x = max(2, bw // 3)
                x0_k = max(0, i - pad_x)
                x1_k = min(w, j + pad_x)
                strip_img = upper[:, x0_k:x1_k]
                dark_mask = (strip_img <= row_dark_thr).astype(np.uint8) * 255
                dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, vkernel)
                cc, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                poly = None
                if cc:
                    biggest = max(cc, key=cv2.contourArea)
                    if cv2.contourArea(biggest) >= 0.25 * bw * y_black_bottom:
                        eps = max(0.5, 0.002 * cv2.arcLength(biggest, True))
                        approx = cv2.approxPolyDP(biggest, eps, True)
                        poly = approx + np.array([[[x0_k, 0]]], dtype=np.int32)
                black_polys.append(poly)
                black_rects.append((i, 0, bw, y_black_bottom))
            i = j
        else:
            i += 1

    black_contours: list[np.ndarray] = []
    for poly, rect in zip(black_polys, black_rects):
        if poly is not None:
            black_contours.append(poly)
        else:
            x, y, bw, bh = rect
            black_contours.append(np.array(
                [[[x, y]], [[x + bw, y]], [[x + bw, y + bh]], [[x, y + bh]]],
                dtype=np.int32,
            ))

    # --- White-key seams via Sobel-x in the pure-white lower band ---
    white_band_top = y_black_bottom + int(0.1 * (h - y_black_bottom))
    band = gray[white_band_top:h - 1, :]
    sobel_x = np.abs(cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3))
    col_strength = sobel_x.mean(axis=0)
    sig = np.convolve(col_strength, np.ones(9) / 9, mode="same")
    med2 = float(np.median(sig))
    p90 = float(np.percentile(sig, 90))
    thr = med2 + 0.2 * max(1.0, p90 - med2)
    peaks: list[int] = []
    for i in range(2, len(sig) - 2):
        c = sig[i]
        if c >= thr and c >= sig[i - 1] and c >= sig[i + 1]:
            peaks.append(i)
    min_sep = max(4, int(w / 72))
    merged: list[int] = []
    for p in peaks:
        if merged and p - merged[-1] < min_sep:
            continue
        merged.append(p)

    black_xs = [(x - 2, x + bw2 + 2) for (x, _, bw2, _) in black_rects]

    def black_key_above(px: int) -> bool:
        return any(lo <= px <= hi for lo, hi in black_xs)

    seam_full: list[int] = []
    seam_white_only: list[int] = []
    for bx in merged:
        (seam_full if not black_key_above(bx) else seam_white_only).append(bx)

    # --- Draw ---
    cv2.line(out, (0, y_black_bottom), (w - 1, y_black_bottom), (0, 0, 255), 2)
    for bx in seam_white_only:
        cv2.line(out, (bx, y_black_bottom + 2), (bx, h - 1), (0, 255, 255), 2)
    for bx in seam_full:
        cv2.line(out, (bx, 0), (bx, h - 1), (0, 255, 255), 2)
    if black_contours:
        cv2.drawContours(out, black_contours, -1, (255, 0, 0), 2)
    return out


# ── Demo ─────────────────────────────────────────────────────────────────────

def _demo_image(path: str) -> None:
    frame = load_image(path)
    bbox = find_keyboard_bbox(frame)
    if bbox is None:
        print("could not locate keyboard")
        return
    warped = warp_to_bbox(frame, bbox)
    labeled = draw_labels_tight_crop(warped)
    out_path = Path(path).with_suffix("").as_posix() + "_labeled.png"
    cv2.imwrite(out_path, labeled)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: uv run python key_labeler.py path/to/photo.jpg")
        raise SystemExit(2)
    _demo_image(sys.argv[1])
