import numpy as np
import cv2

from seg_to_keys import isolate_white, warp_key_lines

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