"""Keyboard corner detection and perspective warp helpers.

This file replaces the useful parts of the old archived auto_calibrate.py
without depending on that archived file.

Public API:
    auto_warp_to_key_tops(frame) -> (warped, corners) or (None, None)
    warp_from_corners(frame, corners, out_height=220) -> warped
"""

from __future__ import annotations

import cv2
import numpy as np

from .seg_to_keys import isolate_white


BLACK_TO_WHITE_KEY_HEIGHT_RATIO = 0.70


def _as_gray_white_mask(frame: np.ndarray) -> np.ndarray:
    """Return single-channel white-key mask."""
    mask = isolate_white(frame)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _fit_line_y_of_x(
    points: np.ndarray,
    *,
    iters: int = 250,
    tol: float = 4.0,
    seed: int = 42,
) -> tuple[float, float] | None:
    """RANSAC fit y = m*x + b."""
    if len(points) < 20:
        return None

    rng = np.random.default_rng(seed)
    xs = points[:, 0].astype(float)
    ys = points[:, 1].astype(float)

    best_mask = None
    best_count = 0

    for _ in range(iters):
        i, j = rng.integers(0, len(points), size=2)
        if i == j or xs[i] == xs[j]:
            continue

        m = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - m * xs[i]

        dist = np.abs(ys - (m * xs + b))
        mask = dist <= tol
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < 20:
        return None

    xin = xs[best_mask]
    yin = ys[best_mask]
    A = np.vstack([xin, np.ones_like(xin)]).T
    m, b = np.linalg.lstsq(A, yin, rcond=None)[0]
    return float(m), float(b)


def _fit_line_x_of_y(
    points: np.ndarray,
    *,
    iters: int = 250,
    tol: float = 6.0,
    seed: int = 7,
) -> tuple[float, float] | None:
    """RANSAC fit x = m*y + b."""
    if len(points) < 20:
        return None

    rng = np.random.default_rng(seed)
    xs = points[:, 0].astype(float)
    ys = points[:, 1].astype(float)

    best_mask = None
    best_count = 0

    for _ in range(iters):
        i, j = rng.integers(0, len(points), size=2)
        if i == j or ys[i] == ys[j]:
            continue

        m = (xs[j] - xs[i]) / (ys[j] - ys[i])
        b = xs[i] - m * ys[i]

        dist = np.abs(xs - (m * ys + b))
        mask = dist <= tol
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < 20:
        return None

    yin = ys[best_mask]
    xin = xs[best_mask]
    A = np.vstack([yin, np.ones_like(yin)]).T
    m, b = np.linalg.lstsq(A, xin, rcond=None)[0]
    return float(m), float(b)


def _intersect_yx(
    horizontal: tuple[float, float],
    vertical: tuple[float, float],
) -> np.ndarray | None:
    """Intersect y = mh*x + bh with x = mv*y + bv."""
    mh, bh = horizontal
    mv, bv = vertical

    denom = 1.0 - mv * mh
    if abs(denom) < 1e-9:
        return None

    x = (mv * bh + bv) / denom
    y = mh * x + bh
    return np.array([x, y], dtype=np.float32)


def _column_extrema(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return topmost and bottommost white pixel per column, or -1."""
    h, _ = mask.shape
    present = mask > 0

    top = np.where(present.any(axis=0), present.argmax(axis=0), -1)

    flipped = present[::-1]
    bottom = np.where(present.any(axis=0), h - 1 - flipped.argmax(axis=0), -1)

    return top, bottom

def find_keyboard_corners(frame: np.ndarray) -> np.ndarray | None:
    """Detect keyboard key-surface corners as TL, TR, BR, BL.

    More forgiving version:
    - tries isolate_white()
    - if that is weak, tries LAB brightness masking directly
    - relaxes blob area/aspect thresholds
    - chooses the best wide component by area * aspect
    """
    H, W = frame.shape[:2]

    mask = _as_gray_white_mask(frame)

    # If isolate_white is too weak, fall back to a direct LAB-brightness mask.
    white_ratio = float((mask > 0).mean())
    if white_ratio < 0.01:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        # Brightest ~18% of pixels, but not below a reasonable brightness.
        cutoff = max(120, int(np.percentile(l, 82)))
        mask = np.where(l >= cutoff, 255, 0).astype(np.uint8)

    # Clean noise and merge adjacent white keys.
    mask = cv2.medianBlur(mask, 5)

    kw = max(9, W // 40)
    smeared = cv2.dilate(mask, np.ones((1, kw), np.uint8), iterations=2)
    smeared = cv2.morphologyEx(
        smeared,
        cv2.MORPH_CLOSE,
        np.ones((5, max(9, W // 80)), np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        smeared,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        cv2.imwrite("debug_warp_mask.png", mask)
        cv2.imwrite("debug_warp_smeared.png", smeared)
        print("[warp_calibration] no contours; wrote debug_warp_mask.png/debug_warp_smeared.png")
        return None

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h <= 0 or w <= 0:
            continue

        area = cv2.contourArea(c)
        aspect = w / max(1, h)
        frac = area / max(1, W * H)

        # Relaxed candidate test.
        if aspect >= 2.2 and frac >= 0.002:
            score = area * min(aspect, 20.0)
            candidates.append((score, area, aspect, c, (x, y, w, h)))

    if not candidates:
        cv2.imwrite("debug_warp_mask.png", mask)
        cv2.imwrite("debug_warp_smeared.png", smeared)
        dbg = frame.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                dbg,
                f"a={cv2.contourArea(c):.0f} asp={w/max(1,h):.1f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite("debug_warp_candidates.jpg", dbg)
        print("[warp_calibration] no valid candidates; wrote debug_warp_candidates.jpg")
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, _, _, contour, (cx, cy, cw, ch) = candidates[0]

    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)

    top_y, bot_y = _column_extrema(blob)
    xs = np.arange(W)

    top_pts = np.stack([xs[top_y >= 0], top_y[top_y >= 0]], axis=1)
    bot_pts = np.stack([xs[bot_y >= 0], bot_y[bot_y >= 0]], axis=1)

    top_line = _fit_line_y_of_x(top_pts, tol=7.0, seed=42)
    bot_line = _fit_line_y_of_x(bot_pts, tol=7.0, seed=43)

    if top_line is None or bot_line is None:
        cv2.imwrite("debug_warp_blob.png", blob)
        print("[warp_calibration] failed top/bottom rail fit; wrote debug_warp_blob.png")
        return None

    m_top, b_top = top_line
    m_bot, b_bot = bot_line

    x_mid = cx + cw // 2
    y_top_mid = int(m_top * x_mid + b_top)
    y_bot_mid = int(m_bot * x_mid + b_bot)

    pure_top = y_top_mid + int(0.50 * (y_bot_mid - y_top_mid))
    pure_bot = y_bot_mid

    y0 = max(0, min(H - 1, pure_top))
    y1 = max(0, min(H - 1, pure_bot))

    if y1 <= y0 + 5:
        cv2.imwrite("debug_warp_blob.png", blob)
        print("[warp_calibration] pure-white band too thin; wrote debug_warp_blob.png")
        return None

    blob_dilated = cv2.dilate(blob, np.ones((1, max(7, W // 80)), np.uint8))
    combined = cv2.bitwise_and(mask, blob_dilated)

    sub = combined[y0 : y1 + 1] > 0
    row_has = sub.any(axis=1)

    if row_has.sum() < 10:
        cv2.imwrite("debug_warp_combined.png", combined)
        print("[warp_calibration] not enough rows for side rails; wrote debug_warp_combined.png")
        return None

    leftmost = np.where(row_has, sub.argmax(axis=1), -1)
    flipped = sub[:, ::-1]
    rightmost = np.where(row_has, W - 1 - flipped.argmax(axis=1), -1)

    row_ys = np.arange(sub.shape[0]) + y0

    keep = (
        row_has
        & (leftmost >= cx - 60)
        & (rightmost <= cx + cw + 60)
        & (rightmost > leftmost)
    )

    left_pts = np.stack([leftmost[keep], row_ys[keep]], axis=1)
    right_pts = np.stack([rightmost[keep], row_ys[keep]], axis=1)

    if len(left_pts) < 10 or len(right_pts) < 10:
        cv2.imwrite("debug_warp_combined.png", combined)
        print("[warp_calibration] too few side points; wrote debug_warp_combined.png")
        return None

    if len(left_pts):
        left_thresh = np.percentile(left_pts[:, 0], 50)
        left_pts = left_pts[left_pts[:, 0] <= left_thresh]

    if len(right_pts):
        right_thresh = np.percentile(right_pts[:, 0], 50)
        right_pts = right_pts[right_pts[:, 0] >= right_thresh]

    left_line = _fit_line_x_of_y(left_pts, tol=12.0, seed=7)
    right_line = _fit_line_x_of_y(right_pts, tol=12.0, seed=8)

    if left_line is None or right_line is None:
        cv2.imwrite("debug_warp_combined.png", combined)
        print("[warp_calibration] failed side rail fit; wrote debug_warp_combined.png")
        return None

    tl = _intersect_yx(top_line, left_line)
    tr = _intersect_yx(top_line, right_line)
    br = _intersect_yx(bot_line, right_line)
    bl = _intersect_yx(bot_line, left_line)

    if any(p is None for p in (tl, tr, br, bl)):
        return None

    corners = np.stack([tl, tr, br, bl]).astype(np.float32)

    if not np.isfinite(corners).all():
        return None

    area = cv2.contourArea(corners.reshape(-1, 1, 2))
    if area < 0.001 * W * H:
        print(f"[warp_calibration] rejected tiny corner area: {area:.1f}")
        return None

    # Debug success image.
    dbg = frame.copy()
    pts = corners.astype(int)
    cv2.polylines(dbg, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    for (x, y), label in zip(pts, ["TL", "TR", "BR", "BL"]):
        cv2.circle(dbg, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            dbg,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite("debug_warp_success.jpg", dbg)

    return corners


def warp_from_corners(
    frame: np.ndarray,
    corners: np.ndarray,
    *,
    out_height: int = 220,
) -> np.ndarray:
    """Perspective-warp TL, TR, BR, BL corners into a rectangular keybed."""
    tl, tr, br, bl = corners.astype(np.float32)

    top_len = float(np.linalg.norm(tr - tl))
    bot_len = float(np.linalg.norm(br - bl))
    left_len = float(np.linalg.norm(bl - tl))
    right_len = float(np.linalg.norm(br - tr))

    avg_w = 0.5 * (top_len + bot_len)
    avg_h = max(1.0, 0.5 * (left_len + right_len))

    out_w = max(120, int(round(avg_w * out_height / avg_h)))

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_height - 1],
            [0, out_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(frame, M, (out_w, out_height))


def tighten_corners_to_key_tops(
    frame: np.ndarray,
    corners: np.ndarray,
    *,
    ratio: float = BLACK_TO_WHITE_KEY_HEIGHT_RATIO,
) -> np.ndarray:
    """Move the bottom edge upward so the warp focuses on key tops.

    If the estimate looks unsafe, returns the original corners.
    """
    try:
        preview = warp_from_corners(frame, corners, out_height=220)
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Strong dark-to-light edge: bottom of black keys.
        dark_to_light = np.clip(sobel_y, 0, None).sum(axis=1)
        search = dark_to_light[: int(0.75 * h)].copy()
        search[: int(0.10 * h)] = 0
        y_black_bottom = int(np.argmax(search))

        # Strong light-to-dark edge above: top of black keys.
        light_to_dark = np.clip(-sobel_y, 0, None).sum(axis=1)
        top_end = min(y_black_bottom - 5, int(0.45 * h))

        if top_end > 5:
            top_search = light_to_dark[:top_end]
            peak = float(top_search.max())
            med = float(np.median(top_search))
            y_black_top = int(np.argmax(top_search)) if peak > 3.0 * max(1.0, med) else 0
        else:
            y_black_top = 0

        black_h = y_black_bottom - y_black_top
        if black_h <= 8:
            return corners

        expected_bottom = int(y_black_top + black_h / ratio)

        # Only tighten if it lands safely below black keys and above old bottom.
        if not (y_black_bottom + 5 < expected_bottom < h - 3):
            return corners

        tl, tr, br, bl = corners.astype(np.float32)

        src = corners.astype(np.float32)
        dst = np.array(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = np.linalg.inv(M)

        pts = np.array(
            [
                [[0, expected_bottom]],
                [[w - 1, expected_bottom]],
            ],
            dtype=np.float32,
        )

        projected = cv2.perspectiveTransform(pts, M_inv).reshape(2, 2)
        new_bl = projected[0]
        new_br = projected[1]

        return np.stack([tl, tr, new_br, new_bl]).astype(np.float32)

    except Exception:
        return corners


def auto_warp_to_key_tops(
    frame: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect keyboard corners and return a tight key-top warp.

    Returns:
        warped, tight_corners

    On failure:
        None, None
    """
    corners = find_keyboard_corners(frame)
    if corners is None:
        return None, None

    tight = tighten_corners_to_key_tops(frame, corners)
    warped = warp_from_corners(frame, tight)

    if warped is None or warped.size == 0:
        return None, None

    return warped, tight