"""Auto-detect keyboard 4 corners from the white-key blob geometry.

Pipeline:

1. ``isolate_white`` + horizontal dilation merges the white keys into a
   connected blob. The keyboard-aspect contour with the largest area is the
   seed.
2. Per-column topmost / bottommost blob pixels are RANSAC-fit to give the
   TOP and BOTTOM rail lines (slanted, not horizontal-only).
3. For thin blobs (angled photos where the blob only captured the pure-
   white region below the blacks), the top rail walks upward through the
   black-key band — rows with sparse whites (key tops between blacks)
   count as keyboard; rows with ~none count as keyboard body and stop the
   walk.
4. Per-row leftmost/rightmost blob pixels in the pure-white lower region
   are RANSAC-fit to give the LEFT and RIGHT rail lines. A pre-filter
   drops shadow-affected rows so the real keyboard edge (not the shadow
   edge) is captured.
5. The 4 rails pairwise-intersect to give the TL/TR/BR/BL corners. These
   are passed to ``warp_from_corners`` for a perspective-correct warp to
   a rectangle, then to ``key_labeler.draw_labels_tight_crop`` for
   labeling.

Run on a list of photos to generate a side-by-side comparison grid:

    uv run python auto_calibrate.py path1.jpg path2.jpg ...

Falls back sensibly but is not perfect; for tricky shots use
``manual_calibrate.py`` instead.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from key_labeler import load_image, draw_labels_tight_crop
from seg_to_keys import isolate_white


# Physical piano geometry: black keys are ~0.60 (acoustic) to ~0.70
# (MIDI controllers) as tall as white keys. Used to infer where key tops
# end from measured black-key height.
BLACK_TO_WHITE_KEY_HEIGHT_RATIO = 0.70


def _column_extrema(mask: np.ndarray):
    """Per-column topmost and bottommost white-pixel y, or -1 if empty."""
    h, w = mask.shape
    ys = np.arange(h)[:, None]
    present = mask > 0
    # top y: first True per column; bottom y: last True per column
    top_idx = np.where(present.any(axis=0), present.argmax(axis=0), -1)
    flipped = present[::-1]
    bot_idx = np.where(present.any(axis=0), h - 1 - flipped.argmax(axis=0), -1)
    return top_idx, bot_idx


def _ransac_line(points: np.ndarray, iters: int = 200, inlier_tol: float = 3.0, rng=None):
    """Fit a line y = m*x + b to 2D points via RANSAC. Returns (m, b, inliers_mask)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(points)
    if n < 2:
        return None
    best_inliers = None
    best_count = 0
    xs = points[:, 0]
    ys = points[:, 1]
    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        xi, yi = points[i]
        xj, yj = points[j]
        if xj == xi:
            continue
        m = (yj - yi) / (xj - xi)
        b = yi - m * xi
        dist = np.abs(ys - (m * xs + b))
        inliers = dist <= inlier_tol
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 2:
        return None
    # Least-squares refit on inliers.
    xin = xs[best_inliers]
    yin = ys[best_inliers]
    A = np.vstack([xin, np.ones_like(xin)]).T
    m, b = np.linalg.lstsq(A, yin, rcond=None)[0]
    return float(m), float(b), best_inliers


def find_corners_auto(frame: np.ndarray, debug: bool = False):
    """Return 4 corners (TL, TR, BR, BL) or None if detection fails."""
    H, W = frame.shape[:2]
    mask = cv2.cvtColor(isolate_white(frame), cv2.COLOR_BGR2GRAY)
    # Horizontal dilate to merge adjacent keys — use a wider kernel so
    # shadows/partial-occlusions don't split the keyboard into fragments.
    kw = max(5, W // 50)
    smeared = cv2.dilate(mask, np.ones((1, kw), np.uint8), iterations=1)

    # Keep only the largest wide-aspect connected component (reject floor/body).
    contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        area = cv2.contourArea(c)
        if (w / h) >= 3.0 and (best is None or area > best[0]):
            best = (area, c, (x, y, w, h))
    if best is None:
        return None if not debug else (None, smeared, None, None)

    _, contour, (cx, cy, cw, ch) = best
    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)

    top_y, bot_y = _column_extrema(blob)
    xs = np.arange(W)
    top_pts = np.stack([xs[top_y >= 0], top_y[top_y >= 0]], axis=1)
    bot_pts = np.stack([xs[bot_y >= 0], bot_y[bot_y >= 0]], axis=1)

    if len(top_pts) < 20 or len(bot_pts) < 20:
        return None if not debug else (None, smeared, None, None)

    top_fit = _ransac_line(top_pts)
    bot_fit = _ransac_line(bot_pts)
    if top_fit is None or bot_fit is None:
        return None if not debug else (None, smeared, None, None)

    m_top, b_top, _ = top_fit
    m_bot, b_bot, _ = bot_fit

    # Extend the top rail upward through the black-key band: those rows still
    # have sparse white pixels (key tops between blacks). Stop when we hit
    # rows with ~no white pixels (keyboard body).
    col_range = slice(max(0, cx - 2), min(W, cx + cw + 2))
    white_per_row = (mask[:, col_range] > 0).sum(axis=1)
    # Black-key-band rows have SPARSE whites (narrow strips between blacks),
    # typical counts ~15–150. Keyboard body rows have ~0. Use a small
    # absolute threshold scaled slightly by width.
    min_white = max(5.0, 0.001 * W)
    # Walk up from the current top rail's midpoint y.
    x_mid = cx + cw // 2
    y_top_mid = int(m_top * x_mid + b_top)
    y_bot_mid_est = int(m_bot * x_mid + b_bot)
    blob_h = max(1, y_bot_mid_est - y_top_mid)
    # Only extend if the blob is VERY thin (aspect > 12). A thick blob
    # means the blacks are already included (top-down or near-top-down view),
    # so extending further just walks into the keyboard body.
    if cw / blob_h > 12:
        # Cap extension at 1.3x the current (short) blob height.
        max_extension = int(1.3 * blob_h)
        y_limit = max(0, y_top_mid - max_extension)
        new_top_mid = y_top_mid
        consecutive_empty = 0
        for y in range(y_top_mid - 1, y_limit - 1, -1):
            if white_per_row[y] >= min_white:
                new_top_mid = y
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                if consecutive_empty >= 15:
                    break
        b_top = new_top_mid - m_top * x_mid

    # Fit LEFT and RIGHT rails using only the PURE-WHITE lower region,
    # where whites are continuous (no black-key interruptions). This gives
    # reliable leftmost/rightmost samples.
    x_mid_row = cx + cw // 2
    y_bot_mid = int(m_bot * x_mid_row + b_bot)
    y_top_extended_mid = int(m_top * x_mid_row + b_top)
    # "Pure white" starts ~60% of the way down from the extended top to bot.
    pure_white_top = y_top_extended_mid + int(0.6 * (y_bot_mid - y_top_extended_mid))
    pure_white_bot = y_bot_mid
    y_top_min = max(0, pure_white_top)
    y_bot_max = min(H - 1, pure_white_bot)
    sub_mask = mask[y_top_min:y_bot_max + 1]
    sub_h = sub_mask.shape[0]
    row_any = sub_mask > 0
    has_any = row_any.any(axis=1)
    leftmost = np.where(has_any, row_any.argmax(axis=1), -1)
    flipped = row_any[:, ::-1]
    rightmost = np.where(has_any, W - 1 - flipped.argmax(axis=1), -1)
    row_ys = np.arange(sub_h) + y_top_min
    keep_rows = has_any & (leftmost >= cx - 20) & (rightmost <= cx + cw + 20)
    left_pts = np.stack([leftmost[keep_rows], row_ys[keep_rows]], axis=1)
    right_pts = np.stack([rightmost[keep_rows], row_ys[keep_rows]], axis=1)

    # Fit left/right rails in (y, x) coordinates since they're near-vertical
    # (slope in y -> x). We'll store them as x = m_side*y + b_side.
    def ransac_xy_as_yx(points, iters=200, tol=3.0, rng=None):
        if rng is None:
            rng = np.random.default_rng(7)
        n = len(points)
        if n < 20:
            return None
        best_count = 0
        best_inliers = None
        xs_arr = points[:, 0].astype(float)
        ys_arr = points[:, 1].astype(float)
        for _ in range(iters):
            i, j = rng.integers(0, n, size=2)
            if i == j or ys_arr[j] == ys_arr[i]:
                continue
            m = (xs_arr[j] - xs_arr[i]) / (ys_arr[j] - ys_arr[i])
            b = xs_arr[i] - m * ys_arr[i]
            d = np.abs(xs_arr - (m * ys_arr + b))
            in_ = d <= tol
            c = int(in_.sum())
            if c > best_count:
                best_count, best_inliers = c, in_
        if best_inliers is None or best_count < 10:
            return None
        yin = ys_arr[best_inliers]; xin = xs_arr[best_inliers]
        A = np.vstack([yin, np.ones_like(yin)]).T
        m, b = np.linalg.lstsq(A, xin, rcond=None)[0]
        return float(m), float(b)

    # Fit left/right rails in (y, x) coordinates since they're near-vertical.
    def ransac_xy_as_yx(points, iters=400, tol=10.0, rng=None):
        if rng is None:
            rng = np.random.default_rng(7)
        n = len(points)
        if n < 20:
            return None
        best_count = 0
        best_inliers = None
        xs_arr = points[:, 0].astype(float)
        ys_arr = points[:, 1].astype(float)
        for _ in range(iters):
            i, j = rng.integers(0, n, size=2)
            if i == j or ys_arr[j] == ys_arr[i]:
                continue
            m = (xs_arr[j] - xs_arr[i]) / (ys_arr[j] - ys_arr[i])
            b = xs_arr[i] - m * ys_arr[i]
            d = np.abs(xs_arr - (m * ys_arr + b))
            in_ = d <= tol
            c = int(in_.sum())
            if c > best_count:
                best_count, best_inliers = c, in_
        if best_inliers is None or best_count < 10:
            return None
        yin = ys_arr[best_inliers]; xin = xs_arr[best_inliers]
        A = np.vstack([yin, np.ones_like(yin)]).T
        m, b = np.linalg.lstsq(A, xin, rcond=None)[0]
        return float(m), float(b)

    # Pre-filter: keep only rows whose rightmost is in the RIGHTMOST 40%
    # (by x) — shadow-edge rows with too-small rightmost are dropped.
    # Same for leftmost but on the left side.
    if len(right_pts) > 0:
        r_thr = np.percentile(right_pts[:, 0], 60)
        right_pts = right_pts[right_pts[:, 0] >= r_thr]
    if len(left_pts) > 0:
        l_thr = np.percentile(left_pts[:, 0], 40)
        left_pts = left_pts[left_pts[:, 0] <= l_thr]

    left_fit = ransac_xy_as_yx(left_pts, tol=10.0)
    right_fit = ransac_xy_as_yx(right_pts, tol=10.0)
    if left_fit is None or right_fit is None:
        return None if not debug else (None, smeared, None, None)
    m_left, b_left = left_fit
    m_right, b_right = right_fit

    # Intersect each pair to get the 4 corners.
    def intersect(m_h, b_h, m_v, b_v):
        # y = m_h*x + b_h   and   x = m_v*y + b_v
        # => x = m_v*(m_h*x + b_h) + b_v
        # => x*(1 - m_v*m_h) = m_v*b_h + b_v
        denom = 1.0 - m_v * m_h
        if abs(denom) < 1e-9:
            return None
        x = (m_v * b_h + b_v) / denom
        y = m_h * x + b_h
        return np.array([x, y])

    tl = intersect(m_top, b_top, m_left, b_left)
    tr = intersect(m_top, b_top, m_right, b_right)
    br = intersect(m_bot, b_bot, m_right, b_right)
    bl = intersect(m_bot, b_bot, m_left, b_left)
    if any(c is None for c in (tl, tr, br, bl)):
        return None if not debug else (None, smeared, None, None)
    corners = np.stack([tl, tr, br, bl]).astype(np.float32)

    if debug:
        return corners, smeared, (top_pts, bot_pts), (top_fit, bot_fit)
    return corners


def tighten_corners_to_tops(
    frame: np.ndarray,
    loose_corners: np.ndarray,
    ratio: float = BLACK_TO_WHITE_KEY_HEIGHT_RATIO,
) -> np.ndarray:
    """Return a tightened corner set that excludes key front-faces.

    Uses the black-key geometry as a yardstick: after warping with the loose
    corners, detect y_black_top and y_black_bottom in the warp, compute the
    expected end of key tops (y_black_top + black_height / ratio), and
    inverse-project that line back to original-image coords to replace the
    bottom rail. The returned corners produce a warp tight to the physical
    top surfaces of the keys, without fronts/shadows/body below.

    Falls back to the input loose_corners if the refinement can't be applied
    reliably (e.g., black keys not clearly detectable in the warp).
    """
    tl, tr, br, bl = loose_corners
    try:
        warp_out_h = 220
        top_len = float(np.linalg.norm(tr - tl))
        bot_len = float(np.linalg.norm(br - bl))
        left_len = float(np.linalg.norm(bl - tl))
        right_len = float(np.linalg.norm(br - tr))
        avg_w = (top_len + bot_len) / 2
        avg_h = (left_len + right_len) / 2
        out_w = max(100, int(avg_w * warp_out_h / max(1.0, avg_h)))
        dst = np.array([
            [0, 0], [out_w - 1, 0], [out_w - 1, warp_out_h - 1], [0, warp_out_h - 1],
        ], dtype=np.float32)
        M_forward = cv2.getPerspectiveTransform(loose_corners.astype(np.float32), dst)
        M_inverse = np.linalg.inv(M_forward)
        warp_preview = cv2.warpPerspective(frame, M_forward, (out_w, warp_out_h))

        g_prev = cv2.cvtColor(warp_preview, cv2.COLOR_BGR2GRAY)
        sy = cv2.Sobel(g_prev, cv2.CV_32F, 0, 1, ksize=3)
        dl = np.clip(sy, 0, None).sum(axis=1)
        s_db = dl[: int(0.75 * warp_out_h)].copy()
        s_db[: int(0.1 * warp_out_h)] = 0
        y_black_bottom_w = int(np.argmax(s_db))
        ld = np.clip(-sy, 0, None).sum(axis=1)
        top_search_end = min(y_black_bottom_w - 5, int(0.4 * warp_out_h))
        if top_search_end > 3:
            s_ld = ld[:top_search_end]
            peak_ld = float(s_ld.max()); med_ld = float(np.median(s_ld))
            y_black_top_w = int(np.argmax(s_ld)) if peak_ld > 3.0 * max(1.0, med_ld) else 0
        else:
            y_black_top_w = 0

        bh_w = y_black_bottom_w - y_black_top_w
        if bh_w <= 10:
            return loose_corners
        wh_w = int(bh_w / ratio)
        expected_bot_w = y_black_top_w + wh_w
        if not (y_black_bottom_w + 5 < expected_bot_w < warp_out_h - 2):
            return loose_corners

        # Inverse-project the expected-bottom line back to original coords.
        p_l_warp = np.array([[0, expected_bot_w]], dtype=np.float32).reshape(-1, 1, 2)
        p_r_warp = np.array([[out_w - 1, expected_bot_w]], dtype=np.float32).reshape(-1, 1, 2)
        new_bl = cv2.perspectiveTransform(p_l_warp, M_inverse)[0, 0]
        new_br = cv2.perspectiveTransform(p_r_warp, M_inverse)[0, 0]
        return np.stack([tl, tr, new_br, new_bl]).astype(np.float32)
    except Exception:
        return loose_corners


def _trim_warp_bottom(warped: np.ndarray) -> np.ndarray:
    """Trim the warp bottom to the *computed* end of the white key tops,
    using black keys as a yardstick. Black keys are ~60% as tall as white
    keys on every piano, so:

        white_key_height = black_key_height / 0.60

    We detect y_black_top (first strong light→dark row, the top of the
    blacks) and y_black_bottom (strongest dark→light, the bottom of blacks).
    Trim at y_black_top + white_key_height. No fragile edge-detection in
    the key-fronts region needed.
    """
    if warped is None or warped.size < 100:
        return warped
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # y_black_bottom: strongest DARK→LIGHT horizontal edge (blacks end, whites begin).
    dark_to_light = np.clip(sobel_y, 0, None).sum(axis=1)
    s_db = dark_to_light[: int(0.75 * h)].copy()
    s_db[: int(0.1 * h)] = 0
    y_black_bottom = int(np.argmax(s_db))

    # y_black_top: strongest LIGHT→DARK row in the TOP 40% of the warp.
    # This bounds away from y_black_bottom (which has strong gradient bleed
    # from the main black/white boundary). For tight warps the blacks start
    # near y=0 so the search often finds no strong top edge — fall back.
    light_to_dark = np.clip(-sobel_y, 0, None).sum(axis=1)
    search_top_end = min(y_black_bottom - 5, int(0.4 * h))
    if search_top_end > 3:
        s_ld = light_to_dark[:search_top_end].copy()
        peak_ld = float(s_ld.max())
        med_ld = float(np.median(s_ld))
        if peak_ld > 3.0 * max(1.0, med_ld):
            y_black_top = int(np.argmax(s_ld))
        else:
            y_black_top = 0
    else:
        y_black_top = 0

    # Physical-ratio trim: black keys are ~BLACK_TO_WHITE_KEY_HEIGHT_RATIO
    # as tall as white keys (piano geometry). Compute expected white-key
    # bottom from measured black-key height; trim there if it's meaningfully
    # above the current warp bottom.
    black_height = y_black_bottom - y_black_top
    if black_height <= 5:
        return warped
    white_height = int(black_height / BLACK_TO_WHITE_KEY_HEIGHT_RATIO)
    expected_bottom = y_black_top + white_height
    if expected_bottom < h - 3 and expected_bottom > y_black_bottom + 5:
        return warped[:expected_bottom + 1, :, :]
    return warped


def warp_from_corners(img: np.ndarray, corners: np.ndarray, out_height: int = 220):
    tl, tr, br, bl = corners
    top_len = float(np.linalg.norm(tr - tl))
    bot_len = float(np.linalg.norm(br - bl))
    left_len = float(np.linalg.norm(bl - tl))
    right_len = float(np.linalg.norm(br - tr))
    avg_w = (top_len + bot_len) / 2
    avg_h = (left_len + right_len) / 2
    out_w = max(100, int(avg_w * out_height / max(1.0, avg_h)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_height - 1],
                    [0, out_height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_height))


def process_one(path: str):
    img = load_image(path)
    result = find_corners_auto(img, debug=True)
    corners, smeared, pts, fits = result
    name = Path(path).stem
    if corners is None:
        print(f"{name}: FAILED to detect corners")
        return name, img, None, None

    vis = img.copy()
    pts_int = corners.astype(int)
    for (x, y), lbl in zip(pts_int, ["TL", "TR", "BR", "BL"]):
        cv2.circle(vis, (x, y), 14, (0, 255, 0), -1)
        cv2.putText(vis, lbl, (x + 18, y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.polylines(vis, [pts_int.reshape(-1, 1, 2)], True, (0, 255, 0), 4)

    # Two warps: loose (includes key fronts — future press detection) and
    # tight (just key tops — for labeling).
    warped_loose = warp_from_corners(img, corners)
    tight_corners = tighten_corners_to_tops(img, corners)
    warped_tight = warp_from_corners(img, tight_corners)
    labeled = draw_labels_tight_crop(warped_tight)
    return name, vis, warped_loose, labeled


CELL_W, CELL_H = 720, 360
def _fit_cell(img, label):
    if img is None:
        img = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(CELL_W / w, CELL_H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(img, (nw, nh))
    out = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    y0 = (CELL_H - nh) // 2
    x0 = (CELL_W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    for w_c, t_c in [(4, (0, 0, 0)), (1, (255, 255, 255))]:
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, t_c, w_c, cv2.LINE_AA)
    return out


def main(paths):
    rows = []
    for p in paths:
        name, vis, warped, labeled = process_one(p)
        rows.append(np.hstack([
            _fit_cell(vis, f"{name} | corners"),
            _fit_cell(warped, f"{name} | warped"),
            _fit_cell(labeled, f"{name} | warped + labels"),
        ]))
    grid = np.vstack(rows)
    out = Path("auto_calib_result.png")
    cv2.imwrite(str(out), grid)
    print(f"wrote {out.resolve()} — {grid.shape[1]}x{grid.shape[0]}")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else [
        "piano_photos/IMG_9064.jpg",
        "piano_photos/IMG_9066.jpg",
        "piano_photos/IMG_9073.jpg",
    ])
