"""Piano keyboard detection + labeling pipeline.

This module implements detection + labeling for *tight* keyboard crops
(where the whole warped image IS the keyboard region, no surrounding
body/floor). The crop is produced either by:

1. ``find_keyboard_bbox`` in this file (heuristic axis-aligned crop), or
2. ``auto_calibrate.find_corners_auto`` (trapezoidal perspective warp), or
3. ``manual_calibrate`` (4-click manual corners).

The labeler annotates the warped keys with:

- a red horizontal line at the dynamically-detected black/white boundary
- blue polygons around each detected black key (actual pixel shape, with
  the per-blob template projection giving consistent key contours even
  on heavy side-view warps where individual key bodies look rectangular)
- yellow vertical lines at white-key seams, drawn through every row
  where no black-key polygon covers the seam's column (= unified rule:
  full-height in E-F/B-C gaps, clipped above polygon-covered rows)

Black-key detection (`_detect_blacks_2d`) uses 2D Otsu connected-component
on the upper band, then splits merged blobs at U-valley positions in the
blob's bottom-y profile. For each merged blob it picks the camera-FAR
outer piece's actual contour as a local template and projects it onto
the inner pieces — the template carries the only "complete" key shape
in the blob, since inner keys have ambiguous boundaries shared with
neighbours. Z-order clipping handles overlap between projected pieces:
the closer piece wins.

White-key seam detection uses Sobel-x on the white band (vertical-edge
strength per column) + local-median gap-fill for missed seams + edge
extrapolation past the first/last detected seam.

A `far_side` parameter (``"right"`` or ``"left"``) selects which outer
piece of each merged blob is used as the local template — set this to
the camera-far direction. When two cameras run as a pair, each cam
passes its own ``far_side`` and its results are fused downstream.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from seg_to_keys import warp_to_piano #isolate_white, warp_key_lines
from stream_webcams import open_canon_streams


def load_image(path: str) -> np.ndarray:
    """Load an image by path. Falls back to PIL for formats OpenCV may not
    ship with (e.g. AVIF on some builds)."""
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image

    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


BLACK_NOTE_SEMITONES = ["C#", "D#", "F#", "G#", "A#"]
WHITE_NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]


def _label_notes_61key(
    black_centers: list[int],
    start_octave: int = 2,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Given N detected black-key center x-positions in warped coords, return
    (black_labels, white_labels) where each entry is ``(x, "NoteName")``.

    Assumes a 61-key board (leftmost black = C#2). If the count differs from
    25, returns empty lists and lets the caller fall back to geometric-only
    output. Gap pattern per octave is SWSSW (S = 1 white between adjacent
    blacks, W = 2 whites between — the E-F and B-C gaps).
    """
    n = len(black_centers)
    if n != 25:
        return [], []
    centers = sorted(black_centers)
    black_labels = [(cx, f"{BLACK_NOTE_SEMITONES[i % 5]}{start_octave + i // 5}")
                    for i, cx in enumerate(centers)]

    # White letters progress starting at D (since leftmost black = C#,
    # the white to its right is D).
    gaps = np.diff(centers)
    small_gap = float(np.median(sorted(gaps)[: max(1, len(gaps) // 2)]))
    white_labels: list[tuple[int, str]] = []
    white_idx = 1  # start at D (index 1 of WHITE_NOTE_NAMES)
    octave = start_octave

    # Left-of-first white = C{start_octave}.
    white_labels.append((centers[0] - int(small_gap * 0.5), f"C{start_octave}"))

    # Use the CANONICAL SWSSW pattern hardcoded for a 61-key board (5
    # octaves of black keys: SWSSW * 4 cross-octave Ws + SWSS for the
    # last partial octave). 24 chars. More robust than per-gap S/W
    # classification, which can misclassify borderline wide gaps when
    # perspective compresses the wide-gap differential.
    canonical = ("SWSSW" * 5)[:24]

    for i in range(n - 1):
        a, b = centers[i], centers[i + 1]
        is_wide = canonical[i] == "W"
        if is_wide:
            # 2 whites between.
            for k, frac in enumerate((0.33, 0.66)):
                letter = WHITE_NOTE_NAMES[white_idx % 7]
                white_labels.append((a + int((b - a) * frac), f"{letter}{octave}"))
                white_idx += 1
                if white_idx % 7 == 0:
                    octave += 1
        else:
            letter = WHITE_NOTE_NAMES[white_idx % 7]
            white_labels.append(((a + b) // 2, f"{letter}{octave}"))
            white_idx += 1
            if white_idx % 7 == 0:
                octave += 1

    # Two more whites past the last black (B, then C next octave).
    last = centers[-1]
    for k in range(2):
        letter = WHITE_NOTE_NAMES[white_idx % 7]
        white_labels.append((last + int(small_gap * (0.5 + k * 0.7)),
                             f"{letter}{octave}"))
        white_idx += 1
        if white_idx % 7 == 0:
            octave += 1
    return black_labels, white_labels


# ── Labeler on a tight warp ──────────────────────────────────────────────────

# ── Black-key detection helpers ─────────────────────────────────────────────

def _split_blob_by_xclip(
    contour: np.ndarray, x0: int, y0: int, bw: int, bh: int,
    n_keys: int, target_w: float, far_side: str = "right",
) -> list[tuple[tuple[int, int, int, int], np.ndarray]] | None:
    """Split a merged-blob into n_keys pieces.
      - BOTH outer pieces (leftmost and rightmost in the blob) keep
        their *actual* contours via x-clip — each has one CORRECT outer
        edge (the blob's left or right boundary, which is the actual
        keyboard's left/right key edge there, not artificial).
      - MIDDLE pieces have artificial U-valley boundaries on both sides,
        so we replace them with a copy of the camera-FAR outer piece's
        contour translated to their U-valley center.

    far_side: which side of the blob is the camera-far side ('right' or
    'left'). Used only to pick which outer piece's shape to project onto
    middle pieces. Won't overwrite either outer piece's own contour.
    """
    if n_keys < 2:
        return None
    local = np.zeros((bh, bw), dtype=np.uint8)
    cv2.drawContours(
        local, [contour - np.array([[[x0, y0]]], dtype=np.int32)],
        -1, 1, thickness=-1,
    )
    bottom_y = np.full(bw, bh, dtype=np.float32)
    for x in range(bw):
        col_nz = np.nonzero(local[:, x])[0]
        if len(col_nz) > 0:
            bottom_y[x] = float(col_nz.max())
    k_s = max(3, bw // 30)
    if k_s % 2 == 0:
        k_s += 1
    bot_sm = np.convolve(bottom_y, np.ones(k_s) / k_s, mode="same")
    candidates = []
    margin = max(2, bw // 20)
    # Catch plateau-left-edges as valleys: a flat bottom of a U-valley
    # (multiple consecutive equal values at the deepest point) wouldn't
    # be caught by strict <. Use < on the left and <= on the right so
    # the plateau's left edge becomes the valley position.
    for i in range(margin, bw - margin):
        if bot_sm[i] < bot_sm[i - 1] and bot_sm[i] <= bot_sm[i + 1]:
            candidates.append(i)
    # Filter spurious local minima from noise/distortion within a single
    # key body via three layers, each robust to camera angle:
    #   1. MIN SPACING: real U-valleys are at least ~half a key-width
    #      apart geometrically (adjacent keys can't be packed closer).
    #   2. MAX VALLEY COUNT: a blob of width bw fits at most
    #      round(bw / median_w) keys, so it has at most that many minus
    #      1 real U-valleys. Greedy by depth keeps the strongest dips.
    #   3. MIN PROMINENCE: small absolute floor (≥2 px below local peak)
    #      rejects flat-region noise on extremely uniform far-blobs.
    win = max(margin, int(0.5 * target_w))
    min_spacing = max(3, int(0.5 * target_w))
    max_valleys = max(0, int(round(bw / target_w)) - 1)

    def _prominence(i: int) -> float:
        lo = max(0, i - win)
        hi = min(bw, i + win + 1)
        return float(bot_sm[lo:hi].max()) - float(bot_sm[i])

    candidates.sort(key=lambda i: bot_sm[i])  # deepest first
    selected: list[int] = []
    for c in candidates:
        if len(selected) >= max_valleys:
            break
        if _prominence(c) < 2.0:
            continue
        if any(abs(c - v) < min_spacing for v in selected):
            continue
        selected.append(c)
    valleys = sorted(selected)
    n_keys = len(valleys) + 1
    if n_keys < 2:
        return None
    splits = [0] + valleys + [bw]

    def _piece_contour(x_lo: int, x_hi: int) -> np.ndarray | None:
        m = np.zeros_like(local)
        m[:, x_lo:x_hi] = local[:, x_lo:x_hi]
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        return max(cs, key=cv2.contourArea)

    # Outer pieces — clamp width to ≤1.3*target_w to keep extreme edges
    # sane while still using their own actual contour.
    max_w = int(round(1.3 * target_w))
    left_lo, left_hi = splits[0], splits[1]
    if (left_hi - left_lo) > max_w:
        left_lo = max(0, left_hi - max_w)
    right_lo, right_hi = splits[-2], splits[-1]
    if (right_hi - right_lo) > max_w:
        right_hi = min(bw, right_lo + max_w)
    left_c = _piece_contour(left_lo, left_hi)
    right_c = _piece_contour(right_lo, right_hi)
    if left_c is None or right_c is None:
        return None
    # Far-outer's contour is the ONLY complete contour we have — it's
    # the only key in the blob whose body wasn't cut short where it
    # overlaps with a neighbour at the bottom. Every other piece in the
    # blob has an incomplete bottom (gets cropped by the U-valley to
    # the next key on the far side). So we project the far-outer's
    # *full* shape onto every other piece, anchored at that piece's
    # bottom-near-corner (the U-valley on the side AWAY from far_side).
    #
    # No clipping — each polygon is meant to represent a *complete*
    # key body, so adjacent polygons may overlap in pixel space. That's
    # fine; the regions still each correspond to one logical key.
    if far_side == "right":
        template_c = right_c
        # Anchor = template's MIN x (its left U-valley = bottom-near-corner
        # on the far-side template).
        template_anchor_local = float(template_c[:, 0, 0].min())
    else:
        template_c = left_c
        template_anchor_local = float(template_c[:, 0, 0].max())

    # Build each piece's full template-projected polygon first.
    raw_polys: list[np.ndarray] = []
    for k in range(n_keys):
        if far_side == "right" and k == n_keys - 1:
            piece_local = template_c
        elif far_side == "left" and k == 0:
            piece_local = template_c
        else:
            if far_side == "right":
                target = float(splits[k])
            else:
                target = float(splits[k + 1])
            shift = int(round(target - template_anchor_local))
            piece_local = template_c + np.array([[[shift, 0]]], dtype=np.int32)
        raw_polys.append(piece_local)

    # Z-order clip: pieces overlap because each was drawn with the full
    # far-outer template. The CLOSER key (camera-near side) takes
    # priority — its full outline shows, and the next-farther key's
    # near-side gets occluded. Process from close→far, accumulating an
    # occluder mask; each subsequent piece's polygon is clipped by it.
    if far_side == "right":
        order = list(range(n_keys))  # 0 = closest (leftmost)
    else:
        order = list(range(n_keys - 1, -1, -1))  # n-1 = closest (rightmost)

    occluder = np.zeros((bh, bw), dtype=np.uint8)
    clipped: list[np.ndarray | None] = [None] * n_keys
    for rank, k in enumerate(order):
        pm = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(pm, [raw_polys[k]], -1, 1, thickness=-1)
        # UNION with the actual blob mask within this piece's x-range:
        # the template defines the *shape* but any visibly-black pixel
        # inside the piece's x-range must be inside the polygon. Without
        # this, the template (extracted from the narrowest far-most key)
        # leaves out real black-key pixels in wider closer pieces.
        x_lo, x_hi = splits[k], splits[k + 1]
        x_range_mask = np.zeros((bh, bw), dtype=np.uint8)
        x_range_mask[:, x_lo:x_hi] = 1
        pm = pm | (local & x_range_mask)
        if rank > 0:
            pm[occluder > 0] = 0
        cs2, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs2:
            clipped[k] = max(cs2, key=cv2.contourArea)
        else:
            clipped[k] = raw_polys[k]
        cv2.drawContours(occluder, [clipped[k]], -1, 1, thickness=-1)

    out: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for k in range(n_keys):
        piece_img = clipped[k] + np.array([[[x0, y0]]], dtype=np.int32)
        bx_, by_, bw_t, bh_t = cv2.boundingRect(piece_img)
        out.append(((bx_, by_, bw_t, bh_t), piece_img))
    return out if len(out) >= 2 else None


def _detect_blacks_2d(
    gray: np.ndarray, y_black_bottom: int, w: int,
    far_side: str = "right",
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    """Direct 2D Otsu connected-component detection of black keys in the
    upper band. Oversized merged blobs (multiple adjacent keys connected
    via shadow / anti-aliasing) get SPLIT into individual keys at the
    U-valleys of the blob's bottom contour. ``far_side`` ('right' or
    'left') tells the splitter which outer-piece's contour to use as the
    template for inner pieces — set this to the direction *away* from
    the camera (e.g. left-mounted cam → far_side='right'). Returns
    (rects, polys) sorted left-to-right.
    """
    upper = gray[:y_black_bottom, :]
    expected_bk = w / 60
    blur = cv2.GaussianBlur(upper, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # MORPH_OPEN removes pixel-level noise. We deliberately SKIP MORPH_CLOSE
    # so thin white slivers between adjacent keys stay visible — Otsu's
    # natural separation does most of the splitting work for us.
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # First pass: collect all blobs that are tall enough to be keys, but
    # without rejecting wide ones (we'll split them after).
    raw: list[tuple[int, int, int, int, np.ndarray]] = []
    for c in contours:
        x_, y_, bw_, bh_ = cv2.boundingRect(c)
        if bh_ < 0.4 * y_black_bottom:
            continue
        if bw_ < max(3, int(0.2 * expected_bk)):
            continue
        if bw_ > 0.6 * w:
            continue  # whole-band over-segment, drop
        raw.append((x_, y_, bw_, bh_, c))
    if not raw:
        return [], []
    # Estimate true single-key width. On side-view warps most blobs are
    # merged (2-3 keys), so simple median is biased high. Use the fact
    # that observed widths follow w_obs = N * w_single (N = 1, 2, 3, …):
    # score each candidate width by how many other widths are close to
    # an integer multiple of it. Constrain the candidate to a plausible
    # range around the geometric expectation (w/60).
    raw_widths = [r[2] for r in raw if r[2] > w / 200]
    median_w = float(expected_bk)
    if raw_widths:
        cand_lo = 0.5 * expected_bk
        cand_hi = 2.5 * expected_bk
        candidates = sorted({rw for rw in raw_widths if cand_lo <= rw <= cand_hi})
        if not candidates:
            # Fallback: take the smallest blobs (likely singles).
            candidates = sorted(set(raw_widths))[: max(1, len(raw_widths) // 4)]
        best_score, best_w = -1.0, float(np.median(candidates))
        for c in candidates:
            score = 0.0
            for ow in raw_widths:
                ratio = ow / c
                n = max(1, int(round(ratio)))
                # Within 20% of an integer multiple → counts as a fit.
                if abs(ratio - n) / n < 0.20:
                    score += 1.0
            if c < 0.7 * expected_bk:
                score -= 0.5 * len(raw_widths)  # penalize over-small candidates
            if score > best_score:
                best_score, best_w = score, c
        median_w = float(best_w)
    # Estimate expected aspect-ratio (h/w) of a single key from the
    # blobs that are clearly single (width ≈ median_w). Used by the
    # per-blob splitter to pick the best outer piece as local template.
    expected_aspect = float(y_black_bottom) / max(1.0, median_w)

    # Split oversized blobs: width > 1.3 * median_w → likely 2+ merged
    # keys. Use a *per-blob local template*: the outermost piece on the
    # camera-far side has a clean outer edge (the blob boundary IS the
    # actual key edge there). We pick whichever outer piece (left or
    # right) has the more key-like aspect ratio, then translate that
    # piece's polygon to each inner piece's center. Inner-key polygons
    # thus get the local-region's actual perspective shape, not a global
    # template that may have wrong distortion.
    rects, polys = [], []
    for x_, y_, bw_, bh_, contour in raw:
        if bw_ <= 1.3 * median_w:
            rects.append((x_, y_, bw_, bh_))
            eps = max(0.5, 0.002 * cv2.arcLength(contour, True))
            polys.append(cv2.approxPolyDP(contour, eps, True))
            continue
        n_keys = max(2, int(round(bw_ / median_w)))
        split = _split_blob_by_xclip(
            contour, x_, y_, bw_, bh_, n_keys, median_w, far_side,
        )
        if split is not None:
            for sub_rect, sub_poly in split:
                rects.append(sub_rect)
                eps = max(0.5, 0.002 * cv2.arcLength(sub_poly, True))
                polys.append(cv2.approxPolyDP(sub_poly, eps, True))
        else:
            # Splitter couldn't find enough U-valleys — use the blob's
            # ACTUAL contour as one polygon. Under-counts keys but at
            # least the polygon traces real pixel edges (vs. the prior
            # axis-aligned-bounding-box fallback that drew fake rects).
            rects.append((x_, y_, bw_, bh_))
            eps = max(0.5, 0.002 * cv2.arcLength(contour, True))
            polys.append(cv2.approxPolyDP(contour, eps, True))
    # Geometric edge guard for the 61-key C-to-C keyboard layout.
    # The keyboard's leftmost white is C2 (so first black-key C#2 sits
    # ~0.7 white-keys from the left edge) and rightmost white is C7
    # preceded by B6 (so last black-key A#6 sits ~1.5 white-keys from
    # the right edge). Any polygon whose outer x extends past these
    # geometric caps is over-extending into the white-key buffer area
    # — likely from Otsu including warp-edge dark artifacts. Trim it.
    # This is symmetric across camera angles because the keyboard's
    # layout doesn't depend on which side the camera is on.
    white_key_w = w / 36.0
    left_geom_cap = max(2, int(0.5 * white_key_w))
    right_geom_cap = max(left_geom_cap + 1, w - int(1.5 * white_key_w))

    def _trim_poly_x(p: np.ndarray, x_lo: int, x_hi: int) -> np.ndarray:
        """Clip polygon to x in [x_lo, x_hi] via rasterize → mask →
        re-extract. Preserves the polygon's actual edge shape on the
        un-clipped sides.
        """
        if int(p[:, 0, 0].min()) >= x_lo and int(p[:, 0, 0].max()) <= x_hi:
            return p
        bb_x, bb_y, bw_p, bh_p = cv2.boundingRect(p)
        pad = 2
        pmask = np.zeros((bh_p + 2 * pad, bw_p + 2 * pad), dtype=np.uint8)
        p_local = p - np.array([[[bb_x - pad, bb_y - pad]]], dtype=np.int32)
        cv2.drawContours(pmask, [p_local], -1, 1, thickness=-1)
        cap_lo = max(0, x_lo - (bb_x - pad))
        cap_hi = min(pmask.shape[1], x_hi - (bb_x - pad) + 1)
        if cap_lo > 0:
            pmask[:, :cap_lo] = 0
        if cap_hi < pmask.shape[1]:
            pmask[:, cap_hi:] = 0
        cs, _ = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return p
        c = max(cs, key=cv2.contourArea)
        return c + np.array([[[bb_x - pad, bb_y - pad]]], dtype=np.int32)

    for i in range(len(polys)):
        new_p = _trim_poly_x(polys[i], left_geom_cap, right_geom_cap)
        if new_p is not polys[i]:
            polys[i] = new_p
            rects[i] = cv2.boundingRect(new_p)

    order = sorted(range(len(rects)), key=lambda k: rects[k][0])
    return [rects[k] for k in order], [polys[k] for k in order]


def _detect_blacks_1d(
    gray: np.ndarray, y_black_bottom: int, w: int,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray | None]]:
    """1D column-projection fallback for black-key detection. Used when 2D
    Otsu can't find the bimodal split (typically top-down warps where the
    upper band is dominated by black-key pixels).
    """
    upper = gray[:y_black_bottom, :]
    mid_top = int(0.2 * y_black_bottom)
    mid_bot = int(0.8 * y_black_bottom)
    strip = upper[mid_top:mid_bot, :]
    col_mean = strip.mean(axis=0)
    ksz = max(5, int(w / 180))
    if ksz % 2 == 0:
        ksz += 1
    sm = np.convolve(col_mean, np.ones(ksz) / ksz, mode="same")
    win_size = max(51, int(w * 0.15))
    if win_size % 2 == 0:
        win_size += 1
    half = win_size // 2
    padded = np.concatenate([sm[half:0:-1], sm, sm[-2:-half - 2:-1]])
    local_mean = np.convolve(padded, np.ones(win_size) / win_size, mode="valid")
    dark_thr_arr = local_mean * 0.65
    med = float(np.median(sm))
    p10 = float(np.percentile(sm, 10))
    global_floor = p10 + 0.8 * (med - p10)
    dark_thr_arr = np.minimum(dark_thr_arr, global_floor)
    dark_thr = float(np.mean(dark_thr_arr))
    dark = sm <= dark_thr_arr
    n = len(dark)
    gap_merge = max(2, int(0.1 * (w / 36)))
    i = 0
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
    expected_bk = w / 60
    min_w = max(3, int(0.35 * expected_bk))
    max_w = max(min_w + 1, int(1.6 * expected_bk))
    row_dark_thr = dark_thr + 0.3 * (med - dark_thr)
    extent_thr = dark_thr + 0.7 * (med - dark_thr)
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
                left = i
                while left > 0 and sm[left - 1] <= extent_thr and (i - left) < bw:
                    left -= 1
                right = j - 1
                while right < len(sm) - 1 and sm[right + 1] <= extent_thr and (right - (j - 1)) < bw:
                    right += 1
                x0_k = max(0, left)
                x1_k = min(w, right + 1)
                strip_img = upper[:, x0_k:x1_k]
                dark_mask = (strip_img <= row_dark_thr).astype(np.uint8) * 255
                dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, vkernel)
                cc, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                poly = None
                if cc:
                    biggest = max(cc, key=cv2.contourArea)
                    if cv2.contourArea(biggest) >= 0.2 * bw * y_black_bottom:
                        eps = max(0.5, 0.002 * cv2.arcLength(biggest, True))
                        approx = cv2.approxPolyDP(biggest, eps, True)
                        poly = approx + np.array([[[x0_k, 0]]], dtype=np.int32)
                black_polys.append(poly)
                black_rects.append((i, 0, bw, y_black_bottom))
            i = j
        else:
            i += 1
    return black_rects, black_polys


def _classify_gaps_local(gaps: np.ndarray, window: int = 4, ratio: float = 1.4) -> str:
    """Per-gap S/W classification: gap > ratio × local-median (excluding self) → W."""
    out = []
    for i in range(len(gaps)):
        lo, hi = max(0, i - window), min(len(gaps), i + window + 1)
        nbrs = np.array([gaps[k] for k in range(lo, hi) if k != i])
        local = float(np.median(nbrs)) if len(nbrs) else float(gaps[i])
        out.append("W" if gaps[i] > ratio * local else "S")
    return "".join(out)


def _project_to_25(
    rects: list[tuple[int, int, int, int]],
    polys: list[np.ndarray | None],
    w: int,
    y_black_bottom: int,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray | None], list[str]]:
    """Use SWSSW alignment to fill in missing black keys. Returns the full
    25-key list with each tagged 'detected' or 'inferred'. Inferred keys
    get template polygons translated from the nearest detected anchor.
    """
    n = len(rects)
    sources = ["detected"] * n
    if n < 4:
        return list(rects), list(polys), sources
    centers = [r[0] + r[2] // 2 for r in rects]
    widths = [r[2] for r in rects]
    gaps = np.diff(centers)
    gap_classes = _classify_gaps_local(gaps)
    canonical = ("SWSSW" * 5)[:24]
    # Truncate observed if longer than canonical (rare: spurious extra blob).
    cmp_obs = gap_classes[: len(canonical)]
    best_off, best_sc = 0, -1
    span = max(1, len(canonical) - len(cmp_obs) + 1)
    for off in range(span):
        sc = sum(
            1 for k, c in enumerate(cmp_obs)
            if (off + k) < len(canonical) and canonical[off + k] == c
        )
        if sc > best_sc:
            best_sc, best_off = sc, off
    canonical_idx = [best_off + k for k in range(n)]
    s_gaps = [int(g) for g, c in zip(gaps, gap_classes) if c == "S"]
    s_unit = float(np.median(s_gaps)) if s_gaps else float(w / 36.0)
    out_rects, out_polys, out_sources = list(rects), list(polys), list(sources)
    assigned = set(canonical_idx)
    for ci in range(25):
        if ci in assigned:
            continue
        left = right = None
        for k, dci in enumerate(canonical_idx):
            if dci < ci and (left is None or dci > left[0]):
                left = (dci, centers[k], widths[k], rects[k], polys[k])
            if dci > ci and (right is None or dci < right[0]):
                right = (dci, centers[k], widths[k], rects[k], polys[k])
        if left is None and right is None:
            continue
        if left is None:
            rci, rx, rw_, rrect, rp = right
            x_inf = rx - (rci - ci) * s_unit
            wid_inf = rw_
            template_poly, template_rect = rp, rrect
        elif right is None:
            lci, lx, lw_, lrect, lp = left
            x_inf = lx + (ci - lci) * s_unit
            wid_inf = lw_
            template_poly, template_rect = lp, lrect
        else:
            lci, lx, lw_, lrect, lp = left
            rci, rx, rw_, rrect, rp = right
            t = (ci - lci) / (rci - lci)
            x_inf = lx + t * (rx - lx)
            wid_inf = lw_ + t * (rw_ - lw_)
            if (ci - lci) <= (rci - ci):
                template_poly, template_rect = lp, lrect
            else:
                template_poly, template_rect = rp, rrect
        wid_inf = max(4, int(wid_inf))
        x_left_ = max(0, int(x_inf - wid_inf / 2))
        x_right_ = min(w, int(x_inf + wid_inf / 2))
        if x_right_ <= x_left_:
            continue
        # Don't extrapolate past the detected-key span: only allow inferred
        # keys within [leftmost_detected - s_unit, rightmost_detected + s_unit].
        # Prevents hallucinated keys past the keyboard's actual end.
        first_detected_x = centers[0] - s_unit
        last_detected_x = centers[-1] + s_unit
        if x_inf < first_detected_x or x_inf > last_detected_x:
            continue
        if template_poly is not None:
            tx_, _, tbw_, _ = template_rect
            tcx_ = tx_ + tbw_ / 2
            shift = int(round(x_inf - tcx_))
            inferred_poly = template_poly + np.array([[[shift, 0]]], dtype=np.int32)
        else:
            inferred_poly = None
        out_rects.append((x_left_, 0, x_right_ - x_left_, y_black_bottom))
        out_polys.append(inferred_poly)
        out_sources.append("inferred")
    order = sorted(range(len(out_rects)), key=lambda k: out_rects[k][0])
    return (
        [out_rects[k] for k in order],
        [out_polys[k] for k in order],
        [out_sources[k] for k in order],
    )


# ── Labeler on a tight warp ──────────────────────────────────────────────────

def draw_labels_tight_crop(
    warped: np.ndarray, label_notes: bool = True, far_side: str = "right",
) -> np.ndarray:
    """Annotate a tight-keyboard-crop warped image with detected key features.

    Assumes the warped image height IS the keyboard (no body above, no
    floor below). Pipeline:

    - Find ``y_black_bottom`` (the black/white boundary) via the strongest
      dark→light horizontal edge (Sobel-y) in the upper portion.
    - Detect black keys via 2D Otsu connected components, splitting any
      merged blob with U-valley analysis + per-blob far-template projection
      (``_detect_blacks_2d``); fall back to 1D column-projection
      (``_detect_blacks_1d``) when 2D yields too few keys.
    - Project the resulting set to the canonical 25 black-key positions
      using SWSSW-pattern alignment (``_project_to_25``), filling missing
      keys with translated template polygons.
    - Detect white-key seams via Sobel-x on the white band; gap-fill and
      edge-extrapolate to cover all expected ~37 boundaries.
    - Draw each seam through every row where no black-key polygon covers
      the seam's column (one unified rule for partial vs full-height).

    ``far_side``: which side of the warp is the camera-FAR side
    (``"right"`` or ``"left"``); used by the per-blob splitter to pick
    the local template piece. Set per-camera in a dual-cam rig.
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

    # --- Black-key detection: 2D primary, 1D fallback, then SWSSW projection ---
    rects_2d, polys_2d = _detect_blacks_2d(gray, y_black_bottom, w, far_side)
    if len(rects_2d) >= 8:
        black_rects = rects_2d
        black_polys: list[np.ndarray | None] = list(polys_2d)
    else:
        black_rects, black_polys = _detect_blacks_1d(gray, y_black_bottom, w)

    # Geometric fill-in: align detected to canonical SWSSW pattern, project
    # missing keys, copy nearest-detected polygon for each inferred key.
    black_rects, black_polys, black_sources = _project_to_25(
        black_rects, black_polys, w, y_black_bottom,
    )

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

    # --- White-key seams: Sobel-x peak detection on the white band.
    # Sum of |∂I/∂x| down each column gives a strong peak at every seam,
    # even when the seam itself is only marginally darker than the white
    # surface (low-contrast side-view warps where a pure column-mean
    # valley would collapse into noise). We don't depend on the
    # black-key list for *position*; black-key polygons are used
    # afterward only by the per-row drawing rule (skip rows where a
    # polygon covers).
    band_top = y_black_bottom + int(0.15 * (h - y_black_bottom))
    band_bot = h - 2
    white_band = gray[band_top:band_bot, :] if band_bot > band_top else gray[band_top:, :]
    seam_peaks: list[int] = []
    black_centers_sorted = sorted(int(x + bw2 / 2) for (x, _, bw2, _) in black_rects)
    if len(black_centers_sorted) >= 2:
        gaps_b = np.diff(black_centers_sorted)
        sorted_gaps_b = np.sort(gaps_b)
        small_gap = float(np.median(sorted_gaps_b[: max(1, len(sorted_gaps_b) // 2)]))
    else:
        small_gap = w / 36.0
    if white_band.size > 0:
        # Sobel-x highlights vertical-edge transitions — seams are exactly
        # vertical dark lines between bright key surfaces. Sum |∂I/∂x| down
        # each column gives a strong peak at every seam, even when the
        # seam itself is only marginally darker than the white surface
        # (low-contrast side-view warps where pure column-mean valleys
        # collapse into the noise floor).
        sx = cv2.Sobel(white_band, cv2.CV_32F, 1, 0, ksize=3)
        col_edge = np.abs(sx).sum(axis=0)
        k_wh = max(3, int(w / 500))
        if k_wh % 2 == 0:
            k_wh += 1
        sm_wh = np.convolve(col_edge, np.ones(k_wh) / k_wh, mode="same")
        seam_thr = float(np.percentile(sm_wh, 70))
        min_sep = max(4, int(w / 80))
        edge_margin = max(6, int(w / 200))
        for i in range(1, len(sm_wh) - 1):
            if i < edge_margin or i > len(sm_wh) - edge_margin:
                continue
            if sm_wh[i] >= seam_thr and sm_wh[i] >= sm_wh[i - 1] and sm_wh[i] >= sm_wh[i + 1]:
                if seam_peaks and i - seam_peaks[-1] < min_sep:
                    if sm_wh[i] > sm_wh[seam_peaks[-1]]:
                        seam_peaks[-1] = i
                else:
                    seam_peaks.append(i)

    # Geometric gap-fill: where two adjacent detected seams are spaced much
    # wider than the *local* median seam-spacing (window ±3), insert
    # evenly-spaced fill seams. Local median tracks the perspective-induced
    # spacing drift across the warp on side-view shots.
    if len(seam_peaks) >= 4:
        seam_gaps = np.diff(seam_peaks)
        filled: list[int] = [seam_peaks[0]]
        for k in range(len(seam_gaps)):
            lo, hi = max(0, k - 3), min(len(seam_gaps), k + 4)
            local_med = float(np.median([seam_gaps[j] for j in range(lo, hi) if j != k]))
            g = seam_gaps[k]
            n_fill = int(round(g / local_med)) - 1
            if n_fill >= 1 and g > 1.4 * local_med:
                step = g / (n_fill + 1)
                for j in range(1, n_fill + 1):
                    filled.append(int(round(seam_peaks[k] + j * step)))
            filled.append(seam_peaks[k + 1])
        seam_peaks = sorted(set(filled))

        # Edge extrapolation: extend past the leftmost / rightmost detected
        # seam using the LOCAL spacing at that end. Side-view warps lose
        # Sobel signal toward the foreshortened end (right side of `74`),
        # leaving the far end un-seamed. Stepping out by the local-end
        # median fills that span. We stretch the step slightly each
        # iteration to accommodate the shrinking-toward-far-end perspective
        # (or equivalently, the farther we get from detected anchors, the
        # less confidence in the exact spacing).
        if len(seam_peaks) >= 4:
            gaps_now = np.diff(seam_peaks)
            edge_n = max(3, len(gaps_now) // 5)
            left_step = float(np.median(gaps_now[:edge_n]))
            right_step = float(np.median(gaps_now[-edge_n:]))
            extended: list[int] = list(seam_peaks)
            x_l, step_l = seam_peaks[0] - left_step, left_step
            i = 0
            while x_l > 2 and i < 8:
                extended.append(int(round(x_l)))
                step_l *= 1.05
                x_l -= step_l
                i += 1
            x_r, step_r = seam_peaks[-1] + right_step, right_step
            i = 0
            while x_r < w - 3 and i < 8:
                extended.append(int(round(x_r)))
                step_r *= 1.05
                x_r += step_r
                i += 1
            seam_peaks = sorted(set(extended))

    # Build a polygon mask of all black-key shapes — the ground-truth
    # representation of where black keys actually are. Filled, so the
    # mask is True wherever a black-key polygon covers a pixel.
    black_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(black_mask, black_contours, -1, color=1, thickness=-1)

    # --- Draw geometric markers ---
    cv2.line(out, (0, y_black_bottom), (w - 1, y_black_bottom), (0, 0, 255), 2)
    # Each seam is drawn through every row where the polygon mask is empty
    # at its column — i.e., wherever there's no black key directly above
    # to obstruct it. Single unified rule: full-height where unobstructed
    # (E|F, B|C gaps), clipped above the polygon's bottom edge where a
    # black key sits, and naturally fills the visible white strip between
    # adjacent keys (the U-valley in a merged blob) above the red line.
    for bx in seam_peaks:
        col = black_mask[:, bx]
        in_run = False
        y_start = 0
        for y in range(h):
            if col[y] == 0 and not in_run:
                y_start = y
                in_run = True
            elif col[y] != 0 and in_run:
                if y - 1 - y_start >= 2:
                    cv2.line(out, (bx, y_start), (bx, y - 1), (0, 255, 255), 2)
                in_run = False
        if in_run and (h - 1 - y_start) >= 2:
            cv2.line(out, (bx, y_start), (bx, h - 1), (0, 255, 255), 2)
    # Detected = solid blue; inferred = solid blue too (so visually all 25
    # keys look outlined). Internal source flag is preserved for the
    # calibration JSON's confidence weighting later.
    for poly, src in zip(black_contours, black_sources):
        cv2.drawContours(out, [poly], -1, (255, 0, 0), 2)

    # --- Note labels (assumes 61-key board, leftmost black = C#2) ---
    if label_notes:
        black_centers = sorted(int(x + bw_ / 2) for (x, _, bw_, _) in black_rects)
        bl, wl = _label_notes_61key(black_centers)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale: assume labels can stagger across 2 rows, so each label
        # only needs to fit within ≈ 1.7 × the spacing between every-
        # other key.
        white_key_w = w / 36.0
        target_label_w = 1.5 * white_key_w
        ref_text = "C#3"
        (rw, _), _ = cv2.getTextSize(ref_text, font, 1.0, 2)
        scale = max(0.35, min(0.9, target_label_w / max(1.0, rw)))
        thick = 1 if scale < 0.5 else 2

        # Stagger: alternate every-other label between two y-rows so
        # adjacent labels never overlap horizontally.
        (_, ref_h), _ = cv2.getTextSize(ref_text, font, scale, thick)
        row_offset = ref_h + 4

        bl_sorted = sorted(bl, key=lambda lab: lab[0])
        for idx, (cx, name) in enumerate(bl_sorted):
            (tw, _), _ = cv2.getTextSize(name, font, scale, thick)
            row = idx % 2
            y = y_black_bottom - 6 - (row * row_offset)
            pos = (cx - tw // 2, y)
            cv2.putText(out, name, pos, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(out, name, pos, font, scale, (0, 255, 255), thick, cv2.LINE_AA)

        wl_sorted = sorted(wl, key=lambda lab: lab[0])
        for idx, (cx, name) in enumerate(wl_sorted):
            if not (0 <= cx < w):
                continue
            (tw, _), _ = cv2.getTextSize(name, font, scale, thick)
            row = idx % 2
            y = h - 6 - (row * row_offset)
            pos = (cx - tw // 2, y)
            cv2.putText(out, name, pos, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(out, name, pos, font, scale, (0, 200, 0), thick, cv2.LINE_AA)
    return out


# ── Demo ─────────────────────────────────────────────────────────────────────

def _demo_image(path: str) -> None:
    frame = load_image(path)
    # bbox = find_keyboard_bbox(frame)
    # if bbox is None:
    #     print("could not locate keyboard")
    #     return
    # warped = warp_to_bbox(frame, bbox)
    warped, _, _ = warp_to_piano(frame)
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
