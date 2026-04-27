"""Per-key region storage for handing off to live press-detection.

This module's job is the *storage and segmentation* layer between the
detection pipeline (``key_labeler.py``) and whatever live press-detection
gets built next. It does NOT do press detection itself — that's the
next dev's task.

What it provides:

- ``build_calibration_data(warped, corners, far_side, ...)`` — runs
  detection on a calibration frame and returns a JSON-serializable
  dict with one entry per key (polygon, label, source/confidence,
  baseline intensity, and a pre-shrunk "safe" subregion bbox).
- ``save_calibration(calib, path)`` / ``load_calibration(path)`` — IO.
- ``Calibration.load(path)`` — loads the JSON AND pre-rasterizes every
  key's safe-region polygon to a ``np.bool_`` mask, plus stores the
  perspective-transform matrix. So the next dev's per-frame work is
  literally: ``warped = calib.warp(frame); gray[mask].mean()`` for each
  key. No contour ops, no per-frame raster — fast enough for live use.

JSON schema (``<photo>_keys.json``):

::

    {
      "version": 1,
      "camera_id": "cam-left-01",
      "far_side": "right",                  # which side is camera-far
      "warp": {
        "corners_tl_tr_br_bl": [[x,y]*4],   # in source-image coords
        "out_size": [w, h]
      },
      "y_black_bottom": int,                # red boundary line
      "keys": [
        {
          "note": "C#2",
          "type": "black" | "white",
          "polygon": [[x,y], ...],          # in warp coords
          "bbox": [x, y, w, h],
          "source": "detected"              # has its own real contour
                  | "template_projected"    # got far-template translated
                  | "inferred"              # SWSSW projected, no anchor
                  | "geometric",            # white keys, derived from seams
          "confidence": 0.0..1.0,
          "baseline_intensity": float,      # mean gray inside safe_bbox
          "safe_bbox": [x, y, w, h]         # eroded sub-rect: edges +
                                            # finger occlusion buffer
        },
        ...
      ]
    }

Use:

    # one-time, per camera mount
    calib = build_calibration_data(warped_frame, corners, far_side="right")
    save_calibration(calib, "cam-left_keys.json")

    # later, in the next dev's live loop
    rt = Calibration.load("cam-left_keys.json")
    warped = rt.warp(frame_bgr)
    gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    for key in rt.keys:
        mean = gray[key.safe_mask].mean()
        # … press-detection logic here, using key.baseline_intensity,
        #   key.confidence, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import cv2
import numpy as np


# Confidence priors per polygon source. Saved to the JSON for the next
# dev to use as an inverse weight on detection thresholds (high-confidence
# keys can flag presses on smaller pixel deltas).
SOURCE_CONFIDENCE = {
    "detected": 0.95,
    "template_projected": 0.75,
    "inferred": 0.55,
    "geometric": 0.85,  # white-key regions derived from seams
}


# ── Calibration build (offline, one-time per camera mount) ─────────────────


def build_calibration_data(
    warped: np.ndarray,
    corners_tl_tr_br_bl: np.ndarray,
    far_side: str = "right",
    camera_id: str = "",
) -> dict:
    """Run detection on a calibration frame and produce the JSON-shape
    dict described in this module's docstring. Caller passes the warp
    corners (in source-image coords) so the same warp can be reproduced
    on live frames.
    """
    # Local imports keep this module decoupled at import time.
    from key_labeler import (
        _detect_blacks_2d,
        _detect_blacks_1d,
        _project_to_25,
        _label_notes_61key,
    )

    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # y_black_bottom (red line) from strongest dark→light Sobel-y row.
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    row_score = np.clip(sobel_y, 0, None).sum(axis=1)
    search = row_score[: int(0.75 * h)].copy()
    search[: int(0.1 * h)] = 0
    y_black_bottom = int(np.argmax(search))
    if y_black_bottom < int(0.3 * h):
        y_black_bottom = int(0.55 * h)

    # Black-key detection (2D primary, 1D fallback) → SWSSW projection.
    rects_2d, polys_2d = _detect_blacks_2d(gray, y_black_bottom, w, far_side)
    if len(rects_2d) >= 8:
        b_rects, b_polys = list(rects_2d), list(polys_2d)
    else:
        b_rects, b_polys = _detect_blacks_1d(gray, y_black_bottom, w)
    b_rects, b_polys, b_sources = _project_to_25(b_rects, b_polys, w, y_black_bottom)

    centers = sorted(int(rx + rw // 2) for (rx, _, rw, _) in b_rects)
    bl, wl = _label_notes_61key(centers)
    bl_by_x = {cx: name for cx, name in bl}

    keys: list[dict] = []
    for rect, poly, src in zip(b_rects, b_polys, b_sources):
        rx, ry, rw, rh = rect
        cx = rx + rw // 2
        if bl_by_x:
            nearest_cx = min(bl_by_x.keys(), key=lambda c: abs(c - cx))
            note = bl_by_x[nearest_cx]
        else:
            # Note labelling requires exactly 25 detected blacks; if the
            # detector hit 24 or 27, label assignment skips and we save
            # the polygon with a placeholder note so the geometry isn't
            # lost. (Listed under "Remaining bugs" in the README.)
            note = "?"
        if poly is None:
            poly = np.array([
                [[rx, ry]], [[rx + rw, ry]],
                [[rx + rw, ry + rh]], [[rx, ry + rh]],
            ], dtype=np.int32)
        # safe_bbox: shrink rect by 25% on each side. Drops anti-aliased
        # edge pixels and leaves a buffer for fingertip occlusion at
        # the very front of the key.
        sx = rx + int(0.25 * rw)
        sy = ry + int(0.25 * rh)
        sw = max(1, int(0.50 * rw))
        sh = max(1, int(0.50 * rh))
        baseline = float(gray[sy:sy + sh, sx:sx + sw].mean()) if sh > 0 and sw > 0 else 0.0
        keys.append({
            "note": note,
            "type": "black",
            "polygon": [[int(p[0][0]), int(p[0][1])] for p in poly],
            "bbox": [int(rx), int(ry), int(rw), int(rh)],
            "source": src,
            "confidence": SOURCE_CONFIDENCE.get(src, 0.5),
            "baseline_intensity": baseline,
            "safe_bbox": [int(sx), int(sy), int(sw), int(sh)],
        })

    # White-key seam positions are derived GEOMETRICALLY from the black
    # centers + canonical SWSSW pattern. This guarantees exactly 37 seam
    # positions for the 36 white-key regions, with each one in its
    # correct canonical slot — independent of Sobel-peak count drift.
    #
    #   For each black center:                   one seam AT the center
    #     (= a white-key boundary that has a black sitting on it)
    #   For each WIDE inter-black gap (W in SWSSW):  one extra seam at
    #     the gap midpoint (= the E-F or B-C full-height boundary)
    #   Past the last black:                     one seam at last + 1.5
    #     × small_gap (the B6-C7 boundary)
    #   Plus the two outer edges (x = 0, x = w-1)
    #
    # Total = 25 + 9 + 1 + 2 = 37 → 36 white regions.  Optional Sobel
    # snap-to-nearest refines each position to a real visual seam if
    # within tolerance; otherwise the geometric position is used as-is.
    band_top = y_black_bottom + int(0.15 * (h - y_black_bottom))
    white_band = gray[band_top:h - 2, :] if h - 2 > band_top else gray[band_top:, :]
    sobel_seams: list[int] = []
    if white_band.size > 0:
        sx_grad = cv2.Sobel(white_band, cv2.CV_32F, 1, 0, ksize=3)
        col_edge = np.abs(sx_grad).sum(axis=0)
        k_wh = max(3, int(w / 500))
        if k_wh % 2 == 0:
            k_wh += 1
        sm = np.convolve(col_edge, np.ones(k_wh) / k_wh, mode="same")
        thr = float(np.percentile(sm, 70))
        min_sep = max(4, int(w / 80))
        em = max(6, int(w / 200))
        for i in range(1, len(sm) - 1):
            if i < em or i > len(sm) - em:
                continue
            if sm[i] >= thr and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]:
                if sobel_seams and i - sobel_seams[-1] < min_sep:
                    if sm[i] > sm[sobel_seams[-1]]:
                        sobel_seams[-1] = i
                else:
                    sobel_seams.append(i)

    # Derive white-key boundaries DIRECTLY from the same `wl` label
    # positions that the labelling output uses. Each adjacent pair of
    # label x-positions defines a region; the seam between them is the
    # midpoint. This guarantees the JSON's white-key regions match the
    # labelling visualisation exactly — they're built from the same
    # positions. Optionally snap each midpoint to a nearby Sobel peak
    # for visual accuracy on the actual seam pixels.
    wl_label_xs = sorted(lab[0] for lab in wl) if wl else []
    if len(wl_label_xs) >= 2:
        # small_gap (white-key width estimate) = median pairwise distance
        # between adjacent label centers.
        wl_diffs = np.diff(wl_label_xs)
        small_gap = float(np.median(wl_diffs))

        snap_tol = max(3, int(0.4 * small_gap))

        def _snap(x: int) -> int:
            best = None
            best_d = snap_tol + 1
            for s in sobel_seams:
                d = abs(s - x)
                if d < best_d:
                    best_d, best = d, s
            return best if best is not None else x

        # Inner seams: midpoint between adjacent white-key label x's.
        inner_seams = [
            _snap((wl_label_xs[i] + wl_label_xs[i + 1]) // 2)
            for i in range(len(wl_label_xs) - 1)
        ]
        # Outer seams: half a white-key-width past the leftmost label
        # (left edge of leftmost white) and past the rightmost label
        # (right edge of rightmost white).
        left_outer = max(0, int(round(wl_label_xs[0] - small_gap / 2)))
        right_outer = min(w - 1, int(round(wl_label_xs[-1] + small_gap / 2)))
        seam_xs = sorted(set([left_outer] + inner_seams + [right_outer]))
    else:
        seam_xs = sorted(set([0] + sobel_seams + [w - 1]))

    # Pre-rasterize the union of all black-key polygons. Each white
    # key's region is its full vertical column MINUS this mask, so the
    # white region only includes pixels that are actually visible
    # white-key surface (above the red line: the gaps between black
    # keys; below the red line: the full unobstructed strip).
    black_mask = np.zeros((h, w), dtype=np.uint8)
    for k in keys:  # only black keys appended so far
        if k["type"] != "black":
            continue
        poly_arr = np.array(k["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(black_mask, [poly_arr], -1, 1, thickness=-1)

    # Assign white-key labels SEQUENTIALLY left-to-right (each label
    # used at most once), not by nearest-neighbor x — nearest-neighbor
    # produced duplicates when detected-seam centers drifted off the
    # SWSSW-pattern label positions by even a few pixels.
    wl_sorted = sorted(wl, key=lambda lab: lab[0])
    white_label_idx = 0
    for i in range(len(seam_xs) - 1):
        x_lo, x_hi = seam_xs[i], seam_xs[i + 1]
        if x_hi - x_lo < 3:
            continue
        if white_label_idx < len(wl_sorted):
            note = wl_sorted[white_label_idx][1]
            white_label_idx += 1
        else:
            note = "?"

        # Full-height column rectangle, then subtract the black-key
        # mask: only visible white-key pixels remain. Find the contour
        # of the resulting region and save that as the polygon.
        col_mask = np.zeros((h, w), dtype=np.uint8)
        col_mask[0:h, x_lo:x_hi] = 1
        white_only = col_mask & (1 - black_mask)
        cs, _ = cv2.findContours(white_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        # Take the largest connected component (typically the lower
        # strip; upper-strip slivers between blacks are dropped if too
        # tiny but kept if they're substantial).
        c = max(cs, key=cv2.contourArea)
        eps = max(0.5, 0.002 * cv2.arcLength(c, True))
        white_poly = cv2.approxPolyDP(c, eps, True)

        bx_w, by_w, bw_w, bh_w = cv2.boundingRect(white_poly)
        # Safe bbox: lower strip below the red line, shrunk inward —
        # unobstructed by black keys, ideal for change detection.
        safe_y0 = y_black_bottom + 2
        safe_y1 = h - 2
        safe_h_total = max(1, safe_y1 - safe_y0)
        sx = x_lo + int(0.20 * (x_hi - x_lo))
        sy = safe_y0 + int(0.10 * safe_h_total)
        sw = max(1, int(0.60 * (x_hi - x_lo)))
        sh = max(1, int(0.50 * safe_h_total))
        baseline = float(gray[sy:sy + sh, sx:sx + sw].mean()) if sh > 0 and sw > 0 else 0.0
        keys.append({
            "note": note,
            "type": "white",
            "polygon": [[int(p[0][0]), int(p[0][1])] for p in white_poly],
            "bbox": [int(bx_w), int(by_w), int(bw_w), int(bh_w)],
            "source": "geometric",
            "confidence": SOURCE_CONFIDENCE["geometric"],
            "baseline_intensity": baseline,
            "safe_bbox": [int(sx), int(sy), int(sw), int(sh)],
        })

    return {
        "version": 1,
        "camera_id": camera_id,
        "far_side": far_side,
        "warp": {
            "corners_tl_tr_br_bl": corners_tl_tr_br_bl.tolist()
            if hasattr(corners_tl_tr_br_bl, "tolist")
            else list(corners_tl_tr_br_bl),
            "out_size": [int(w), int(h)],
        },
        "y_black_bottom": int(y_black_bottom),
        "keys": keys,
    }


def save_calibration(calib: dict, path: str | Path) -> None:
    """Write a calibration dict to JSON. Pretty-printed for git diffs."""
    Path(path).write_text(json.dumps(calib, indent=2))


def load_calibration(path: str | Path) -> dict:
    """Read a calibration dict from JSON. Returns the raw dict for
    callers who want to do their own thing; for runtime use, prefer
    ``Calibration.load`` which also pre-rasterizes safe-region masks.
    """
    return json.loads(Path(path).read_text())


# ── Runtime-ready loader (pre-rasterized masks + warp matrix) ─────────────


@dataclass
class RuntimeKey:
    """One key, ready for live use. ``safe_mask`` is a (h, w) boolean
    array marking the safe-region pixels — index ``gray[safe_mask]``
    directly to get the pixels, no per-frame contour ops needed.
    """
    note: str
    type: str
    source: str
    confidence: float
    baseline_intensity: float
    bbox: tuple[int, int, int, int]
    safe_bbox: tuple[int, int, int, int]
    polygon: np.ndarray             # (N, 1, 2) int32, in warp coords
    safe_mask: np.ndarray           # (h, w) bool, in warp coords


@dataclass
class Calibration:
    """Runtime-ready calibration. Pre-computes everything the live
    consumer needs so per-frame work stays minimal.
    """

    far_side: str
    camera_id: str
    warp_corners: np.ndarray        # (4, 2) float32, src-image coords
    warp_size: tuple[int, int]      # (w, h)
    y_black_bottom: int
    keys: list[RuntimeKey] = field(default_factory=list)
    _M: np.ndarray | None = None    # cv2 perspective matrix, src→warp

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        w, h = d["warp"]["out_size"]
        keys: list[RuntimeKey] = []
        for k in d["keys"]:
            sx, sy, sw, sh = k["safe_bbox"]
            sm = np.zeros((h, w), dtype=np.bool_)
            sm[sy:sy + sh, sx:sx + sw] = True
            keys.append(RuntimeKey(
                note=k["note"],
                type=k["type"],
                source=k["source"],
                confidence=float(k["confidence"]),
                baseline_intensity=float(k["baseline_intensity"]),
                bbox=tuple(k["bbox"]),
                safe_bbox=tuple(k["safe_bbox"]),
                polygon=np.array(k["polygon"], dtype=np.int32).reshape(-1, 1, 2),
                safe_mask=sm,
            ))
        corners = np.array(d["warp"]["corners_tl_tr_br_bl"], dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        return cls(
            far_side=d["far_side"],
            camera_id=d.get("camera_id", ""),
            warp_corners=corners,
            warp_size=(int(w), int(h)),
            y_black_bottom=int(d["y_black_bottom"]),
            keys=keys,
            _M=cv2.getPerspectiveTransform(corners, dst),
        )

    def warp(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply the stored perspective warp to a source-image frame.
        Output shape is the calibration's ``warp_size``.
        """
        return cv2.warpPerspective(frame_bgr, self._M, self.warp_size)
