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

    # White-key regions: strip below y_black_bottom between adjacent
    # detected seams. Re-derive seams here so the saved white slots
    # match the labeler's rendering.
    band_top = y_black_bottom + int(0.15 * (h - y_black_bottom))
    white_band = gray[band_top:h - 2, :] if h - 2 > band_top else gray[band_top:, :]
    seam_xs: list[int] = []
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
                if seam_xs and i - seam_xs[-1] < min_sep:
                    if sm[i] > sm[seam_xs[-1]]:
                        seam_xs[-1] = i
                else:
                    seam_xs.append(i)
        seam_xs = [0] + sorted(seam_xs) + [w - 1]
    else:
        seam_xs = [0, w - 1]

    wl_by_x = sorted(wl, key=lambda lab: lab[0])
    for i in range(len(seam_xs) - 1):
        x_lo, x_hi = seam_xs[i], seam_xs[i + 1]
        if x_hi - x_lo < 3:
            continue
        cx_w = (x_lo + x_hi) // 2
        if not wl_by_x:
            note = "?"
        else:
            note = min(wl_by_x, key=lambda lab: abs(lab[0] - cx_w))[1]
        ry, rh = y_black_bottom + 2, h - 2 - y_black_bottom
        rw, rx = x_hi - x_lo, x_lo
        poly = np.array([
            [[rx, ry]], [[rx + rw, ry]],
            [[rx + rw, ry + rh]], [[rx, ry + rh]],
        ], dtype=np.int32)
        sx = rx + int(0.20 * rw)
        sy = ry + int(0.10 * rh)
        sw = max(1, int(0.60 * rw))
        sh = max(1, int(0.50 * rh))
        baseline = float(gray[sy:sy + sh, sx:sx + sw].mean()) if sh > 0 and sw > 0 else 0.0
        keys.append({
            "note": note,
            "type": "white",
            "polygon": [[int(p[0][0]), int(p[0][1])] for p in poly],
            "bbox": [int(rx), int(ry), int(rw), int(rh)],
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
