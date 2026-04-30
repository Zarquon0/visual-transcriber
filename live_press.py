"""Live piano key-press detector using the Calibration pipeline.

Loads a pre-saved _keys.json calibration file, applies the stored
perspective warp to every camera frame, and detects key presses by
comparing per-key safe-mask mean intensities against a quiet-keyboard
baseline collected at startup.

Usage
-----
Canon camera with saved calibration:
    uv run python live_press.py --calib piano_photos/live_1776972098_keys.json

Fallback to any webcam (index 0):
    uv run python live_press.py --calib piano_photos/live_1776972098_keys.json --cam 0

Explicit baseline length:
    uv run python live_press.py --calib ... --baseline 90

Controls
--------
  ESC / q   quit
  r         reset detector and rebuild baseline
  s         save current warped frame to disk
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from core.calibration import Calibration
from core.press_detector import NoteEvent, PressDetector
from core.stream_webcams import open_canon_streams

N_BASELINE_FRAMES = 60  # ~2 s at 30 FPS — keep keys up during this window


def _status(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    for wt, c in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
        cv2.putText(frame, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, wt, cv2.LINE_AA)


def _side_by_side(left: np.ndarray, right: np.ndarray, h: int = 360) -> np.ndarray:
    def fit(img: np.ndarray) -> np.ndarray:
        ih, iw = img.shape[:2]
        s = h / ih
        return cv2.resize(img, (max(1, int(iw * s)), h))
    return np.hstack([fit(left), fit(right)])


def run_live(calib_path: str, cam_index: int | None, n_baseline: int) -> None:
    rt = Calibration.load(calib_path)
    print(f"[live_press] loaded {len(rt.keys)} keys from {calib_path}")

    is_canon = False
    if cam_index is None:
        try:
            stream = open_canon_streams(silent=True)[0]
            stream.start()
            is_canon = True
        except (RuntimeError, IndexError):
            pass
    if not is_canon:
        stream = cv2.VideoCapture(cam_index or 0)
        if not stream.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_index or 0}")

    print(f"[live_press] baseline={n_baseline} frames  —  ESC/q quit  r reset  s save")

    detector:     PressDetector | None = None
    baseline_buf: list[np.ndarray]     = []
    last_warped:  np.ndarray | None    = None
    save_n = 0

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            break

        warped = rt.warp(frame)
        if warped is not None and warped.size > 0:
            last_warped = warped

        right = last_warped.copy() if last_warped is not None else np.zeros_like(frame)

        if warped is not None and warped.size > 0:
            if detector is None:
                baseline_buf.append(warped.copy())
                if len(baseline_buf) >= n_baseline:
                    detector     = PressDetector(rt, baseline_buf)
                    baseline_buf = []
                    print(f"[live_press] baseline ready — {len(rt.keys)} keys tracked")
                status = f"building baseline {len(baseline_buf)}/{n_baseline} (keep keys up)"
            else:
                events = detector.update(warped)
                for ev in events:
                    print(f"[{ev.event.upper():7s}] {ev.note:<5}  t={ev.time:.3f}")
                right  = detector.draw_overlay(right)
                active = detector.active_notes()
                status = (f"pressed: {' '.join(active)}" if active else "no keys pressed")
        else:
            status = "warp failed — is the calibration JSON correct for this camera?"

        display = _side_by_side(frame, right)
        _status(display, status)
        cv2.imshow("live_press", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            detector     = None
            baseline_buf = []
            print("[live_press] reset — rebuilding baseline")
        elif key == ord("s") and last_warped is not None:
            out_path = f"live_press_{save_n:04d}.png"
            cv2.imwrite(out_path, last_warped)
            print(f"[live_press] saved {out_path}")
            save_n += 1

    (stream.stop if is_canon else stream.release)()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live piano key-press detector")
    p.add_argument("--calib",    required=True,
                   help="Path to a _keys.json calibration file")
    p.add_argument("--cam",      type=int, default=None,
                   help="OpenCV camera index (default: auto-detect Canon)")
    p.add_argument("--baseline", type=int, default=N_BASELINE_FRAMES,
                   help=f"frames for quiet-keyboard baseline (default {N_BASELINE_FRAMES} ≈ 2 s)")
    args = p.parse_args()
    run_live(args.calib, args.cam, args.baseline)
