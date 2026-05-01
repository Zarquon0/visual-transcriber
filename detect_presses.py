"""Minimal key press detector — fixed warp from calibration JSON.

The warp matrix is loaded once from the _keys.json file and applied with
cv2.warpPerspective every frame — no per-frame corner re-detection, so
the crop is pixel-stable. That's what lets the baseline comparison work.

Usage
-----
    uv run python detect_presses.py --calib cam0_keys.json
    uv run python detect_presses.py --calib cam0_keys.json --cam 1

Controls
--------
  ESC / q   quit
  r         restart baseline collection
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from core.calibration import Calibration
from core.press_detector import PressDetector

N_BASELINE = 60  # ~2 s at 30 fps — keep keys unpressed


def _overlay_status(img: np.ndarray, text: str) -> None:
    h = img.shape[0]
    for thickness, color in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
        cv2.putText(img, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness, cv2.LINE_AA)


def run(calib_path: str, cam_index: int, n_baseline: int) -> None:
    calib = Calibration.load(calib_path)
    print(f"[detect_presses] {len(calib.keys)} keys loaded from {calib_path}")
    print(f"[detect_presses] warp size: {calib.warp_size}")
    print(f"[detect_presses] camera: {cam_index}")
    print(f"[detect_presses] ESC/q quit   r reset baseline")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index}")

    baseline_buf: list[np.ndarray] = []
    detector: PressDetector | None = None

    def reset() -> None:
        nonlocal baseline_buf, detector
        baseline_buf = []
        detector = None
        print("[detect_presses] reset — collecting new baseline")

    print(f"[detect_presses] collecting baseline ({n_baseline} frames) — keep all keys up")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # Fixed warp — the key stability fix
        warped = calib.warp(frame)

        if detector is None:
            # Baseline collection phase
            if warped is not None and warped.size > 0:
                baseline_buf.append(warped.copy())

            n = len(baseline_buf)
            if n >= n_baseline:
                detector = PressDetector(calib, baseline_buf)
                baseline_buf = []
                print(f"[detect_presses] baseline ready — detecting presses")

            status = f"collecting baseline {n}/{n_baseline} — keep keys up"
            disp = warped if warped is not None else np.zeros((*calib.warp_size[::-1], 3), np.uint8)

        else:
            # Detection phase
            if warped is not None and warped.size > 0:
                events = detector.update(warped)
                for ev in events:
                    tag = "PRESS  " if ev.event == "press" else "release"
                    print(f"  [{tag}] {ev.note:<6}  score={ev.score:.2f}  thr={ev.threshold:.2f}")

            active = detector.active_notes()
            status = f"pressed: {' '.join(sorted(active))}" if active else "no keys pressed"

            disp = detector.draw_overlay(warped) if warped is not None else np.zeros(
                (*calib.warp_size[::-1], 3), np.uint8
            )

        _overlay_status(disp, status)
        cv2.imshow("detect_presses — warped", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Minimal key press detector")
    p.add_argument("--calib", required=True,
                   help="path to _keys.json produced by calibrate_live.py")
    p.add_argument("--cam", type=int, default=0,
                   help="OpenCV camera index (default: 0)")
    p.add_argument("--baseline", type=int, default=N_BASELINE,
                   help=f"frames for quiet baseline (default {N_BASELINE})")
    args = p.parse_args()

    run(args.calib, args.cam, args.baseline)
