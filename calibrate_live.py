"""Capture one Canon frame and generate a _keys.json calibration file.

Run once whenever the camera is repositioned. The output JSON is what
live_press.py consumes for warp + per-key press detection.

Usage
-----
    uv run python calibrate_live.py
    uv run python calibrate_live.py --out my_setup_keys.json
    uv run python calibrate_live.py --far-side left   # camera on right side
    uv run python calibrate_live.py --no-preview       # skip confirmation window

Controls (preview window)
-------------------------
  ENTER / y   save and exit
  r           retake (capture another frame)
  ESC / q     abort without saving
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from core.calibration import build_calibration_data, save_calibration
from core.key_labeler import draw_labels_tight_crop
from core.seg_to_keys import warp_to_piano
from core.stream_webcams import open_canon_streams

WARMUP_SECS = 2


def _capture_frame() -> np.ndarray:
    """Open the first Canon stream, warm up, and return one frame."""
    streams = open_canon_streams(silent=False)
    if not streams:
        raise RuntimeError("No Canon camera found. Check USB connection and webcam mode.")
    stream = streams[0]
    stream.start()
    try:
        time.sleep(WARMUP_SECS)
        ok, frame = stream.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read a frame from the Canon stream.")
        return frame
    finally:
        stream.stop()


def _try_warp(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Run warp_to_piano; return (warped, corners) or (None, None) on failure."""
    try:
        warped, _, corners = warp_to_piano(frame)
    except Exception as e:
        print(f"[calibrate_live] warp error: {e}")
        return None, None
    if warped is None or corners is None:
        return None, None
    # warp_to_piano returns np.zeros_like(frame) on detection failure, not None
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 5:
        print("[calibrate_live] warp returned a blank image — keyboard not detected.")
        return None, None
    return warped, corners


def _show_preview(frame: np.ndarray, labeled: np.ndarray) -> str:
    """Show side-by-side preview. Returns 'save', 'retake', or 'abort'."""
    def fit(img: np.ndarray, h: int = 400) -> np.ndarray:
        ih, iw = img.shape[:2]
        s = h / ih
        return cv2.resize(img, (max(1, int(iw * s)), h))

    display = np.hstack([fit(frame), fit(labeled)])
    for wt, c in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
        cv2.putText(display, "ENTER/y=save   r=retake   ESC/q=abort",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, wt, cv2.LINE_AA)
    win = "calibrate_live - confirm"
    cv2.imshow(win, display)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    print("[calibrate_live] Preview open — CLICK THE WINDOW to give it focus, then:")
    print("  y / ENTER = save    r = retake    ESC / q = abort")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, ord("y")):   # ENTER or y
            cv2.destroyAllWindows()
            return "save"
        elif key == ord("r"):
            cv2.destroyAllWindows()
            return "retake"
        elif key in (27, ord("q")):  # ESC or q
            cv2.destroyAllWindows()
            return "abort"


def run(out_path: str, far_side: str, no_preview: bool) -> None:
    while True:
        print("[calibrate_live] capturing frame from Canon...")
        frame = _capture_frame()
        print("[calibrate_live] warping to keyboard...")
        warped, corners = _try_warp(frame)

        if warped is None:
            print("[calibrate_live] ERROR: warp failed — keyboard not detected.")
            print("  Tips: ensure the full keyboard is visible and well-lit, then retry.")
            sys.exit(1)

        labeled = draw_labels_tight_crop(warped, far_side=far_side)

        if not no_preview:
            action = _show_preview(frame, labeled)
            if action == "retake":
                print("[calibrate_live] retaking...")
                continue
            elif action == "abort":
                print("[calibrate_live] aborted — no file written.")
                sys.exit(0)

        # Build and save
        print("[calibrate_live] building calibration data...")
        calib = build_calibration_data(warped, corners,
                                       far_side=far_side, camera_id="canon")
        n_black = sum(1 for k in calib["keys"] if k["type"] == "black")
        n_white = sum(1 for k in calib["keys"] if k["type"] == "white")
        save_calibration(calib, out_path)
        print(f"[calibrate_live] saved {out_path}")
        print(f"  {len(calib['keys'])} keys: {n_black} black, {n_white} white")
        print(f"  warp size: {calib['warp']['out_size']}")
        print(f"\nNext step:")
        print(f"  uv run python live_press.py --calib {out_path}")
        break


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate _keys.json from live Canon camera")
    p.add_argument("--out",        default="canon_calib_keys.json",
                   help="output path for the calibration JSON (default: canon_calib_keys.json)")
    p.add_argument("--far-side",   default="right", choices=["right", "left"],
                   help="which side of the keyboard is camera-far (default: right)")
    p.add_argument("--no-preview", action="store_true",
                   help="skip the confirmation window and save immediately")
    args = p.parse_args()
    run(args.out, args.far_side, args.no_preview)
