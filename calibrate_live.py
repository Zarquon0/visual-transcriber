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
from core.warp_calibration import auto_warp_to_key_tops
from pathlib import Path

WARMUP_SECS = 2


def _capture_frame(stream_index: int = 0) -> np.ndarray:
    """Open one selected Canon stream, warm up, and return one frame.

    Important:
    Do NOT call open_canon_streams() here, because that opens every Canon
    camera even when we only want one. On macOS, opening both cameras can
    cause AVFoundation/OpenCV warnings or hangs.
    """
    from core.stream_webcams import find_canon_indices, _load_config

    print(f"[calibrate_live] locating Canon stream {stream_index}...")

    indices = find_canon_indices()
    if not indices:
        raise RuntimeError("No Canon camera found. Check USB connection and webcam mode.")

    if stream_index < 0 or stream_index >= len(indices):
        raise RuntimeError(
            f"Requested --stream {stream_index}, but only {len(indices)} Canon camera(s) "
            f"were detected. Available stream numbers: 0..{len(indices) - 1}"
        )

    cv_index = indices[stream_index]
    cfg = _load_config()

    width = int(cfg.get("resolution", {}).get("width", 1280))
    height = int(cfg.get("resolution", {}).get("height", 720))
    fps = cfg.get("fps")

    rotate_values = cfg.get("camera_rotate", [])
    rotate = 0
    if isinstance(rotate_values, list) and stream_index < len(rotate_values):
        rotate = int(rotate_values[stream_index] or 0)

    rotate_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotate_code = rotate_codes.get(rotate % 360)

    print(
        f"[calibrate_live] opening stream {stream_index} "
        f"(OpenCV index {cv_index}, {width}x{height}, rotate={rotate})..."
    )

    cap = cv2.VideoCapture(cv_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open Canon camera at OpenCV index {cv_index}.")

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up and drain a few frames.
        deadline = time.time() + WARMUP_SECS
        frame = None
        ok = False

        while time.time() < deadline:
            ok, frame = cap.read()
            time.sleep(0.03)

        # Try a few final reads after warmup.
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)

        if not ok or frame is None:
            raise RuntimeError(
                f"Opened camera {stream_index}, but failed to read a valid frame."
            )

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        print(f"[calibrate_live] captured frame: {frame.shape[1]}x{frame.shape[0]}")
        return frame

    finally:
        cap.release()


def _try_warp(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Run the repo's original live keyboard warp.

    This intentionally uses seg_to_keys.warp_to_piano, not the experimental
    auto-calibration helper.
    """
    try:
        result = warp_to_piano(frame)
    except Exception as e:
        print(f"[calibrate_live] warp_to_piano error: {e}")
        return None, None

    # Current repo version appears to return:
    #     warped, debug_or_mask, corners
    # but this keeps it tolerant if that changes slightly.
    if isinstance(result, tuple):
        if len(result) >= 3:
            warped = result[0]
            corners = result[2]
        elif len(result) == 2:
            warped, corners = result
        else:
            warped = result[0]
            corners = None
    else:
        warped = result
        corners = None

    if warped is None or warped.size == 0:
        print("[calibrate_live] warp_to_piano returned empty output.")
        return None, None

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 5:
        print("[calibrate_live] warp_to_piano returned a blank image.")
        return None, None

    if corners is None:
        print("[calibrate_live] warp_to_piano did not return corners.")
        return None, None

    return warped, np.asarray(corners, dtype=np.float32)

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


def run(out_path: str, far_side: str, no_preview: bool, stream_index: int = 0) -> None:
    while True:
        print(f"[calibrate_live] capturing frame from Canon stream {stream_index}...")
        frame = _capture_frame(stream_index)
        print("[calibrate_live] warping to keyboard...")
        cv2.imwrite("debug_calibrate_frame.jpg", frame)
        print("[calibrate_live] wrote debug_calibrate_frame.jpg")
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

        # Save matching source frame so validate_calibration.py can find it.
        out_json = Path(out_path)
        if out_json.name.endswith("_keys.json"):
            src_name = out_json.name.replace("_keys.json", ".jpg")
            src_path = out_json.with_name(src_name)
        else:
            src_path = out_json.with_suffix(".jpg")

        cv2.imwrite(str(src_path), frame)
        print(f"[calibrate_live] saved source frame {src_path}")
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
    p.add_argument("--stream", type=int, default=0,
                   help="which Canon stream to calibrate (0=first, 1=second; default: 0)")
    args = p.parse_args()
    run(args.out, args.far_side, args.no_preview, args.stream)
