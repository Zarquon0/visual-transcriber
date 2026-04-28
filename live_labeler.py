"""Live and static piano key labeling.

Still image: identical to key_labeler._demo_image                                                                                                                           

Live stream: two modes controlled by --mode:
  bbox  — fast axis-aligned crop (find_keyboard_bbox).
  auto  — RANSAC corners + perspective correction; corners cached every
          CORNER_REFRESH frames so re-detection doesn't tank FPS.

Usage
-----
For the still image, run it with --cam 0 with --live, to skip the canon autodetect and go straight
to the OpenCV fallback.
  uv run python live_labeler.py --live --cam 0
Can also point it at a still image
  uv run python live_labeler.py test_piano.avif  

  ctrl+c to quit lol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

#from auto_calibrate import find_corners_auto, tighten_corners_to_tops, warp_from_corners
from seg_to_keys import warp_to_piano
from key_labeler import draw_labels_tight_crop, load_image#, warp_to_bbox, find_keyboard_bbox
from stream_webcams import open_canon_streams

CORNER_REFRESH = 45


def _corner_overlay(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    pts = corners.astype(int)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for (x, y), lbl in zip(pts, ["TL", "TR", "BR", "BL"]):
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(vis, lbl, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def _status(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    for w, c in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
        cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, w, cv2.LINE_AA)


def _side_by_side(left: np.ndarray, right: np.ndarray, h: int = 360) -> np.ndarray:
    def fit(img):
        ih, iw = img.shape[:2]
        s = h / ih
        return cv2.resize(img, (max(1, int(iw * s)), h))
    return np.hstack([fit(left), fit(right)])


# ── Still image ───────────────────────────────────────────────────────────────

def run_image(path: str) -> None:
    frame = load_image(path)
    # bbox = find_keyboard_bbox(frame)
    # if bbox is None:
    #     print("could not locate keyboard")
    #     return
    # labeled = draw_labels_tight_crop(warp_to_bbox(frame, bbox))
    labeled = draw_labels_tight_crop(warp_to_piano(frame))
    out = Path(path).with_suffix("").as_posix() + "_labeled.png"
    cv2.imwrite(out, labeled)
    print(f"wrote {out}")


# ── Live stream ───────────────────────────────────────────────────────────────

def run_live(cam_index: int | None, mode: str) -> None:
    is_canon = False
    if cam_index is None:
        try:
            stream = open_canon_streams(silent=True)[0]
            stream.start()
            is_canon = True
        except RuntimeError:
            pass
    if not is_canon:
        stream = cv2.VideoCapture(cam_index or 0)
        if not stream.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_index or 0}")

    print(f"[live_labeler] mode={mode}  —  ESC/q quit  r re-detect  s save")
    cached_corners: np.ndarray | None = None
    last_labeled: np.ndarray | None = None
    frame_count = save_n = 0

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            break

        redetect = frame_count % CORNER_REFRESH == 0

        # if mode == "bbox":
        #     bbox = find_keyboard_bbox(frame)
        #     labeled = draw_labels_tight_crop(warp_to_bbox(frame, bbox)) if bbox else None
        #     overlay = frame
        # else:
        #     if redetect or cached_corners is None:
        #         cached_corners = find_corners_auto(frame)
        #     if cached_corners is not None:
        #         try:
        #             tight = tighten_corners_to_tops(frame, cached_corners)
        #             labeled = draw_labels_tight_crop(warp_from_corners(frame, tight))
        #         except Exception as e:
        #             labeled = None
        #     else:
        #         labeled = None
        #     overlay = _corner_overlay(frame, cached_corners) if cached_corners is not None else frame
        try:
            if redetect or cached_corners is None or labeled is None:
                warped, _trans, corners = warp_to_piano(frame)
                cached_corners = corners
                labeled = draw_labels_tight_crop(warped)
        except Exception as e:
            cached_corners = None
            labeled = None

        if labeled is not None:
            last_labeled = labeled
        overlay = _corner_overlay(frame, cached_corners) if cached_corners is not None else frame

        right = last_labeled if last_labeled is not None else np.zeros_like(frame)
        display = _side_by_side(overlay, right)
        _status(display, f"mode={mode} frame={frame_count} {'re-detect' if redetect else 'cached'}"
                         + ("  [no keyboard]" if labeled is None else ""))
        cv2.imshow("live_labeler", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            cached_corners, frame_count = None, -1
        elif key == ord("s") and last_labeled is not None:
            out = f"live_labeled_{save_n:04d}.png"
            cv2.imwrite(out, last_labeled)
            print(f"[live_labeler] saved {out}")
            save_n += 1

        frame_count += 1

    (stream.stop if is_canon else stream.release)()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", help="path to a still image")
    p.add_argument("--live", action="store_true")
    p.add_argument("--cam", type=int, default=None, help="camera index (default: auto-detect)")
    p.add_argument("--mode", choices=["bbox", "auto"], default="auto",
                   help="live only: bbox = fast crop, auto = RANSAC corners (default)")
    args = p.parse_args()

    if args.live or args.image is None:
        run_live(args.cam, args.mode)
    else:
        run_image(args.image)
