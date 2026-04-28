"""
[STILL IN PROGRESS] key press detection - need to pass in a calibration json 
(generated with calibration.py) and to be used live.
"""


import argparse
import json
import time
from typing import Any, Dict, List

import cv2
import numpy as np

#from auto_calibrate import find_corners_auto, tighten_corners_to_tops, warp_from_corners
from calibration import Calibration
from key_labeler import draw_labels_tight_crop#, find_keyboard_bbox, warp_to_bbox
from live_labeler import CORNER_REFRESH, _corner_overlay, _side_by_side, _status
from seg_to_keys import warp_to_piano
from stream_webcams import open_canon_streams


def initialise(calibration_path: str) -> None:
    calibration = Calibration.load(calibration_path)

    states = {}
    for k in calibration.keys:
        states[k.note] = {
            "baseline": k.baseline_intensity,
            "is_pressed": False,
            "counter": 0,
        }

    return {
        "calibration": calibration,
        "states": states,
        "prev_grayscale": None,
        "events": [],
        "threshold": 10,
        "motion_threshold": 5,
        "debounce": 3,
    }

def process_frame(detector: Dict[str, Any], warped: np.ndarray) -> List[Dict[str, Any]]:
    grayscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)

    events_out = []
    prev_grayscale = detector["prev_grayscale"]

    for key in detector["calibration"].keys:
        note = key.note
        state = detector["states"][note]

        region = grayscale[key.safe_mask]
        if region.size == 0:
            continue

        intensity = float(np.mean(region))
        delta = intensity - state["baseline"]

        motion = 0.0
        if prev_grayscale is not None:
            prev_region = prev_grayscale[key.safe_mask]
            if prev_region.shape == region.shape:
                motion = float(np.mean(cv2.absdiff(region, prev_region)))

        uncertainty = 1 - key.confidence

        active = (
            abs(delta) > detector["threshold"] * (1 + uncertainty)
            or motion > detector["motion_threshold"] * (1 + uncertainty)
        )

        if active:
            state["counter"] += 1
        else:
            state["counter"] -= 1

        state["counter"] = int(np.clip(state["counter"], 0, detector["debounce"]))

        now = time.time()

        if not state["is_pressed"] and state["counter"] >= detector["debounce"]:
            state["is_pressed"] = True
            e = {"note": note, "event": "on", "time": now}
            detector["events"].append(e)
            events_out.append(e)

        if state["is_pressed"] and state["counter"] == 0:
            state["is_pressed"] = False
            e = {"note": note, "event": "off", "time": now}
            detector["events"].append(e)
            events_out.append(e)

        if not state["is_pressed"]:
            state["baseline"] = 0.995 * state["baseline"] + 0.005 * intensity
    
    detector["prev_grayscale"] = grayscale
    return events_out

def save_events(detector: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(detector["events"], f, indent=2)

def run_live(cam_index: int | None, mode: str, calibration_path: str) -> None:
    detector = initialise(calibration_path)

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

    print(f"[live_labeler]  —  ESC/q quit  r re-detect  s save")

    cached_corners = None
    last_labeled = None
    frame_count = save_n = 0

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            break

        redetect = frame_count % CORNER_REFRESH == 0
        warped = None

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
        overlay = _corner_overlay(frame, cached_corners) if cached_corners is not None else frame

        if labeled is not None:
            last_labeled = labeled

        if warped is not None:
            events = process_frame(detector, warped)
            for e in events:
                print(e["note"], e["event"])

        if labeled is not None:
            last_labeled = labeled

        right = last_labeled if last_labeled is not None else np.zeros_like(frame)
        display = _side_by_side(overlay, right)

        _status(display, f"frame={frame_count}")
        cv2.imshow("key_press", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            cached_corners, frame_count = None, -1
        elif key == ord("s"):
            out = f"session_{save_n}.json"
            save_events(detector, out)
            print(f"[saved {out}]")
            save_n += 1

        frame_count += 1

    save_events(detector, "session_final.json")
    print("[saved session_final.json]")

    (stream.stop if is_canon else stream.release)()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=None, help="camera index (default: auto-detect)")
    p.add_argument("--mode", choices=["bbox", "auto"], default="auto",
                   help="live only: bbox = fast crop, auto = RANSAC corners (default)")
    p.add_argument("--calibration", required=True)
    args = p.parse_args()

    run_live(args.cam, args.mode, args.calibration)