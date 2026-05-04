"""main.py — live piano transcription pipeline.

Controls:
    SPACE → calibrate from current frame, begin detection and transcription
    ESC   → save MIDI and exit
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from auto_calibrate import calibrate_frame
from detection import Detector
from key_labeler import draw_labels_tight_crop
from record import _build_transcribe_lut, draw_overlay_with_pressed, open_streams, overlay_from_dict
from seg_to_keys import warp_to_piano
from transcribe import Transcriber

DEBUG = False


def open_stream():
    """Open and start the first available Canon camera stream."""
    streams = open_streams(allow_iphone=False)
    if not streams:
        raise RuntimeError("no camera found")
    streams[0].start()
    return streams[0]


def run_preview_loop(stream):
    """Warp and label each frame until SPACE (calibrate) or ESC (quit).

    Returns (keys_dict, calib_warped) on SPACE, or (None, None) on ESC.
    """
    cv2.namedWindow("piano", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            continue

        warped, warp_trans, _ = warp_to_piano(frame, debug=False)
        if warp_trans is not None and warped.size > 0:
            cv2.imshow("piano", draw_labels_tight_crop(warped))
        else:
            cv2.imshow("piano", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None, None
        if key == ord(" "):
            result = calibrate_frame(frame)
            if result is None:
                print("auto-calibration failed — try repositioning the camera")
                continue
            keys_dict, calib_warped, _ = result
            print(f"calibrated: {len(keys_dict['keys'])} keys")
            return keys_dict, calib_warped


def build_detector(keys_dict, rest_warped_bgr):
    """Create a diff-mode Detector with the calibration frame as rest baseline."""
    detector = Detector(keys_dict, use_diff_only=True, use_mediapipe=True)
    rest_gray = cv2.cvtColor(rest_warped_bgr, cv2.COLOR_BGR2GRAY)
    detector.set_rest_mean_frame(rest_gray, rest_warped_bgr)
    return detector


def run_detection_loop(stream, detector, transcriber, key_lut, overlay):
    """Detect keypresses each frame, update transcriber, display results. ESC to exit."""
    M = detector.det_state["M"]
    W, H = detector.det_state["W"], detector.det_state["H"]
    polys, sbb, types = overlay

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            continue

        warped = cv2.warpPerspective(frame, M, (W, H))
        detector.set_source_frame(frame)
        pressed, _ = detector.process(warped)

        keys_now = [
            key_lut[ki] for ki in pressed
            if ki < len(key_lut) and key_lut[ki] is not None
        ]
        transcriber.update(keys_now)

        cv2.imshow("piano", draw_overlay_with_pressed(frame, polys, sbb, types, pressed))

        if DEBUG:
            raw = detector._last_diff_raw_mask
            counted = detector._last_diff_counted_mask
            if raw is not None and counted is not None:
                cv2.imshow("diff_debug", np.vstack([raw, counted]))

        if cv2.waitKey(1) & 0xFF == 27:
            break


def save_midi(transcriber):
    """Save the transcription as a MIDI file in midi_outs/."""
    midi_dir = Path("midi_outs")
    midi_dir.mkdir(exist_ok=True)
    midi_path = midi_dir / time.strftime("%Y%m%d_%H%M%S.mid")
    transcriber.save_midi(str(midi_path))
    print(f"saved MIDI → {midi_path}")


def main():
    stream = open_stream()
    try:
        keys_dict, calib_warped = run_preview_loop(stream)
        if keys_dict is None:
            return

        detector = build_detector(keys_dict, calib_warped)
        transcriber = Transcriber(fps=30.0)
        key_lut = _build_transcribe_lut(keys_dict)
        overlay = overlay_from_dict(keys_dict)

        run_detection_loop(stream, detector, transcriber, key_lut, overlay)
        save_midi(transcriber)
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
