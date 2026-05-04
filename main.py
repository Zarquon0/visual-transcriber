"""main.py — live piano transcription pipeline.

Controls:
    SPACE → calibrate from current frame, begin detection and transcription
    c     → recalibrate from current frame (during detection)
    b     → recapture REST baseline from N quiet-keyboard frames
            (during detection, hands AWAY from keys)
    ESC   → save MIDI and exit

Three windows mirror record.py's live setup:
    ``piano``           — source camera view with HUD; press flashes red
                          during detection
    ``warp_cam0``       — calibration inspector (raw warp + colored-fill
                          segmentation), opened/refreshed on every
                          calibration
    ``warp_lines_cam0`` — 7-panel pipeline diagnostic view, shown during
                          detection (raw warp + presses → segmentation →
                          MP hand mask → MP extended warp → press diff
                          raw → counted (-hand -boundary) → per-key
                          activation + scores)

Tunable hyperparameters below mirror the snappy-onset settings we've been
running interactively in record.py. Adjust at the top of the file rather
than via CLI flags — this script is the simplified, production-shape
entry point; record.py keeps the dev-time hotkey tuning surface.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from auto_calibrate import calibrate_frame
from detection import Detector
from playback import _label_panel
from record import (
    _build_transcribe_lut,
    draw_overlay_with_pressed,
    draw_warp_colored,
    open_streams,
    overlay_from_dict,
)
from seg_to_keys import warp_to_piano
from stream_webcams import CanonStream, _load_config
from transcribe import Transcriber

DEBUG = False

# Camera index. Auto-detect via ``open_canon_streams`` has been
# unreliable on this machine — it consistently picks the laptop webcam
# instead of the Canon. Set this explicitly to the Canon's OpenCV index
# (0 in our setup). Set to ``None`` to fall back to auto-detect.
CAM_INDEX = 0

# Calibration: pixels of case-top trimmed from the warp output. Higher =
# more aggressive crop. Tune until polygons hug the actual key tops.
TOP_CROP = 10

# Detection thresholds. Smaller = more sensitive / more responsive but
# noisier. These match our tested live-record values.
SMOOTH_WINDOW = 1            # rolling mean over per-key counts (1 = no smoothing)
PRESS_PIXELS = 5             # activated pixels needed per key to fire
MIN_BLOB_AREA = 1            # CC area floor (1 disables the filter)
BOUNDARY_MARGIN = 1          # px erosion on every polygon for pixel→key assignment

# Multi-frame baseline (`b` during detection). Averages this many warped
# frames as the rest reference for ``press_diff.detect_press_regions``.
# Longer = better AE/AWB-settled mean at the cost of waiting.
BASELINE_FRAMES = 60


# ── HUD helpers ──────────────────────────────────────────────────────────

def _hud(img, lines, color=(255, 255, 255)):
    """Stamp left-aligned text lines onto the top-left of img (in-place)."""
    for i, t in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)


def open_stream():
    """Open and start the Canon camera stream.

    Uses an explicit OpenCV index (``CAM_INDEX``) by default to avoid
    auto-detect picking the wrong camera. Falls back to
    ``open_canon_streams`` if ``CAM_INDEX`` is None.

    Dual-cam direction (TODO):
        Replace the single stream with a list, or use
        ``stream_webcams.DualCanonStream`` for synced reads. Each
        downstream stage (preview, calibration, detection) then runs
        per-camera in parallel — see ``run_preview_loop`` /
        ``run_detection_loop`` for the per-cam touch points.
    """
    if CAM_INDEX is not None:
        s = CanonStream(CAM_INDEX, _load_config(), show_stats=False)
        if not s.cap.isOpened():
            raise RuntimeError(f"could not open camera at index {CAM_INDEX}")
        s.start()
        return s
    streams = open_streams(allow_iphone=False)
    if not streams:
        raise RuntimeError("no camera found")
    streams[0].start()
    return streams[0]


def calibrate_from_frame(frame):
    """Run auto_calibrate.calibrate_frame and pop the inspector window.

    Returns (keys_dict, calib_warped) on success, or (None, None) if
    auto-calibration fails on the given frame.
    """
    result = calibrate_frame(frame, top_crop=TOP_CROP)
    if result is None:
        print("auto-calibration failed — try repositioning the camera")
        return None, None
    keys_dict, calib_warped, _ = result
    print(f"calibrated: {len(keys_dict['keys'])} keys")
    colored = draw_warp_colored(calib_warped, keys_dict)
    cv2.imshow("warp_cam0", np.vstack([calib_warped, colored]))
    return keys_dict, calib_warped


def run_preview_loop(stream):
    """Show the live source camera frame until SPACE (calibrate) or ESC (quit).

    Returns (keys_dict, calib_warped) on SPACE, or (None, None) on ESC.

    Why no live labels: the labeling pipeline (Otsu → U-valley splits →
    SWSSW alignment → Sobel-X seams → text positioning) was designed to
    run **once** at calibration. Running it per frame produces visible
    flicker as polygons / labels jitter on noise. Preview shows the raw
    camera feed; SPACE triggers a single locked-in calibration.
    """
    cv2.namedWindow("piano", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            continue

        disp = frame.copy()
        _hud(disp, [
            "PREVIEW",
            "SPACE: calibrate (start detection)",
            "ESC:   quit",
        ])
        cv2.imshow("piano", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None, None
        if key == ord(" "):
            keys_dict, calib_warped = calibrate_from_frame(frame)
            if keys_dict is not None:
                return keys_dict, calib_warped


def build_detector(keys_dict, rest_warped_bgr):
    """Create a diff-mode Detector tuned for low onset latency.

    The single SPACE-press warped frame is the initial rest baseline,
    matching William's original ``press_diff`` design. Press ``b`` during
    detection to refine the baseline by averaging ``BASELINE_FRAMES``
    consecutive frames (better for cameras with unstable AE/AWB).
    """
    detector = Detector(
        keys_dict,
        use_diff_only=True,
        use_mediapipe=True,
        smooth_window=SMOOTH_WINDOW,
    )
    detector.set_diff_thresholds(PRESS_PIXELS, MIN_BLOB_AREA)
    detector.set_diff_boundary_margin(BOUNDARY_MARGIN)
    rest_gray = cv2.cvtColor(rest_warped_bgr, cv2.COLOR_BGR2GRAY)
    detector.set_rest_mean_frame(rest_gray, rest_warped_bgr)
    return detector


def _build_warp_lines_panels(detector, keys_dict, warped, pressed):
    """Construct the 7-panel pipeline diagnostic stack identical to
    record.py's ``warp_lines_cam0`` view (and playback.py's --diff)."""
    warp_with_press = warped.copy()
    for ki in pressed:
        try:
            poly = np.array(
                keys_dict["keys"][ki]["polygon"], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.drawContours(
                warp_with_press, [poly], -1, (0, 0, 255), 3, cv2.LINE_AA,
            )
        except Exception:
            pass

    panels = [
        _label_panel(warp_with_press, "1. RAW WARP + PRESSES"),
        _label_panel(draw_warp_colored(warped, keys_dict), "2. SEGMENTATION"),
        _label_panel(detector._last_hand_viz, "3. HAND MASK (MP)"),
    ]
    if detector._last_mp_ext_viz is not None:
        panels.append(_label_panel(
            detector._last_mp_ext_viz, "4. MP EXTENDED WARP"
        ))
    panels.append(_label_panel(
        detector._last_diff_raw_mask, "5. PRESS DIFF (raw mask)"
    ))
    panels.append(_label_panel(
        detector._last_diff_counted_mask,
        "6. COUNTED (-hand -boundary) <- press signal",
    ))
    panels.append(_label_panel(
        detector._last_diff_overlay,
        "7. PER-KEY ACTIVATION + SCORES",
    ))
    return np.vstack(panels)


def run_detection_loop(stream, detector, transcriber, key_lut, keys_dict):
    """Detect keypresses, update transcriber, support live recalibration.

    Returns one of:
        ``"quit"``        — ESC pressed, save MIDI and exit
        ``("recalibrate", new_frame)`` — c pressed, caller should rebuild
                                         keys_dict / detector / lut from
                                         the captured frame

    Hotkeys (piano window focused):
        c   → recalibrate from the current frame
        b   → recapture REST baseline over BASELINE_FRAMES quiet frames
        ESC → save MIDI and exit

    Dual-cam direction (TODO):
        Take ``streams``, ``detectors``, ``key_luts``, and ``keys_dicts``
        as parallel lists indexed by camera. Per-frame: read all cams
        (or DualCanonStream.read() for synced), warp each, run each
        Detector, fuse pressed sets. Recommended fusion: per-key primary
        camera — split the keyboard at midline, low-index keys belong to
        one cam, high-index to the other. Single Transcriber, called
        once per frame with the fused press set.
    """
    M = detector.det_state["M"]
    W, H = detector.det_state["W"], detector.det_state["H"]
    polys_src, sbb_src, types = overlay_from_dict(keys_dict)
    cv2.namedWindow("warp_lines_cam0", cv2.WINDOW_NORMAL)

    # Baseline-capture state: when active, accumulates warped means and
    # suppresses detection / transcribe updates.
    baseline_active = False
    bl_remaining = 0
    bl_gray_accum = None
    bl_bgr_accum = None
    bl_n = 0

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            continue

        warped = cv2.warpPerspective(frame, M, (W, H))

        if baseline_active:
            # Accumulate, don't run detection.
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            if bl_gray_accum is None:
                bl_gray_accum = np.zeros(warped_gray.shape, dtype=np.float64)
                bl_bgr_accum = np.zeros(warped.shape, dtype=np.float64)
            bl_gray_accum += warped_gray.astype(np.float64)
            bl_bgr_accum += warped.astype(np.float64)
            bl_n += 1
            bl_remaining -= 1
            if bl_remaining <= 0:
                mean_gray = (bl_gray_accum / bl_n).astype(np.uint8)
                mean_bgr = (bl_bgr_accum / bl_n).astype(np.uint8)
                detector.set_rest_mean_frame(mean_gray, mean_bgr)
                baseline_active = False
                bl_gray_accum = bl_bgr_accum = None
                bl_n = 0
                print(f"REST baseline updated ({BASELINE_FRAMES} frames)")
            pressed = set()
        else:
            detector.set_source_frame(frame)
            pressed, _ = detector.process(warped)
            keys_now = [
                key_lut[ki] for ki in pressed
                if ki < len(key_lut) and key_lut[ki] is not None
            ]
            transcriber.update(keys_now)

        # Source view + HUD.
        disp = draw_overlay_with_pressed(frame, polys_src, sbb_src, types, pressed)
        if baseline_active:
            _hud(disp, [
                f"REST CAPTURE: keep hands AWAY ({bl_remaining} frames left)",
                "ESC: save MIDI & quit",
            ], color=(0, 200, 200))
        else:
            _hud(disp, [
                f"DETECTING — pressed: {sorted(pressed) if pressed else 'none'}",
                "c: recalibrate   b: recapture REST baseline   ESC: save MIDI & quit",
            ])
        cv2.imshow("piano", disp)
        cv2.imshow("warp_lines_cam0",
                   _build_warp_lines_panels(detector, keys_dict, warped, pressed))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return "quit"
        if key == ord("c") and not baseline_active:
            return ("recalibrate", frame)
        if key == ord("b") and not baseline_active:
            baseline_active = True
            bl_remaining = BASELINE_FRAMES
            print(f"REST CAPTURE started — keep hands AWAY for {BASELINE_FRAMES} frames")


def save_midi(transcriber):
    """Save the transcription as a MIDI file in midi_outs/."""
    midi_dir = Path("midi_outs")
    midi_dir.mkdir(exist_ok=True)
    midi_path = midi_dir / time.strftime("%Y%m%d_%H%M%S.mid")
    transcriber.save_midi(str(midi_path))
    print(f"saved MIDI → {midi_path}")


def main():
    """Entry: open camera → preview → SPACE calibrates → detection loop.

    Detection loop supports ``c`` to recalibrate without leaving the
    session (jumps back to a fresh calibration on the captured frame,
    rebuilds Detector / LUT, resumes detection). Transcriber persists
    across recalibrations so MIDI accumulates the whole session.

    Dual-cam direction (TODO):
        Hold a list of ``(stream, detector, key_lut, keys_dict)`` tuples,
        one per camera. ``run_preview_loop`` becomes per-cam preview
        windows (or a tiled view); SPACE calibrates all cams from the
        same instant. ``run_detection_loop`` reads all cams per frame,
        runs each Detector, fuses pressed sets, then drives a single
        shared Transcriber.
    """
    stream = open_stream()
    transcriber = None
    try:
        keys_dict, calib_warped = run_preview_loop(stream)
        if keys_dict is None:
            return

        transcriber = Transcriber(fps=30.0)
        detector = build_detector(keys_dict, calib_warped)
        key_lut = _build_transcribe_lut(keys_dict)

        while True:
            outcome = run_detection_loop(
                stream, detector, transcriber, key_lut, keys_dict,
            )
            if outcome == "quit":
                break
            if isinstance(outcome, tuple) and outcome[0] == "recalibrate":
                _, frame = outcome
                new_keys, new_warped = calibrate_from_frame(frame)
                if new_keys is None:
                    continue   # stay with old calibration if auto-cal fails
                keys_dict = new_keys
                detector = build_detector(keys_dict, new_warped)
                key_lut = _build_transcribe_lut(keys_dict)
    finally:
        if transcriber is not None:
            try:
                save_midi(transcriber)
            except Exception as e:
                print(f"save_midi failed: {e}")
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
