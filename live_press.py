"""Live piano key-press detector — single or dual camera.

With two calibration files, each camera runs its own PressDetector. The final
active note set is the union of the cameras, which is the correct behavior for
partial views where each camera only sees part of the keyboard.

Usage
-----
Single camera (original behaviour):
    uv run python live_press.py --calib cam0_keys.json

Dual camera:
    uv run python live_press.py --calib cam0_keys.json cam1_keys.json

Controls
--------
  ESC / q   quit
  r         reset all detectors and rebuild baselines
  s         save current warped frame(s) to disk
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from core.calibration import Calibration
from core.press_detector import NoteEvent, PressDetector
from core.stream_webcams import open_canon_streams
#from core.seg_to_keys import warp_to_piano

N_BASELINE_FRAMES = 60  # ~2 s at 30 FPS


# ── display helpers ───────────────────────────────────────────────────────────

def _status(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    for wt, c in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
        cv2.putText(frame, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, wt, cv2.LINE_AA)

def _top_scores(detectors: list[PressDetector | None], n: int = 5) -> str:
    rows = []
    for cam_i, det in enumerate(detectors):
        if det is None:
            continue
        for r in det.debug_rows():
            rows.append((r.score, cam_i, r.note))

    rows.sort(reverse=True, key=lambda x: x[0])
    if not rows:
        return "scores: none"

    return "scores: " + " ".join(
        f"cam{cam}:{note}={score:.2f}"
        for score, cam, note in rows[:n]
    )

def _fit(img: np.ndarray, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = h / max(1, ih)
    return cv2.resize(img, (max(1, int(iw * s)), h))


def _build_display(
    raws: list[np.ndarray],
    warpeds: list[np.ndarray | None],
    detectors: list[PressDetector | None],
    combined_active: set[str],
    panel_h: int = 360,
) -> np.ndarray:
    """Stack one column per camera: [raw | warped+overlay], then hstack all."""
    panels = []
    for i, (raw, warped, det) in enumerate(zip(raws, warpeds, detectors)):
        placeholder = np.zeros_like(raw) if raw is not None else np.zeros((panel_h, panel_h, 3), np.uint8)

        left = _fit(raw, panel_h) if raw is not None else _fit(placeholder, panel_h)

        if warped is not None and det is not None:
            right = _fit(det.draw_overlay(warped), panel_h)
        elif warped is not None:
            right = _fit(warped, panel_h)
        else:
            right = _fit(placeholder, panel_h)

        panels.append(np.hstack([left, right]))

    if not panels:
        return np.zeros((panel_h, panel_h * 2, 3), np.uint8)
    return np.hstack(panels)


# ── combined state tracker (for emitting clean press/release events) ──────────

class CombinedTracker:
    """Tracks union-combination of multiple detectors, emitting unified events.

    This is the correct behavior for partial camera views:
    - If only cam0 sees a key, cam0 may trigger it.
    - If only cam1 sees a key, cam1 may trigger it.
    - If both cameras see a key, either camera may trigger it.

    This favors recall over precision, which is what you want before you have
    calibrated per-camera reliability scores.
    """

    def __init__(self, n_detectors: int):
        self._n = n_detectors
        self._prev_active: set[str] = set()

    def update(self, detectors: list[PressDetector]) -> tuple[set[str], list[dict]]:
        """Return (combined_active_set, new_events_list)."""
        if not detectors:
            return set(), []

        sets = [set(d.active_notes()) for d in detectors]

        if self._n == 1:
            combined = sets[0]
        else:
            combined = set().union(*sets)

        events = []
        now = time.perf_counter()

        for note in combined - self._prev_active:
            events.append({"note": note, "event": "press", "time": now})

        for note in self._prev_active - combined:
            events.append({"note": note, "event": "release", "time": now})

        self._prev_active = combined
        return combined, events


# ── main loop ─────────────────────────────────────────────────────────────────

def run_live(
    calib_paths: list[str],
    cam_indices: list[int] | None,
    n_baseline: int,
) -> None:
    calibrations = [Calibration.load(p) for p in calib_paths]
    n_cams = len(calibrations)
    print(f"[live_press] {n_cams} calibration(s) loaded")
    for i, (p, rt) in enumerate(zip(calib_paths, calibrations)):
        print(f"  cam{i}: {len(rt.keys)} keys  ← {p}")

    # ── open streams ─────────────────────────────────────────────────────────
    streams = []
    is_canon = False

    if cam_indices is None:
        try:
            all_streams = open_canon_streams(silent=False)
            if len(all_streams) < n_cams:
                print(f"[live_press] WARNING: only {len(all_streams)} Canon camera(s) "
                      f"found for {n_cams} calibration(s). Using what's available.")
            streams = all_streams[:n_cams]
            for s in streams:
                s.start()
            is_canon = True
        except RuntimeError as e:
            print(f"[live_press] Canon auto-detect failed ({e}); falling back to OpenCV indices")

    if not is_canon:
        indices = cam_indices if cam_indices else list(range(n_cams))
        for idx in indices[:n_cams]:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {idx}")
            streams.append(cap)

    n_active = len(streams)
    calibrations = calibrations[:n_active]
    print(f"[live_press] {n_active} stream(s) opened")
    print(f"[live_press] baseline={n_baseline} frames — keep keys up during collection")
    print(f"[live_press] ESC/q quit  r reset  s save")

    # ── state ─────────────────────────────────────────────────────────────────
    detectors:    list[PressDetector | None] = [None] * n_active
    baseline_bufs: list[list[np.ndarray]]   = [[] for _ in range(n_active)]
    last_warpeds:  list[np.ndarray | None]  = [None] * n_active
    tracker = CombinedTracker(n_active)
    save_n = 0
    
    def reset_all():
        nonlocal detectors, baseline_bufs, tracker
        detectors      = [None] * n_active
        baseline_bufs  = [[] for _ in range(n_active)]
        tracker        = CombinedTracker(n_active)
        print("[live_press] reset — rebuilding baselines")

    # ── frame loop ────────────────────────────────────────────────────────────
    while True:
        raws:    list[np.ndarray | None] = []
        warpeds: list[np.ndarray | None] = []

        for i, (stream, rt) in enumerate(zip(streams, calibrations)):
            ok, frame = stream.read()
            raw = frame if (ok and frame is not None) else None

            warped = None
            if raw is not None:
                try:
                    warped = rt.warp(raw)
                    if warped is not None and warped.size > 0:
                        last_warpeds[i] = warped
                except Exception as e:
                    print(f"[live_press] cam{i} calibration warp failed: {e}")
                    warped = None

            raws.append(raw)
            warpeds.append(warped)

        # ── baseline collection ───────────────────────────────────────────────
        all_ready = all(d is not None for d in detectors)
        if not all_ready:
            for i, warped in enumerate(warpeds):
                if detectors[i] is not None:
                    continue
                if warped is not None and warped.size > 0:
                    baseline_bufs[i].append(warped.copy())
                if len(baseline_bufs[i]) >= n_baseline:
                    detectors[i] = PressDetector(calibrations[i], baseline_bufs[i])
                    baseline_bufs[i] = []
                    print(f"[live_press] cam{i} baseline ready ({len(calibrations[i].keys)} keys)")

            ready = sum(1 for d in detectors if d is not None)
            frame_counts = "/".join(
                "ready" if detectors[i] is not None else str(len(baseline_bufs[i]))
                for i in range(n_active)
            )
            status = (
                f"building baseline — cam(s) ready: {ready}/{n_active}  "
                f"frames: {frame_counts}/{n_baseline}"
            )
            combined_active: set[str] = set()
        else:
            # ── detection ────────────────────────────────────────────────────
            for i, (det, warped) in enumerate(zip(detectors, warpeds)):
                if det is not None and warped is not None and warped.size > 0:
                    det.update(warped)

            combined_active, events = tracker.update(detectors)
            for ev in events:
                tag = "PRESS  " if ev["event"] == "press" else "release"
                print(f"[{tag}] {ev['note']:<6}  t={ev['time']:.3f}")

            if combined_active:
                status = f"pressed: {' '.join(sorted(combined_active))}"
            elif n_active > 1:
                status = f"no keys pressed  (union of {n_active} cameras)"
            else:
                status = "no keys pressed"

        # ── display ──────────────────────────────────────────────────────────
        # For display: use last_warpeds so the panel stays non-black when warp fails.
        display_warpeds = [last_warpeds[i] if warpeds[i] is None else warpeds[i]
                           for i in range(n_active)]
        display_raws    = [np.zeros((360, 640, 3), np.uint8) if r is None else r
                           for r in raws]

        disp = _build_display(display_raws, display_warpeds, detectors, combined_active)
        score_text = _top_scores(detectors)
        _status(disp, f"{status}    {score_text}")
        cv2.imshow("live_press", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            reset_all()
        elif key == ord("s"):
            for i, w in enumerate(last_warpeds):
                if w is not None:
                    out_path = f"live_press_cam{i}_{save_n:04d}.png"
                    cv2.imwrite(out_path, w)
                    print(f"[live_press] saved {out_path}")
            save_n += 1

    for stream in streams:
        (stream.stop if hasattr(stream, "stop") else stream.release)()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live piano key-press detector (1 or 2 cameras)")
    p.add_argument("--calib", required=True, nargs="+",
                   help="path(s) to _keys.json calibration file(s); "
                        "two files enables dual-camera union fusion")
    p.add_argument("--cam", type=int, nargs="*", default=None,
                   help="OpenCV camera index/indices (default: auto-detect Canon)")
    p.add_argument("--baseline", type=int, default=N_BASELINE_FRAMES,
                   help=f"frames for quiet-keyboard baseline (default {N_BASELINE_FRAMES} ≈ 2 s)")
    args = p.parse_args()

    run_live(args.calib, args.cam, args.baseline)
