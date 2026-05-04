"""main.py — live piano transcription pipeline.

Top-level orchestration is here; reusable per-frame loop bodies and
helpers live in ``pipeline.py``. Tunable hyperparameters at the top of
this file.

Controls (focus the ``piano`` window):
    SPACE → calibrate from current frame, begin detection + transcription
    c     → recalibrate from current frame
    b     → recapture REST baseline over BASELINE_FRAMES quiet frames
    ESC   → save MIDI and exit

Set ``DUAL_CAM = True`` for synced two-camera mode: SPACE calibrates
both cams (one each with far_side="right"/"left"); per-cam press sets
are fused via near-side-weighted voting before driving the Transcriber.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2

from record import _build_transcribe_lut
from transcribe import Transcriber

import pipeline


# ── Camera ──────────────────────────────────────────────────────────────
DUAL_CAM = True
CAM_INDEX = 0                     # single-cam OpenCV index (None = auto-detect)
CAM_INDICES = [0, 1]              # dual-cam indices (cam0, cam1)
FAR_SIDES = ["right", "left"]     # camera-far direction per cam in dual mode

# ── Calibration ─────────────────────────────────────────────────────────
TOP_CROP = 10                     # pixels of case-top trimmed from the warp output

# ── Detection (lower = more sensitive / responsive) ────────────────────
SMOOTH_WINDOW = 1                 # rolling-mean over per-key counts
PRESS_PIXELS = 3                  # activated pixels needed per key to fire
MIN_BLOB_AREA = 1                 # CC area floor (1 disables the filter)
BOUNDARY_MARGIN = 1               # px erosion for pixel→key assignment

# ── Baseline (`b` hotkey) ───────────────────────────────────────────────
BASELINE_FRAMES = 60

# ── HandGate (fingertip-to-key candidate filter) ───────────────────────
HAND_GATE = True
HAND_GATE_TTL = 5
HAND_GATE_NEIGHBORS = 1

# ── Dual-cam fusion ─────────────────────────────────────────────────────
# Simple UNION across cams (any cam fires → fire) + a per-key TTL
# hold-over so brief same-key gaps between cams don't flicker a held
# note. (Legacy weighted-vote knobs kept for opt-in via pipeline.fuse_dual.)
DUAL_CAM_TEMPORAL_WINDOW = 3      # frames a key stays pressed after both cams drop it
DUAL_CAM_WEIGHT_FLOOR = 0.3
DUAL_CAM_VOTE_THRESHOLD = 0.5


def _build_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        cam_index=CAM_INDEX, cam_indices=CAM_INDICES, far_sides=FAR_SIDES,
        top_crop=TOP_CROP,
        smooth_window=SMOOTH_WINDOW, press_pixels=PRESS_PIXELS,
        min_blob_area=MIN_BLOB_AREA, boundary_margin=BOUNDARY_MARGIN,
        baseline_frames=BASELINE_FRAMES,
        hand_gate=HAND_GATE, hand_gate_ttl=HAND_GATE_TTL,
        hand_gate_neighbors=HAND_GATE_NEIGHBORS,
        dual_cam_temporal_window=DUAL_CAM_TEMPORAL_WINDOW,
        dual_cam_weight_floor=DUAL_CAM_WEIGHT_FLOOR,
        dual_cam_vote_threshold=DUAL_CAM_VOTE_THRESHOLD,
    )


def run_single(cfg):
    """Single-camera transcription session.

    Open camera → preview until SPACE → build Detector / HandGate /
    Transcriber on the captured calibration → detection loop with `c`
    (recalibrate) and `b` (re-baseline) hotkeys → save MIDI on ESC.
    """
    stream = pipeline.open_stream_single(cfg)
    transcriber, hand_gate = None, None
    try:
        # 1. Preview until SPACE → first calibration
        keys_dict, calib_warped = pipeline.preview_until_calibrate(cfg, stream)
        if keys_dict is None:
            return

        # 2. Build runtime objects from the calibration
        transcriber = Transcriber(fps=30.0)
        detector = pipeline.build_detector(cfg, keys_dict, calib_warped)
        key_lut = _build_transcribe_lut(keys_dict)
        hand_gate = pipeline.build_hand_gate(cfg, keys_dict)

        # 3. Detection loop. `c` returns ("recalibrate", frame); we rebuild.
        while True:
            outcome = pipeline.detect_until_quit_or_recalib(
                cfg, stream, detector, transcriber, key_lut, keys_dict, hand_gate,
            )
            if outcome == "quit":
                break
            _, frame = outcome
            keys_dict, calib_warped = pipeline.calibrate_from_frame(
                cfg, frame, camera_id="cam0",
            )
            if keys_dict is None:
                continue   # auto-cal failed; keep prior calibration
            detector = pipeline.build_detector(cfg, keys_dict, calib_warped)
            key_lut = _build_transcribe_lut(keys_dict)
            if hand_gate is not None:
                hand_gate.close()
            hand_gate = pipeline.build_hand_gate(cfg, keys_dict)
    finally:
        # 4. Save MIDI + clean up
        if transcriber is not None:
            try: pipeline.save_midi(transcriber)
            except Exception as e: print(f"save_midi failed: {e}")
        if hand_gate is not None:
            try: hand_gate.close()
            except Exception: pass
        stream.stop()
        cv2.destroyAllWindows()


def run_dual(cfg):
    """Dual-camera transcription session with weighted-vote fusion.

    Open both Canons synced via DualCanonStream → preview both side by
    side until SPACE → calibrate each cam (with its own ``far_side``) →
    build per-cam Detector + per-cam coverage weights → run fused
    detection loop → save MIDI on ESC.

    HandGate is forced off in dual-mode: it would add a second MediaPipe
    inference per cam (4 total per frame), pushing us over the 33 ms
    budget at 30 fps. The Detector's own MP-aware hand mask already
    excludes hand pixels in the diff, which is the dominant guard.
    """
    cfg.hand_gate = False  # see docstring — MP cost prohibitive in dual
    stream = pipeline.open_stream_dual(cfg)
    transcriber, hand_gates = None, [None, None]
    try:
        # 1. Side-by-side preview until SPACE → calibrate both cams
        results = pipeline.preview_until_calibrate_dual(cfg, stream)
        if results is None:
            return
        keys_dicts = [r[0] for r in results]
        warpeds = [r[1] for r in results]
        if len(keys_dicts[0]["keys"]) != len(keys_dicts[1]["keys"]):
            print(f"WARNING: cams disagree on key count: "
                  f"{[len(k['keys']) for k in keys_dicts]}. Fusion assumes matched indices.")

        # 2. Build per-cam runtime objects + coverage weights
        transcriber = Transcriber(fps=30.0)
        detectors = [pipeline.build_detector(cfg, keys_dicts[ci], warpeds[ci])
                     for ci in range(2)]
        key_lut = _build_transcribe_lut(keys_dicts[0])
        hand_gates = [pipeline.build_hand_gate(cfg, keys_dicts[ci]) for ci in range(2)]
        n_keys = detectors[0].n_keys
        weights = [pipeline.per_cam_weights(n_keys, cfg.far_sides[ci],
                                            cfg.dual_cam_weight_floor)
                   for ci in range(2)]
        print(f"DUAL-CAM fusion: floor={cfg.dual_cam_weight_floor} "
              f"threshold={cfg.dual_cam_vote_threshold} far_sides={cfg.far_sides}")

        # 3. Detection loop. `c` returns ("recalibrate", [f0, f1]); rebuild both.
        while True:
            outcome = pipeline.detect_until_quit_or_recalib_dual(
                cfg, stream, detectors, transcriber, key_lut,
                keys_dicts, hand_gates, weights,
            )
            if outcome == "quit":
                break
            _, frames = outcome
            new_results = []
            for ci, frame in enumerate(frames):
                nk, nw = pipeline.calibrate_from_frame(
                    cfg, frame, far_side=cfg.far_sides[ci], camera_id=f"cam{ci}",
                )
                if nk is None:
                    new_results = None
                    break
                new_results.append((nk, nw))
            if new_results is None:
                continue
            keys_dicts = [r[0] for r in new_results]
            detectors = [pipeline.build_detector(cfg, keys_dicts[ci], new_results[ci][1])
                         for ci in range(2)]
            key_lut = _build_transcribe_lut(keys_dicts[0])
            for ci in range(2):
                if hand_gates[ci] is not None:
                    hand_gates[ci].close()
            hand_gates = [pipeline.build_hand_gate(cfg, keys_dicts[ci]) for ci in range(2)]
            weights = [pipeline.per_cam_weights(detectors[0].n_keys, cfg.far_sides[ci],
                                                cfg.dual_cam_weight_floor)
                       for ci in range(2)]
    finally:
        # 4. Save MIDI + clean up both cams
        if transcriber is not None:
            try: pipeline.save_midi(transcriber)
            except Exception as e: print(f"save_midi failed: {e}")
        for hg in hand_gates:
            if hg is not None:
                try: hg.close()
                except Exception: pass
        try: stream.stop()
        except Exception: pass
        cv2.destroyAllWindows()


def main():
    cfg = _build_cfg()
    if DUAL_CAM:
        run_dual(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
