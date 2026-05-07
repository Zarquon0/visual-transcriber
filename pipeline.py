"""pipeline.py — orchestration for the live transcription pipeline.

Two top-level entry points called from ``main.py``:

    run_single(cfg) — one Canon stream, single Detector + HandGate
    run_dual(cfg)   — two Canons via DualCanonStream, per-cam Detector +
                      HandGate, weighted-vote fusion across cams

Both paths share the same building blocks (preview loop with HUD,
calibration on SPACE, detection loop with c/b hotkeys, 7-panel
warp_lines view, Transcriber + MIDI save). ``cfg`` is a SimpleNamespace
of constants supplied by main.py — no module-level config here.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

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
from stream_webcams import CanonStream, _load_config
from transcribe import Transcriber


# ── HUD + small helpers ─────────────────────────────────────────────────

def _hud(img, lines, color=(255, 255, 255)):
    """Stamp left-aligned text lines onto the top-left of img (in-place)."""
    for i, t in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)


def _resize_to_height(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / max(1, h)
    return cv2.resize(img, (int(round(w * scale)), target_h))


# ── Camera open ─────────────────────────────────────────────────────────

def open_stream_single(cfg):
    """Open one Canon, or replay a recorded clip if cfg.recording_path is set.

    Recording playback uses ``VideoFileStream`` which exposes the same
    ``start() / read() / stop()`` interface as ``CanonStream`` and paces
    frames to the source FPS so the live detection loop runs at clip
    speed.
    """
    rec = getattr(cfg, "recording_path", None)
    if rec:
        from stream_webcams import VideoFileStream
        s = VideoFileStream(rec)
        s.start()
        return s
    if cfg.cam_index is not None:
        s = CanonStream(cfg.cam_index, _load_config(), show_stats=False)
        if not s.cap.isOpened():
            raise RuntimeError(f"could not open camera at index {cfg.cam_index}")
        s.start()
        return s
    streams = open_streams(allow_iphone=False)
    if not streams:
        raise RuntimeError("no camera found")
    streams[0].start()
    return streams[0]


def open_stream_dual(cfg):
    """Open both Canons as a synced DualCanonStream, or replay two recorded
    clips if cfg.recording_paths is a 2-tuple of paths."""
    recs = getattr(cfg, "recording_paths", None)
    if recs:
        from stream_webcams import DualVideoFileStream
        s = DualVideoFileStream(recs[0], recs[1])
        s.start()
        return s
    from stream_webcams import DualCanonStream
    s = DualCanonStream(cfg.cam_indices[0], cfg.cam_indices[1],
                        _load_config(), show_stats=False)
    s.start()
    return s


# ── Calibration + Detector + HandGate builders ──────────────────────────

def calibrate_from_frame(cfg, frame, far_side="right", camera_id="live"):
    """Calibrate from a single frame, pop the inspector window for that cam.

    Uses ``cfg.keyboard_layout`` to pick the SWSSW span (61-key C2-C7 by
    default; 25-key C4-C6 when LAYOUT_25KEY is set in main.py).
    """
    layout = getattr(cfg, "keyboard_layout", None)
    result = calibrate_frame(
        frame, top_crop=cfg.top_crop, far_side=far_side, camera_id=camera_id,
        layout=layout,
    )
    if result is None:
        print(f"auto-calibration failed [{camera_id}]")
        return None, None
    keys_dict, calib_warped, _ = result
    print(f"calibrated [{camera_id}]: {len(keys_dict['keys'])} keys "
          f"(far_side={far_side}, layout={layout})")
    return keys_dict, calib_warped


def build_detector(cfg, keys_dict, rest_warped_bgr):
    """Diff-mode Detector tuned to cfg's snappy-onset values."""
    detector = Detector(
        keys_dict, use_diff_only=True, use_mediapipe=True,
        smooth_window=cfg.smooth_window,
    )
    detector.set_diff_thresholds(cfg.press_pixels, cfg.min_blob_area)
    detector.set_diff_boundary_margin(cfg.boundary_margin)
    rest_gray = cv2.cvtColor(rest_warped_bgr, cv2.COLOR_BGR2GRAY)
    detector.set_rest_mean_frame(rest_gray, rest_warped_bgr)
    return detector


def build_hand_gate(cfg, keys_dict):
    """HandGate fingertip-to-key candidate filter from in-memory keys_dict.
    Returns None if disabled or unavailable."""
    if not cfg.hand_gate:
        return None
    try:
        from calibration import Calibration, save_calibration
        from hand_gate import HandGate
    except Exception as e:
        print(f"HandGate unavailable: {e}")
        return None
    tmp = Path("recordings") / "_snapshots" / "_pipeline_tmp_keys.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(keys_dict, tmp)
    return HandGate(
        Calibration.load(tmp),
        candidate_ttl_frames=cfg.hand_gate_ttl,
        include_neighbors=cfg.hand_gate_neighbors,
    )


# ── Visualization ───────────────────────────────────────────────────────

def build_warp_lines_panels(detector, keys_dict, warped, pressed, cam_role=None):
    """The 7-panel pipeline diagnostic stack.

    ``cam_role`` (optional, e.g. "LEFT CAM") prefixes the top panel's
    label so dual-cam users can immediately tell which window is which
    physical camera.
    """
    warp_with_press = warped.copy()
    for ki in pressed:
        try:
            poly = np.array(
                keys_dict["keys"][ki]["polygon"], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.drawContours(warp_with_press, [poly], -1,
                             (0, 0, 255), 3, cv2.LINE_AA)
        except Exception:
            pass
    far_side = keys_dict.get("far_side", "right")
    role_prefix = f"[{cam_role} | far={far_side}] " if cam_role else ""
    panels = [
        _label_panel(warp_with_press, f"{role_prefix}1. RAW WARP + PRESSES"),
        _label_panel(draw_warp_colored(warped, keys_dict), "2. SEGMENTATION"),
        _label_panel(detector._last_hand_viz, "3. HAND MASK (MP)"),
    ]
    if detector._last_mp_ext_viz is not None:
        panels.append(_label_panel(detector._last_mp_ext_viz, "4. MP EXTENDED WARP"))
    panels.append(_label_panel(detector._last_diff_raw_mask, "5. PRESS DIFF (raw mask)"))
    panels.append(_label_panel(detector._last_diff_counted_mask,
                               "6. COUNTED (-hand -boundary) <- press signal"))
    panels.append(_label_panel(detector._last_diff_overlay,
                               "7. PER-KEY ACTIVATION + SCORES"))
    return np.vstack(panels)


def save_midi(transcriber):
    midi_dir = Path("midi_outs")
    midi_dir.mkdir(exist_ok=True)
    midi_path = midi_dir / time.strftime("%Y%m%d_%H%M%S.mid")
    transcriber.save_midi(str(midi_path))
    print(f"saved MIDI → {midi_path}")


# ── Dual-cam fusion ─────────────────────────────────────────────────────

def per_cam_weights(n_keys, far_side, floor):
    """Linear weight 1.0 (camera-near edge) → floor (camera-far edge).

    Cam with far_side="right" sits left of the keyboard: key 0 is its
    near edge (weight 1.0), key n-1 is far (weight=floor).
    """
    if far_side == "right":
        return np.linspace(1.0, floor, n_keys, dtype=np.float32)
    return np.linspace(floor, 1.0, n_keys, dtype=np.float32)


def fuse_dual(per_cam_pressed, per_cam_weights, threshold):
    """Weighted-vote fusion. score[k] = Σ w_ci[k] * (1 if k in pressed_ci else 0).

    With weights tapering 1.0 → 0.3 across keyboard and threshold 0.5:
        cam alone on its NEAR side (w >= 0.5) fires
        cam alone on its FAR side (w 0.3) does NOT fire on its own
        both cams agreeing always fires (sum >= 0.6)
    """
    n_keys = len(per_cam_weights[0])
    score = np.zeros(n_keys, dtype=np.float32)
    for ci, pressed in enumerate(per_cam_pressed):
        w = per_cam_weights[ci]
        for k in pressed:
            if 0 <= k < n_keys:
                score[k] += w[k]
    return {int(k) for k, s in enumerate(score) if s >= threshold}


# ── Single-cam preview + detection loops ────────────────────────────────

def preview_until_calibrate(cfg, stream):
    """Source-frame preview until SPACE (calibrate + return) or ESC (None)."""
    cv2.namedWindow("piano", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            continue
        disp = frame.copy()
        _hud(disp, ["PREVIEW", "SPACE: calibrate (start detection)", "ESC: quit"])
        cv2.imshow("piano", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return None, None
        if k == ord(" "):
            kd, w = calibrate_from_frame(cfg, frame, camera_id="cam0")
            if kd is not None:
                return kd, w


def detect_until_quit_or_recalib(cfg, stream, detector, transcriber, key_lut, keys_dict, hand_gate):
    """Single-cam detection loop. Returns 'quit' or ('recalibrate', frame)."""
    M = detector.det_state["M"]
    W, H = detector.det_state["W"], detector.det_state["H"]
    polys_src, sbb_src, types = overlay_from_dict(keys_dict)
    cv2.namedWindow("warp_lines_cam0", cv2.WINDOW_NORMAL)

    bl_active = False
    bl_remaining = 0
    bl_g = bl_b = None
    bl_n = 0

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            continue
        warped = cv2.warpPerspective(frame, M, (W, H))

        if bl_active:
            wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            if bl_g is None:
                bl_g = np.zeros(wg.shape, dtype=np.float64)
                bl_b = np.zeros(warped.shape, dtype=np.float64)
            bl_g += wg.astype(np.float64)
            bl_b += warped.astype(np.float64)
            bl_n += 1
            bl_remaining -= 1
            if bl_remaining <= 0:
                detector.set_rest_mean_frame(
                    (bl_g / bl_n).astype(np.uint8),
                    (bl_b / bl_n).astype(np.uint8),
                )
                bl_active = False
                bl_g = bl_b = None
                bl_n = 0
                print(f"REST baseline updated ({cfg.baseline_frames} frames)")
            pressed = set()
        else:
            detector.set_source_frame(frame)
            pressed, _ = detector.process(warped)
            if hand_gate is not None:
                pressed = pressed & hand_gate.candidate_keys(frame)
            keys_now = [
                key_lut[ki] for ki in pressed
                if ki < len(key_lut) and key_lut[ki] is not None
            ]
            transcriber.update(keys_now)

        disp = draw_overlay_with_pressed(frame, polys_src, sbb_src, types, pressed)
        if bl_active:
            _hud(disp, [
                f"REST CAPTURE: keep hands AWAY ({bl_remaining} frames left)",
                "ESC: save MIDI & quit",
            ], color=(0, 200, 200))
        else:
            _hud(disp, [
                f"DETECTING — pressed: {sorted(pressed) if pressed else 'none'}",
                "c: recalibrate   b: re-baseline   ESC: save MIDI & quit",
            ])
        cv2.imshow("piano", disp)
        cv2.imshow("warp_lines_cam0",
                   build_warp_lines_panels(detector, keys_dict, warped, pressed))

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return "quit"
        if k == ord("c") and not bl_active:
            return ("recalibrate", frame)
        if k == ord("b") and not bl_active:
            bl_active = True
            bl_remaining = cfg.baseline_frames
            print(f"REST CAPTURE started — keep hands AWAY for {cfg.baseline_frames} frames")


# ── Dual-cam preview + detection loops ──────────────────────────────────

def preview_until_calibrate_dual(cfg, stream):
    """Side-by-side preview until SPACE (calibrate both) or ESC (None)."""
    cv2.namedWindow("piano", cv2.WINDOW_NORMAL)
    while True:
        f0, f1 = stream.read()
        if f0 is None or f1 is None:
            cv2.waitKey(1)
            continue
        composite = np.hstack([_resize_to_height(f0, 540), _resize_to_height(f1, 540)])
        _hud(composite, ["DUAL-CAM PREVIEW", "SPACE: calibrate both", "ESC: quit"])
        cv2.imshow("piano", composite)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return None
        if k == ord(" "):
            results = []
            for ci, frame in enumerate([f0, f1]):
                kd, w = calibrate_from_frame(
                    cfg, frame, far_side=cfg.far_sides[ci], camera_id=f"cam{ci}",
                )
                if kd is None:
                    results = None
                    break
                results.append((kd, w))
            if results is not None:
                return results


def detect_until_quit_or_recalib_dual(cfg, stream, detectors, transcriber, key_lut,
                                       keys_dicts, hand_gates, weights):
    """Dual-cam detection loop with weighted-vote fusion. Returns 'quit' or
    ('recalibrate', [frame0, frame1]).

    Per-cam Detector.process + HandGate.candidate_keys is the per-frame
    hot path; both call MediaPipe (~25 ms each) which releases the GIL
    during C++ inference. Running the two cams in parallel threads cuts
    that wall-time roughly in half.
    """
    from concurrent.futures import ThreadPoolExecutor

    Ms = [d.det_state["M"] for d in detectors]
    sizes = [(d.det_state["W"], d.det_state["H"]) for d in detectors]
    overlays = [overlay_from_dict(kd) for kd in keys_dicts]
    cv2.namedWindow("warp_lines_cam0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("warp_lines_cam1", cv2.WINDOW_NORMAL)

    bl_active = False
    bl_remaining = 0
    bl_g = [None, None]
    bl_b = [None, None]
    bl_n = 0

    # Per-key TTL counter for the union-with-hold fusion. When either
    # cam fires key k, ttl[k] is refreshed to ``temporal_window``. Each
    # frame the array decays by 1; ttl > 0 → key is pressed in fused.
    n_keys = detectors[0].n_keys
    press_ttl = np.zeros(n_keys, dtype=np.int32)

    # Two-worker pool kept for the lifetime of this loop. Each cam's
    # Detector + HandGate are independent objects so per-thread state is
    # already isolated; we only need a barrier (.result()) before fusion.
    executor = ThreadPoolExecutor(max_workers=2)

    def _process_cam(ci, frames, warpeds):
        detectors[ci].set_source_frame(frames[ci])
        pressed, _ = detectors[ci].process(warpeds[ci])
        if hand_gates[ci] is not None:
            pressed = pressed & hand_gates[ci].candidate_keys(frames[ci])
        return pressed

    try:
        while True:
            f0, f1 = stream.read()
            if f0 is None or f1 is None:
                cv2.waitKey(1)
                continue
            frames = [f0, f1]
            warpeds = [cv2.warpPerspective(frames[ci], Ms[ci], sizes[ci]) for ci in range(2)]

            if bl_active:
                for ci in range(2):
                    wg = cv2.cvtColor(warpeds[ci], cv2.COLOR_BGR2GRAY)
                    if bl_g[ci] is None:
                        bl_g[ci] = np.zeros(wg.shape, dtype=np.float64)
                        bl_b[ci] = np.zeros(warpeds[ci].shape, dtype=np.float64)
                    bl_g[ci] += wg.astype(np.float64)
                    bl_b[ci] += warpeds[ci].astype(np.float64)
                bl_n += 1
                bl_remaining -= 1
                if bl_remaining <= 0:
                    for ci in range(2):
                        detectors[ci].set_rest_mean_frame(
                            (bl_g[ci] / bl_n).astype(np.uint8),
                            (bl_b[ci] / bl_n).astype(np.uint8),
                        )
                    bl_active = False
                    bl_g = [None, None]
                    bl_b = [None, None]
                    bl_n = 0
                    print("REST baseline updated for both cams")
                per_cam_pressed = [set(), set()]
                fused_pressed = set()
            else:
                # Fan out the two cams' MP-heavy detection in parallel.
                futures = [
                    executor.submit(_process_cam, ci, frames, warpeds)
                    for ci in range(2)
                ]
                per_cam_pressed = [fut.result() for fut in futures]
                # ── UNIFIED OUTPUT: union of per-cam press sets, with
                # per-key temporal hold-over (TTL refresh on either cam,
                # decay 1/frame). This is the single press set that
                # drives ``transcriber.update`` below.
                press_ttl = np.maximum(press_ttl - 1, 0)
                for k in (per_cam_pressed[0] | per_cam_pressed[1]):
                    if 0 <= k < n_keys:
                        press_ttl[k] = cfg.dual_cam_temporal_window
                fused_pressed = {int(k) for k in range(n_keys) if press_ttl[k] > 0}
                keys_now = [
                    key_lut[ki] for ki in fused_pressed
                    if ki < len(key_lut) and key_lut[ki] is not None
                ]
                transcriber.update(keys_now)

            disps = []
            for ci in range(2):
                polys_src, sbb_src, types = overlays[ci]
                d = draw_overlay_with_pressed(frames[ci], polys_src, sbb_src, types, fused_pressed)
                d = _resize_to_height(d, 540)
                # Cam role indicator stamped at top-center of each half.
                role = ("LEFT CAM" if cfg.far_sides[ci] == "right"
                        else "RIGHT CAM")
                role_text = f"{role}  (far={cfg.far_sides[ci]})"
                (tw, _), _ = cv2.getTextSize(role_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tx = max(10, (d.shape[1] - tw) // 2)
                cv2.putText(d, role_text, (tx, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(d, role_text, (tx, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 255), 1, cv2.LINE_AA)
                disps.append(d)
            composite = np.hstack(disps)
            if bl_active:
                _hud(composite, [
                    f"REST CAPTURE (both): keep hands AWAY ({bl_remaining} left)",
                    "ESC: save MIDI & quit",
                ], color=(0, 200, 200))
            else:
                _hud(composite, [
                    f"DETECTING — fused: {sorted(fused_pressed) if fused_pressed else 'none'}",
                    f"  cam0:{sorted(per_cam_pressed[0])}  cam1:{sorted(per_cam_pressed[1])}",
                    "c: recalibrate (both)   b: re-baseline (both)   ESC: save MIDI & quit",
                ])
            cv2.imshow("piano", composite)
            for ci in range(2):
                role = ("LEFT CAM" if cfg.far_sides[ci] == "right"
                        else "RIGHT CAM")
                cv2.imshow(f"warp_lines_cam{ci}",
                           build_warp_lines_panels(detectors[ci], keys_dicts[ci],
                                                   warpeds[ci], per_cam_pressed[ci],
                                                   cam_role=role))

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                return "quit"
            if k == ord("c") and not bl_active:
                return ("recalibrate", frames)
            if k == ord("b") and not bl_active:
                bl_active = True
                bl_remaining = cfg.baseline_frames
                print(f"REST CAPTURE started — keep hands AWAY for {cfg.baseline_frames} frames")
    finally:
        executor.shutdown(wait=False)


# Public API exposed to main.py:
#   open_stream_single(cfg) / open_stream_dual(cfg)
#   preview_until_calibrate(cfg, stream) / preview_until_calibrate_dual(cfg, stream)
#   calibrate_from_frame(cfg, frame, far_side, camera_id)
#   build_detector(cfg, keys_dict, warped)
#   build_hand_gate(cfg, keys_dict)
#   detect_until_quit_or_recalib(cfg, stream, detector, transcriber, key_lut, keys_dict, hand_gate)
#   detect_until_quit_or_recalib_dual(cfg, stream, detectors, transcriber, key_lut, keys_dicts, hand_gates, weights)
#   per_cam_weights(n_keys, far_side, floor)  /  fuse_dual(per_cam_pressed, weights, threshold)
#   save_midi(transcriber)
