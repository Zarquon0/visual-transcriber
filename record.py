"""record.py — capture Canon camera frames to disk for offline analysis.

PNG dumps (lossless, no JPEG block artifacts) suitable for line/edge
analysis later. Optional segmentation overlay so you can see whether a
saved keys.json still aligns with the current camera mount.

Usage:
    uv run python record.py
    uv run python record.py --keys piano_photos/IMG_9064_keys.json
    uv run python record.py --keys piano_photos/IMG_9064_keys.json:0 \
                            --keys piano_photos/IMG_9072_keys.json:1
    uv run python record.py --rotate
    uv run python record.py --no-iphone

Controls (preview window must be focused):
    r / SPACE → toggle recording on/off
    s         → save a single snapshot of the current frame(s)
    o         → toggle segmentation overlay on/off
    ESC / q   → quit

PNGs go to recordings/<timestamp>/cam<i>/<000123>.png. Any --keys files
are copied alongside the recording for later reference.

Note: the overlay is only meaningful if the camera is in the same
position as when the keys.json was generated. If keys are misaligned,
re-run manual_calibrate.py against a fresh frame first.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from stream_webcams import open_canon_streams, CanonStream, _load_config
from stream_webcams import find_specific_camera_index


_NOTE_LABEL_TO_ENUM = None


def _label_to_key(label: str):
    """Parse 'C#3' → transcribe.Key(Note.CS, 3). None for '?' or malformed."""
    global _NOTE_LABEL_TO_ENUM
    if not label or label == "?":
        return None
    if _NOTE_LABEL_TO_ENUM is None:
        from transcribe import Key, Note
        _NOTE_LABEL_TO_ENUM = ({
            "C": Note.C, "C#": Note.CS, "D": Note.D, "D#": Note.DS,
            "E": Note.E, "F": Note.F, "F#": Note.FS, "G": Note.G,
            "G#": Note.GS, "A": Note.A, "A#": Note.AS, "B": Note.B,
        }, Key)
    table, KeyCls = _NOTE_LABEL_TO_ENUM
    for i, ch in enumerate(label):
        if ch.isdigit() or (ch == "-" and i < len(label) - 1):
            try:
                octave = int(label[i:])
            except ValueError:
                return None
            note = table.get(label[:i])
            return KeyCls(note, octave) if note is not None else None
    return None


def _build_transcribe_lut(keys_dict):
    return [_label_to_key(k.get("note", "")) for k in keys_dict["keys"]]


def parse_keys_args(items: list[str]) -> dict[int, Path]:
    """Each --keys arg is 'path' (assigned to next free cam index) or 'path:idx'."""
    result: dict[int, Path] = {}
    next_default = 0
    for item in items or []:
        if ":" in item and item.rsplit(":", 1)[1].isdigit():
            p_str, idx_s = item.rsplit(":", 1)
            idx = int(idx_s)
        else:
            p_str = item
            while next_default in result:
                next_default += 1
            idx = next_default
            next_default += 1
        result[idx] = Path(p_str)
    return result


def overlay_from_dict(d: dict):
    """Return polygons + safe_bboxes mapped back to source-frame coords."""
    src_corners = np.array(d["warp"]["corners_tl_tr_br_bl"], dtype=np.float32)
    W, H = d["warp"]["out_size"]
    dst_corners = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32
    )
    M_dst_to_src = cv2.getPerspectiveTransform(dst_corners, src_corners)

    polys_src, sbb_src, types = [], [], []
    for k in d["keys"]:
        poly_w = np.array(k["polygon"], dtype=np.float32).reshape(-1, 1, 2)
        polys_src.append(
            cv2.perspectiveTransform(poly_w, M_dst_to_src).astype(np.int32)
        )
        sx, sy, sw, sh = k["safe_bbox"]
        sbb_w = np.array(
            [[sx, sy], [sx + sw, sy], [sx + sw, sy + sh], [sx, sy + sh]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        sbb_src.append(
            cv2.perspectiveTransform(sbb_w, M_dst_to_src).astype(np.int32)
        )
        types.append(k["type"])
    return polys_src, sbb_src, types


def load_overlay(path: Path):
    return overlay_from_dict(json.loads(path.read_text()))


def calib_stats(d: dict) -> str:
    keys = d["keys"]
    blacks = sum(1 for k in keys if k["type"] == "black")
    whites = sum(1 for k in keys if k["type"] == "white")
    labeled = sum(1 for k in keys if k["note"] != "?")
    return f"{len(keys)} keys (b:{blacks} w:{whites}, labeled:{labeled})"


# BGR. Alternating per-type so adjacent keys are visually distinct.
_BLACK_COLORS = [(203, 192, 255), (255, 255, 0)]   # pink, cyan
_WHITE_COLORS = [(0, 255, 255), (128, 255, 128)]   # yellow, light-green


def palette_for(types):
    """Per-key BGR color, alternating within each type so any two adjacent
    keys of the same type get different fills."""
    counts = {"black": 0, "white": 0}
    out = []
    for t in types:
        pal = _BLACK_COLORS if t == "black" else _WHITE_COLORS
        out.append(pal[counts[t] % len(pal)])
        counts[t] += 1
    return out


def draw_overlay(frame, polys_src, sbb_src, types):
    """Filled per-key colors with alpha, plus thin outline for hard edges."""
    fill = frame.copy()
    colors = palette_for(types)
    for poly, color in zip(polys_src, colors):
        cv2.fillPoly(fill, [poly], color)
    blended = cv2.addWeighted(fill, 0.45, frame, 0.55, 0)
    for poly, color in zip(polys_src, colors):
        cv2.polylines(blended, [poly], True, color, 1, cv2.LINE_AA)
    for sbb in sbb_src:
        cv2.polylines(blended, [sbb], True, (255, 0, 255), 1, cv2.LINE_AA)
    return blended


def draw_warp_colored(warped_img, keys_dict):
    """Same palette logic but on the warped strip directly. Adds index
    numbers per key so you can call out which one is wrong."""
    out = warped_img.copy()
    fill = warped_img.copy()
    keys = keys_dict["keys"]
    types = [k["type"] for k in keys]
    colors = palette_for(types)
    polys = [np.array(k["polygon"], dtype=np.int32).reshape(-1, 1, 2) for k in keys]
    for poly, color in zip(polys, colors):
        cv2.fillPoly(fill, [poly], color)
    out = cv2.addWeighted(fill, 0.5, out, 0.5, 0)
    for poly, color in zip(polys, colors):
        cv2.polylines(out, [poly], True, color, 1, cv2.LINE_AA)
    # Number each black key 1..N for unambiguous reference.
    bi = 0
    for k, color in zip(keys, colors):
        if k["type"] != "black":
            continue
        bi += 1
        x, y, bw, bh = k["bbox"]
        cv2.putText(out, str(bi), (x + 2, y + bh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(bi), (x + 2, y + bh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_overlay_with_pressed(frame, polys_src, sbb_src, types, pressed_set):
    """Same as draw_overlay but pressed keys get a bright red fill."""
    fill = frame.copy()
    colors = palette_for(types)
    for ki, (poly, color) in enumerate(zip(polys_src, colors)):
        c = (0, 0, 255) if ki in pressed_set else color
        cv2.fillPoly(fill, [poly], c)
    blended = cv2.addWeighted(fill, 0.45, frame, 0.55, 0)
    for ki, (poly, color) in enumerate(zip(polys_src, colors)):
        c = (0, 0, 255) if ki in pressed_set else color
        thickness = 3 if ki in pressed_set else 1
        cv2.polylines(blended, [poly], True, c, thickness, cv2.LINE_AA)
    return blended


def open_streams(allow_iphone: bool) -> list[CanonStream]:
    try:
        streams = open_canon_streams(allow_iphone=allow_iphone, silent=False)
        if streams:
            return streams
    except RuntimeError as e:
        print(f"open_canon_streams failed: {e}; trying single-cam fallback")
    idx = find_specific_camera_index(("EOS", "Canon", "iPhone"))
    if idx is None:
        return []
    cfg = _load_config()
    s = CanonStream(idx, cfg, show_stats=False)
    return [s] if s.cap.isOpened() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keys", action="append", default=[],
        help="path to keys.json, optionally suffixed with ':<cam_idx>'. Repeatable.",
    )
    ap.add_argument("--rotate", action="store_true",
                    help="rotate frames 180° (camera mounted upside-down)")
    ap.add_argument("--no-iphone", action="store_true",
                    help="exclude iPhone Continuity Camera")
    ap.add_argument("--cam-index", type=int, default=None,
                    help="override camera index (0..3). Bypasses Canon auto-detection.")
    ap.add_argument("--top-crop", type=int, default=10,
                    help="initial top_crop value (px trimmed off warp top). "
                         "Default 10. Adjust at runtime with /, comma, 0.")
    ap.add_argument("--hand-gate", action="store_true",
                    help="gate visual detections with MediaPipe fingertip candidates")
    ap.add_argument("--hand-neighbors", type=int, default=0,
                    help="neighbor key expansion for HandGate (default 0)")
    ap.add_argument("--hand-ttl", type=int, default=5,
                    help="frames a candidate key stays live after fingertip leaves (default 5)")
    ap.add_argument("--hands-debug", action="store_true",
                    help="overlay raw fingertip landmarks and candidate key outlines")
    ap.add_argument("--mediapipe", action="store_true",
                    help="use MediaPipe Hands for the hand mask in live press detection")
    ap.add_argument("--diff", action="store_true",
                    help="use press_diff approach as primary live "
                         "press detection (absdiff vs rest baseline + threshold + "
                         "top-frame blob filter, scored per key polygon).")
    ap.add_argument("--diff-press-pixels", type=int, default=20,
                    help="per-key activated-pixel count to fire a press in --diff mode")
    ap.add_argument("--diff-min-blob-area", type=int, default=15,
                    help="connected-component area floor on the post-hand "
                         "activation mask (drops noise specks)")
    ap.add_argument("--diff-boundary-margin", type=int, default=1,
                    help="px to erode each key polygon when assigning activated "
                         "pixels (default 1 → 2-px gap between adjacent keys; "
                         "0 = no gap, fastest signal recovery but adjacent keys "
                         "may cross-fire on boundary blobs)")
    ap.add_argument("--smooth-window", type=int, default=2,
                    help="rolling-mean window over per-key counts before "
                         "threshold check (default 2 — lower = snappier; was "
                         "previously 5 and felt laggy at press onset)")
    ap.add_argument("--transcribe", action="store_true",
                    help="instantiate a Transcriber driven by cam0 presses: "
                         "plays a soundfont synth on every detected press and "
                         "writes a MIDI file on quit (requires fluidsynth + a "
                         ".sf2 in sound_fonts/)")
    args = ap.parse_args()

    if args.cam_index is not None:
        cfg = _load_config()
        s = CanonStream(args.cam_index, cfg, show_stats=True)
        if not s.cap.isOpened():
            print(f"failed to open --cam-index {args.cam_index}")
            sys.exit(1)
        streams = [s]
        print(f"opened cam at explicit index {args.cam_index}")
    else:
        streams = open_streams(allow_iphone=not args.no_iphone)
    if not streams:
        print("no camera found")
        sys.exit(1)
    for s in streams:
        s.start()
    print(f"opened {len(streams)} stream(s)")

    keys_map = parse_keys_args(args.keys)
    overlays: dict[int, tuple] = {}
    for ci, p in keys_map.items():
        if ci >= len(streams):
            print(f"warn: --keys for cam{ci} but only {len(streams)} stream(s)")
            continue
        if not p.exists():
            print(f"warn: {p} not found")
            continue
        overlays[ci] = load_overlay(p)
        print(f"overlay cam{ci} ← {p}")

    overlay_on = bool(overlays)
    recording = False
    rec_dir: Path | None = None
    rec_idx = 0
    snapshot_dir = Path("recordings/_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pending_keys: dict[int, dict] = {}
    # Default crop trims the case-top band that warp_to_piano includes by
    # design (back-of-keys buffer). 10 px works for most camera angles;
    # press '/' to crop further or '0' to reset to raw warp.
    top_crop = args.top_crop
    det_enabled = False
    det_press_set: dict[int, set[int]] = {}

    # Live detectors per cam — full 4-channel pipeline shared with playback.
    from detection import Detector
    detectors: dict[int, Detector] = {}
    hand_gates: dict[int, object] = {}
    # Transcriber driven by cam0 only (a single sound output per session).
    # Built lazily once the first cam0 keys_dict is available, then per-frame
    # `transcriber.update([Key, ...])` is called with cam0's pressed set.
    transcriber = None
    transcribe_luts: dict[int, list] = {}
    # Combined rest+chaos baseline capture state.
    # In --diff mode, REST is the only baseline that matters (CHAOS is
    # unused). Use a longer REST capture (~2 s) to let auto-exposure /
    # auto-white-balance settle, and skip CHAOS entirely.
    REST_FRAMES = 60 if args.diff else 30
    CHAOS_FRAMES = 0 if args.diff else 60
    bl_phase: dict[int, str] = {}   # cam_idx → "rest" | "chaos" | None
    bl_remaining: dict[int, int] = {}
    bl_rest_buf: dict[int, list] = {}
    bl_chaos_buf: dict[int, list] = {}
    bl_rest_gray_accum: dict[int, np.ndarray] = {}
    bl_rest_bgr_accum: dict[int, np.ndarray] = {}
    bl_rest_n: dict[int, int] = {}

    # Pre-build detection state + Detector for any --keys files supplied
    # at launch, so 'd' works without needing 'c' first.
    for ci, p in keys_map.items():
        if ci < len(streams) and p.exists():
            kd_pre = json.loads(p.read_text())
            detectors[ci] = Detector(
                kd_pre,
                use_mediapipe=args.mediapipe,
                use_diff_only=args.diff,
            )
            if args.diff:
                detectors[ci].set_diff_thresholds(args.diff_press_pixels, args.diff_min_blob_area)
                detectors[ci].set_diff_boundary_margin(args.diff_boundary_margin)
            if args.transcribe:
                transcribe_luts[ci] = _build_transcribe_lut(kd_pre)
            if args.hand_gate:
                from calibration import Calibration
                from hand_gate import HandGate
                hand_gates[ci] = HandGate(
                    Calibration.load(p),
                    candidate_ttl_frames=args.hand_ttl,
                    include_neighbors=args.hand_neighbors,
                )

    if args.transcribe:
        from transcribe import Transcriber
        transcriber = Transcriber(fps=30.0)
        print("Transcriber ENABLED — driven by cam0 presses")

    print("controls: r/SPACE=record  s=snap  o=overlay  c=calibrate  "
          "/=crop+5  ,=crop-5  0=crop reset  k=save  d=detect  "
          "b=baseline (rest then chaos)  -/+=margins ESC=quit")
    # Pre-create the main preview window at a known on-screen position
    # so it doesn't spawn off the visible monitor (e.g. after monitor
    # changes / disconnects). Other windows (warp_cam0, warp_lines_*)
    # follow OS defaults but can be dragged.
    cv2.namedWindow("recorder", cv2.WINDOW_NORMAL)
    cv2.moveWindow("recorder", 50, 50)
    cv2.resizeWindow("recorder", 1280, 540)
    try:
        while True:
            reads = [s.read() for s in streams]
            if not all(ok and f is not None for ok, f in reads):
                continue
            raw = [f for _, f in reads]
            if args.rotate:
                raw = [cv2.rotate(f, cv2.ROTATE_180) for f in raw]

            if recording and rec_dir is not None:
                for ci, f in enumerate(raw):
                    cv2.imwrite(str(rec_dir / f"cam{ci}" / f"{rec_idx:06d}.png"), f)
                rec_idx += 1

            # Per-frame live press detection — same Detector class as playback.
            # Two-phase baseline capture via 'b' hotkey: 30 frames REST
            # (hands away) then 60 frames CHAOS (hover, no presses).
            if det_enabled:
                from analyze import (
                    skin_mask as _ssm,
                    channel_temp_diff as _ctd,
                    channel_brightness as _cb,
                    channel_slope as _cs,
                    channel_lines as _cl,
                )
                for ci, f in enumerate(raw):
                    if ci not in detectors:
                        continue
                    M = detectors[ci].det_state["M"]
                    W, H = detectors[ci].det_state["W"], detectors[ci].det_state["H"]
                    warped = cv2.warpPerspective(f, M, (W, H))
                    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    # Baseline capture in progress?
                    phase = bl_phase.get(ci)
                    if phase == "rest":
                        # Accumulate per-pixel mean of warped gray + BGR
                        # (BGR needed for press_diff baseline).
                        if ci not in bl_rest_gray_accum:
                            bl_rest_gray_accum[ci] = np.zeros(
                                warped_gray.shape, dtype=np.float64
                            )
                            bl_rest_bgr_accum[ci] = np.zeros(
                                warped.shape, dtype=np.float64
                            )
                            bl_rest_n[ci] = 0
                        bl_rest_gray_accum[ci] += warped_gray.astype(np.float64)
                        bl_rest_bgr_accum[ci] += warped.astype(np.float64)
                        bl_rest_n[ci] += 1
                        # Also collect raw line scores for slope baseline (no skin mask yet).
                        sk_empty = np.zeros(warped.shape[:2], dtype=np.uint8)
                        bl_rest_buf.setdefault(ci, []).append({
                            "bright": _cb(warped_gray, sk_empty, detectors[ci].det_state),
                            "slope": _cs(warped_gray, sk_empty, detectors[ci].det_state),
                        })
                        bl_remaining[ci] -= 1
                        if bl_remaining[ci] <= 0:
                            # Finalize rest, transition to chaos.
                            rest_mean_gray = (
                                bl_rest_gray_accum[ci] / max(1, bl_rest_n[ci])
                            ).astype(np.uint8)
                            rest_mean_bgr = (
                                bl_rest_bgr_accum[ci] / max(1, bl_rest_n[ci])
                            ).astype(np.uint8)
                            detectors[ci].set_rest_mean_frame(
                                rest_mean_gray, rest_mean_bgr
                            )
                            bb = np.array([d["bright"] for d in bl_rest_buf[ci]])
                            ss = np.array([d["slope"] for d in bl_rest_buf[ci]])
                            with np.errstate(invalid="ignore"):
                                detectors[ci].set_brightness_baseline(
                                    np.nanmean(bb, axis=0)
                                )
                                slope_mean = np.nanmean(ss, axis=0)
                                slope_std = np.maximum(np.nanstd(ss, axis=0), 1.0)
                            detectors[ci].set_slope_baseline(slope_mean, slope_std)
                            if CHAOS_FRAMES <= 0:
                                # --diff mode: no chaos needed.
                                print(
                                    f"cam{ci} REST captured ({bl_rest_n[ci]} frames). "
                                    f"Detector ready (diff mode)."
                                )
                                bl_phase[ci] = None
                            else:
                                print(
                                    f"cam{ci} REST captured ({bl_rest_n[ci]} frames). "
                                    f"Now CHAOS phase: hover hands without pressing for "
                                    f"{CHAOS_FRAMES} frames."
                                )
                                bl_phase[ci] = "chaos"
                                bl_remaining[ci] = CHAOS_FRAMES
                                bl_chaos_buf[ci] = []
                    elif phase == "chaos":
                        rest_gray = (
                            bl_rest_gray_accum[ci] / max(1, bl_rest_n[ci])
                        ).astype(np.uint8)
                        sk_chaos = _ssm(warped)
                        td = _ctd(warped_gray, sk_chaos, detectors[ci].det_state, rest_gray)
                        ln = _cl(warped_gray, sk_chaos, detectors[ci].det_state)
                        bl_chaos_buf[ci].append({"td": td, "ln": ln})
                        bl_remaining[ci] -= 1
                        if bl_remaining[ci] <= 0:
                            tds = np.array([d["td"] for d in bl_chaos_buf[ci]])
                            lns = np.array([d["ln"] for d in bl_chaos_buf[ci]])
                            with np.errstate(invalid="ignore"):
                                td_mean = np.nanmean(tds, axis=0)
                                td_std = np.maximum(np.nanstd(tds, axis=0), 1.0)
                            ln_max = np.nanmax(lns, axis=0)
                            detectors[ci].set_tempdiff_chaos_stats(td_mean, td_std)
                            # Line threshold = chaos_max × 1.0 with floor.
                            detectors[ci].set_line_thresholds(
                                np.maximum(ln_max, 5.0).astype(np.float32)
                            )
                            # Brightness threshold = global default for now.
                            detectors[ci].set_brightness_thresholds(
                                np.full(detectors[ci].n_keys, 10.0, dtype=np.float32)
                            )
                            print(
                                f"cam{ci} CHAOS captured. Detector ready. "
                                f"line_thr median={np.median(ln_max):.1f}  "
                                f"tempdiff_chaos_mean median={np.median(td_mean):.1f}"
                            )
                            bl_phase[ci] = None

                    # Pass source frame so MediaPipe (running on extended warp)
                    # has the original-camera frame to project from.
                    detectors[ci].set_source_frame(f)
                    # Run detection (will be partial until both baselines captured).
                    pressed, line_viz = detectors[ci].process(warped)
                    if args.hand_gate and ci in hand_gates:
                        candidate_set = hand_gates[ci].candidate_keys(f)
                        det_press_set[ci] = pressed & candidate_set
                    else:
                        det_press_set[ci] = pressed
                    # Drive the Transcriber from cam0's pressed set.
                    if (
                        transcriber is not None
                        and ci == 0
                        and ci in transcribe_luts
                    ):
                        lut = transcribe_luts[ci]
                        keys_now = [
                            lut[ki] for ki in det_press_set[ci]
                            if ki < len(lut) and lut[ki] is not None
                        ]
                        transcriber.update(keys_now)
                    # HUD on viz: capture status or live detection state.
                    if phase == "rest":
                        msg = f"REST CAPTURE — keep hands away ({bl_remaining[ci]} left)"
                        col = (0, 200, 200)
                    elif phase == "chaos":
                        msg = f"CHAOS CAPTURE — hover, NO presses ({bl_remaining[ci]} left)"
                        col = (0, 165, 255)
                    else:
                        if args.diff:
                            ready = detectors[ci]._rest_mean_bgr is not None
                        else:
                            ready = (
                                detectors[ci]._line_thresholds is not None
                                and detectors[ci]._rest_mean_frame is not None
                            )
                        msg = (
                            f"DETECTING — pressed: {sorted(pressed) if pressed else 'none'}"
                            if ready
                            else "BASELINES NOT SET — press b"
                        )
                        col = (0, 255, 0) if ready else (0, 0, 255)
                    cv2.putText(line_viz, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(line_viz, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, col, 1, cv2.LINE_AA)
                    if args.diff:
                        from playback import _label_panel
                        warp_with_press = warped.copy()
                        for pk in pressed:
                            try:
                                kdict_keys = detectors[ci].det_state
                                # Recover polygon from per_key_mask via contour
                                m = kdict_keys["per_key_mask"][pk]
                                cnts_pk, _ = cv2.findContours(
                                    m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                )
                                cv2.drawContours(
                                    warp_with_press, cnts_pk, -1,
                                    (0, 0, 255), 3, cv2.LINE_AA,
                                )
                            except Exception:
                                pass
                        # Live segmentation panel uses the inspector's colored draw.
                        try:
                            seg = draw_warp_colored(warped, pending_keys.get(ci))
                        except Exception:
                            seg = warped.copy()
                        det = detectors[ci]
                        panels = [
                            _label_panel(warp_with_press, "1. RAW WARP + PRESSES"),
                            _label_panel(seg, "2. SEGMENTATION"),
                            _label_panel(det._last_hand_viz, "3. HAND MASK (MP)"),
                        ]
                        if args.mediapipe and det._last_mp_ext_viz is not None:
                            panels.append(_label_panel(
                                det._last_mp_ext_viz, "4. MP EXTENDED WARP"
                            ))
                        panels.append(_label_panel(
                            det._last_diff_raw_mask, "5. PRESS DIFF (raw mask)"
                        ))
                        panels.append(_label_panel(
                            det._last_diff_counted_mask,
                            "6. COUNTED (-hand -boundary) <- press signal",
                        ))
                        panels.append(_label_panel(
                            det._last_diff_overlay,
                            "7. PER-KEY ACTIVATION + SCORES",
                        ))
                        # Stamp the HUD msg on the top panel.
                        cv2.putText(panels[0], msg, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(panels[0], msg, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    col, 1, cv2.LINE_AA)
                        cv2.imshow(f"warp_lines_cam{ci}", np.vstack(panels))
                    else:
                        cv2.imshow(f"warp_lines_cam{ci}", np.vstack([warped, line_viz]))
                    if args.hands_debug and ci in hand_gates:
                        cv2.imshow(f"hand_warped_debug_cam{ci}",
                                   hand_gates[ci].draw_warped_debug(warped))
            else:
                det_press_set = {}

            tiles = []
            for ci, f in enumerate(raw):
                disp = f
                if overlay_on and ci in overlays:
                    if ci in det_press_set:
                        disp = draw_overlay_with_pressed(
                            disp, *overlays[ci], pressed_set=det_press_set[ci]
                        )
                    else:
                        disp = draw_overlay(disp, *overlays[ci])
                ph = 540
                pw = max(1, int(disp.shape[1] * (ph / disp.shape[0])))
                disp = cv2.resize(disp, (pw, ph))
                lines = [f"cam{ci}  top_crop={top_crop}"]
                if recording:
                    lines.append(f"REC {rec_idx}")
                if ci in pending_keys:
                    lines.append(calib_stats(pending_keys[ci]))
                lines.append("r=rec s=snap o=ov c=calib [/]=crop k=save ESC=quit")
                for i, t in enumerate(lines):
                    y = 25 + i * 24
                    cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    color = (0, 0, 255) if t.startswith("REC") else (255, 255, 255)
                    cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 1, cv2.LINE_AA)
                tiles.append(disp)
            if len(tiles) == 1:
                preview = tiles[0]
            else:
                h = min(t.shape[0] for t in tiles)
                tiles = [
                    cv2.resize(t, (max(1, int(t.shape[1] * (h / t.shape[0]))), h))
                    for t in tiles
                ]
                preview = np.hstack(tiles)

            cv2.imshow("recorder", preview)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k in (ord("r"), ord(" ")):
                if recording:
                    print(f"stopped: {rec_idx} frames in {rec_dir}")
                    recording = False
                    # Auto-archive: bundle calibration + warp/segmented/overlay
                    # snapshots + manifest.md so the recording is self-describing.
                    if rec_dir is not None:
                        try:
                            from archive_recording import archive
                            archive(rec_dir)
                        except Exception as e:
                            print(f"archive failed (non-fatal): {e}")
                else:
                    ts = int(time.time())
                    rec_dir = Path(f"recordings/{ts}")
                    for ci in range(len(streams)):
                        (rec_dir / f"cam{ci}").mkdir(parents=True, exist_ok=True)
                    # Prefer the in-memory pending calibration over the
                    # --keys file if both exist — pending reflects the
                    # user's most recent 'c' press, which is presumably
                    # the one they want to record against.
                    for ci in range(len(streams)):
                        out_keys = rec_dir / f"cam{ci}_keys.json"
                        if ci in pending_keys:
                            from calibration import save_calibration
                            save_calibration(pending_keys[ci], out_keys)
                        elif ci in keys_map and keys_map[ci].exists():
                            shutil.copy(keys_map[ci], out_keys)
                    rec_idx = 0
                    recording = True
                    print(f"recording → {rec_dir}")
            elif k == ord("s"):
                ts = int(time.time())
                for ci, f in enumerate(raw):
                    p = snapshot_dir / f"snap_{ts}_cam{ci}.png"
                    cv2.imwrite(str(p), f)
                    print(f"saved {p}")
            elif k == ord("o"):
                overlay_on = not overlay_on
                print(f"overlay: {overlay_on}")
            elif k == ord("c"):
                # Auto-calibrate from current frames, swap overlays in place,
                # and pop a "warp_cam<i>" window showing the rectified strip
                # with polygons drawn — the unambiguous "did it work" view.
                from auto_calibrate import calibrate_frame
                from key_labeler import draw_labels_tight_crop
                for ci, f in enumerate(raw):
                    res = calibrate_frame(f, top_crop=top_crop, camera_id=f"cam{ci}")
                    if res is None:
                        print(f"cam{ci}: auto-calibrate failed (no blob)")
                        continue
                    keys_dict, warped_img, _ = res
                    overlays[ci] = overlay_from_dict(keys_dict)
                    pending_keys[ci] = keys_dict
                    detectors[ci] = Detector(keys_dict, smooth_window=args.smooth_window, use_mediapipe=args.mediapipe, use_diff_only=args.diff)
                    if args.diff:
                        detectors[ci].set_diff_thresholds(args.diff_press_pixels, args.diff_min_blob_area)
                        detectors[ci].set_diff_boundary_margin(args.diff_boundary_margin)
                    if args.transcribe:
                        transcribe_luts[ci] = _build_transcribe_lut(keys_dict)
                    if args.hand_gate:
                        from calibration import save_calibration, Calibration
                        from hand_gate import HandGate
                        tmp = snapshot_dir / f"_tmp_cam{ci}_keys.json"
                        save_calibration(keys_dict, tmp)
                        hand_gates[ci] = HandGate(
                            Calibration.load(tmp),
                            candidate_ttl_frames=args.hand_ttl,
                            include_neighbors=args.hand_neighbors,
                        )
                    print(f"cam{ci} calibrated: {calib_stats(keys_dict)}")
                    # Warp inspector: raw warp on top, colored+numbered overlay below.
                    colored = draw_warp_colored(warped_img, keys_dict)
                    cv2.imshow(f"warp_cam{ci}", np.vstack([warped_img, colored]))
                overlay_on = bool(overlays)
            elif k in (ord("["), ord("]"), ord("/"), ord("\\"), ord(","), ord("0")):
                # Crop more / less / reset:
                #   / or ]    → crop +5 (more pixels trimmed off top)
                #   \ or [ or , → crop -5 (undo one step, restore those pixels)
                #   0         → reset crop to 0 (full undo)
                if k == ord("0"):
                    top_crop = 0
                elif k in (ord("]"), ord("/")):
                    top_crop += 5
                else:
                    top_crop = max(0, top_crop - 5)
                print(f"top_crop = {top_crop}; recalibrating...")
                # Auto-recalibrate from current frames so you see the new
                # warp immediately without needing to also press 'c'.
                from auto_calibrate import calibrate_frame
                from key_labeler import draw_labels_tight_crop
                for ci, f in enumerate(raw):
                    res = calibrate_frame(f, top_crop=top_crop, camera_id=f"cam{ci}")
                    if res is None:
                        print(f"cam{ci}: auto-calibrate failed at top_crop={top_crop}")
                        continue
                    keys_dict, warped_img, _ = res
                    overlays[ci] = overlay_from_dict(keys_dict)
                    pending_keys[ci] = keys_dict
                    detectors[ci] = Detector(keys_dict, smooth_window=args.smooth_window, use_mediapipe=args.mediapipe, use_diff_only=args.diff)
                    if args.diff:
                        detectors[ci].set_diff_thresholds(args.diff_press_pixels, args.diff_min_blob_area)
                        detectors[ci].set_diff_boundary_margin(args.diff_boundary_margin)
                    if args.transcribe:
                        transcribe_luts[ci] = _build_transcribe_lut(keys_dict)
                    if args.hand_gate:
                        from calibration import save_calibration, Calibration
                        from hand_gate import HandGate
                        tmp = snapshot_dir / f"_tmp_cam{ci}_keys.json"
                        save_calibration(keys_dict, tmp)
                        hand_gates[ci] = HandGate(
                            Calibration.load(tmp),
                            candidate_ttl_frames=args.hand_ttl,
                            include_neighbors=args.hand_neighbors,
                        )
                    print(f"cam{ci} (top_crop={top_crop}): {calib_stats(keys_dict)}")
                    colored = draw_warp_colored(warped_img, keys_dict)
                    cv2.imshow(f"warp_cam{ci}", np.vstack([warped_img, colored]))
                overlay_on = bool(overlays)
            elif k == ord("k"):
                if not pending_keys:
                    print("no calibration in memory yet — press 'c' first")
                else:
                    ts = int(time.time())
                    for ci, kd in pending_keys.items():
                        kp = snapshot_dir / f"calib_{ts}_cam{ci}_keys.json"
                        from calibration import save_calibration
                        save_calibration(kd, kp)
                        print(f"saved {kp}")
            elif k == ord("d"):
                if not detectors:
                    print("no detection state — press 'c' or pass --keys first")
                else:
                    det_enabled = not det_enabled
                    if args.diff:
                        any_det = next(iter(detectors.values()))
                        print(
                            f"detect: {det_enabled}  "
                            f"press_pix={any_det._diff_press_pixel_count}  "
                            f"min_blob={any_det._diff_min_blob_area}  "
                            f"bnd={any_det._diff_boundary_margin}"
                        )
                    else:
                        print(
                            f"detect: {det_enabled}  "
                            f"margin_blk={args.diff_press_pixels if args.diff else 'n/a'}"
                        )
            elif k == ord("b"):
                # Two-phase baseline capture:
                #   Phase 1 (rest):  30 frames, hands AWAY → rest mean,
                #                    brightness baseline, slope baseline.
                #   Phase 2 (chaos): 60 frames, hands HOVERING (no press) →
                #                    line thresholds, tempdiff chaos stats.
                if not detectors:
                    print("no detector — press 'c' first")
                elif not det_enabled:
                    print("enable detect with 'd' first, then 'b' to capture baselines")
                else:
                    for ci in detectors:
                        bl_phase[ci] = "rest"
                        bl_remaining[ci] = REST_FRAMES
                        bl_rest_buf[ci] = []
                        bl_chaos_buf[ci] = []
                        bl_rest_gray_accum.pop(ci, None)
                        bl_rest_n.pop(ci, None)
                    print(
                        f"BASELINE CAPTURE: REST phase first ({REST_FRAMES} frames, "
                        f"hands AWAY), then CHAOS phase ({CHAOS_FRAMES} frames, "
                        "hover but DO NOT PRESS)."
                    )
            elif k == ord("-") or k == ord("_"):
                for det in detectors.values():
                    det.set_margin_black(max(0.1, det.margin_black - 0.1))
                    det.set_margin_white(max(0.1, det.margin_white - 0.1))
                med = next(iter(detectors.values()), None)
                if med is not None:
                    print(
                        f"black margin: {med.margin_black:.2f}, "
                        f"white margin: {med.margin_white:.2f}"
                    )
            elif k == ord("=") or k == ord("+"):
                for det in detectors.values():
                    det.set_margin_black(det.margin_black + 0.1)
                    det.set_margin_white(det.margin_white + 0.1)
                med = next(iter(detectors.values()), None)
                if med is not None:
                    print(
                        f"black margin: {med.margin_black:.2f}, "
                        f"white margin: {med.margin_white:.2f}"
                    )
    finally:
        if recording:
            print(f"final: {rec_idx} frames in {rec_dir}")
        for hg in hand_gates.values():
            hg.close()
        if transcriber is not None:
            midi_dir = Path("midi_outs")
            midi_dir.mkdir(exist_ok=True)
            midi_path = midi_dir / time.strftime("%Y%m%d_%H%M%S.mid")
            try:
                transcriber.save_midi(str(midi_path))
                print(f"saved MIDI to {midi_path}")
            except Exception as e:
                print(f"transcriber save_midi failed: {e}")
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
