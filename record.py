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
from live_test import _find_specific_camera_index


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


def build_detection_state(keys_dict: dict):
    """Pre-rasterize masks needed for live anomalous-line detection."""
    from analyze import build_overlays
    return build_overlays(keys_dict)


def detect_pressed(warped_bgr, det_state, threshold: float) -> set[int]:
    """Run anomalous-LSD-line detection on a warped frame; return key
    indices whose interior emergent-line length exceeds threshold."""
    from analyze import skin_mask, channel_lines
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    sk = skin_mask(warped_bgr)
    scores = channel_lines(gray, sk, det_state)
    return {int(i) for i, s in enumerate(scores) if s > threshold}


def detect_pressed_with_viz(warped_bgr, det_state, threshold: float):
    """Same as detect_pressed but also returns the per-segment-classified
    visualization (green=anomalous, yellow=boundary, red=skin, blue=outside-polys).
    """
    from playback import diagnose_lines
    from analyze import skin_mask
    sk = skin_mask(warped_bgr)
    scores, viz = diagnose_lines(warped_bgr, det_state, sk)
    pressed = {int(i) for i, s in enumerate(scores) if s > threshold}
    return pressed, viz


def open_streams(allow_iphone: bool) -> list[CanonStream]:
    try:
        streams = open_canon_streams(allow_iphone=allow_iphone, silent=False)
        if streams:
            return streams
    except RuntimeError as e:
        print(f"open_canon_streams failed: {e}; trying single-cam fallback")
    idx = _find_specific_camera_index(("EOS", "Canon", "iPhone"))
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
    det_states: dict[int, dict] = {}      # cam_idx → build_detection_state(...)
    det_n_sigma = 3.0                     # press = score > mean + n_sigma * std
    det_press_set: dict[int, set[int]] = {}
    det_baseline_mean: dict[int, np.ndarray] = {}
    det_baseline_std: dict[int, np.ndarray] = {}
    det_baseline_buf: dict[int, list] = {}
    det_baseline_capturing: dict[int, int] = {}
    BASELINE_FRAMES = 60

    # Pre-build detection state for any --keys files supplied at launch,
    # so 'd' works without needing 'c' first.
    for ci, p in keys_map.items():
        if ci < len(streams) and p.exists():
            det_states[ci] = build_detection_state(json.loads(p.read_text()))

    print("controls: r/SPACE=record  s=snap  o=overlay  c=calibrate  "
          "/=crop+5  ,=crop-5  0=crop reset  k=save  d=detect  "
          "b=chaos baseline (hands hovering, no press)  -/+=n_sigma  ESC=quit")
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

            # Per-frame live press detection. Press = score > mean + N·σ
            # over the captured CHAOS baseline (hands hovering, no presses).
            if det_enabled:
                from playback import diagnose_lines
                from analyze import skin_mask
                for ci, f in enumerate(raw):
                    if ci not in det_states:
                        continue
                    M = det_states[ci]["M"]
                    W, H = det_states[ci]["W"], det_states[ci]["H"]
                    warped = cv2.warpPerspective(f, M, (W, H))
                    sk = skin_mask(warped)
                    raw_scores, line_viz = diagnose_lines(warped, det_states[ci], sk)
                    n = len(raw_scores)
                    # Capturing chaos baseline?
                    if det_baseline_capturing.get(ci, 0) > 0:
                        det_baseline_buf.setdefault(ci, []).append(raw_scores.copy())
                        det_baseline_capturing[ci] -= 1
                        if det_baseline_capturing[ci] == 0:
                            stack = np.stack(det_baseline_buf[ci])
                            det_baseline_mean[ci] = stack.mean(axis=0)
                            det_baseline_std[ci] = np.maximum(stack.std(axis=0), 1.0)
                            print(f"cam{ci} chaos baseline captured "
                                  f"({len(det_baseline_buf[ci])} frames). "
                                  f"median mean={np.median(det_baseline_mean[ci]):.1f}, "
                                  f"median std={np.median(det_baseline_std[ci]):.1f}")
                            det_baseline_buf[ci] = []
                    mean = det_baseline_mean.get(ci, np.zeros(n, dtype=np.float32))
                    std = det_baseline_std.get(ci, np.ones(n, dtype=np.float32))
                    z = (raw_scores - mean) / std
                    det_press_set[ci] = {int(i) for i, zi in enumerate(z) if zi > det_n_sigma}
                    # Annotate viz with capture or detection status.
                    if det_baseline_capturing.get(ci, 0) > 0:
                        msg = f"CAPTURING CHAOS BASELINE... {det_baseline_capturing[ci]} frames left"
                    else:
                        has_baseline = ci in det_baseline_mean
                        msg = (f"σ-thresh n={det_n_sigma:.1f}  "
                               f"baseline={'set' if has_baseline else 'NOT set (press b)'}")
                    cv2.putText(line_viz, msg, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(line_viz, msg, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow(f"warp_lines_cam{ci}", np.vstack([warped, line_viz]))
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
                    det_states[ci] = build_detection_state(keys_dict)
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
                    det_states[ci] = build_detection_state(keys_dict)
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
                if not det_states:
                    print("no detection state — press 'c' or pass --keys first")
                else:
                    det_enabled = not det_enabled
                    print(f"detect: {det_enabled}  threshold={det_threshold}")
            elif k == ord("b"):
                # Capture chaos baseline: hover hands above keys, cast
                # shadows, but DO NOT PRESS. We need realistic noise floor.
                if not det_states:
                    print("no detection state — press 'c' first")
                elif not det_enabled:
                    print("enable detect with 'd' first, then capture baseline")
                else:
                    for ci in det_states:
                        det_baseline_capturing[ci] = BASELINE_FRAMES
                        det_baseline_buf[ci] = []
                    print(f"capturing chaos baseline over {BASELINE_FRAMES} frames — "
                          "MOVE HANDS ABOVE KEYBOARD without pressing any keys")
            elif k == ord("-") or k == ord("_"):
                det_n_sigma = max(0.5, det_n_sigma - 0.5)
                print(f"n_sigma: {det_n_sigma}")
            elif k == ord("=") or k == ord("+"):
                det_n_sigma += 0.5
                print(f"n_sigma: {det_n_sigma}")
    finally:
        if recording:
            print(f"final: {rec_idx} frames in {rec_dir}")
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
