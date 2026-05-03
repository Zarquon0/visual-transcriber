"""playback.py — replay a recorded clip with live press-detection overlay.

Same detection pipeline as record.py's 'd' mode, but frames come from a
recording folder instead of the camera. Useful for iterating on detector
parameters against captured data.

Usage:
    uv run python playback.py recordings/<folder>
    uv run python playback.py recordings/<folder> --threshold 6
    uv run python playback.py recordings/<folder> --keys path/to/cam0_keys.json
    uv run python playback.py recordings/<folder> --stride 2 --speed 1.0

Controls:
    SPACE  → pause/resume
    -/+    → threshold down/up
    [/]    → step backward / forward by 1 frame (only when paused)
    s      → save current frame with overlay to recordings/_snapshots/
    ESC/q  → quit
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import csv
from record import (
    overlay_from_dict,
    draw_overlay,
    draw_overlay_with_pressed,
    build_detection_state,
)
from analyze import (
    skin_mask, channel_lines, channel_brightness, channel_slope,
    channel_temp_diff, apply_global_shift,
)


class FrameSource:
    """Abstraction over a cam0/ folder of PNGs OR a cam0.mp4 file.

    Teammates who clone the repo get cam0.mp4 (committed) but not raw
    PNG frames (gitignored — too big). This class falls back to MP4
    seek-based reading when PNGs aren't available so playback works
    against either source.
    """

    def __init__(self, folder: Path, stride: int = 1):
        cam = folder / "cam0"
        png_paths = sorted(cam.glob("*.png")) if cam.is_dir() else []
        if png_paths:
            self.mode = "png"
            self.png_paths = png_paths[::stride]
            self.n = len(self.png_paths)
            return
        mp4 = folder / "cam0.mp4"
        if not mp4.exists():
            raise SystemExit(
                f"no cam0/ frames and no cam0.mp4 in {folder} — "
                "either extract frames or grab the .mp4 via GitHub Releases"
            )
        self.mode = "mp4"
        self.cap = cv2.VideoCapture(str(mp4))
        if not self.cap.isOpened():
            raise SystemExit(f"could not open {mp4}")
        full_n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stride = max(1, stride)
        self.n = full_n // self.stride
        self._last_idx = -1

    def __len__(self) -> int:
        return self.n

    def read(self, idx: int) -> np.ndarray | None:
        if self.mode == "png":
            return cv2.imread(str(self.png_paths[idx]))
        target = idx * self.stride
        # Sequential reads are fast (no seek). Seek only when jumping.
        if target != self._last_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.cap.read()
        self._last_idx = target if ok else -1
        return frame if ok else None


def auto_find_chaos_folder(press_folder: Path) -> Path | None:
    parent = press_folder.parent
    for cand in parent.glob("*_chaos"):
        if (cand / "cam0").is_dir():
            return cand
    return None


def compute_rest_mean_frame(rest_folder: Path, det_state: dict) -> np.ndarray | None:
    """Mean grayscale warped frame over all rest frames — the canonical
    'unpressed' image that temporal-diff compares against."""
    try:
        frames = FrameSource(rest_folder)
    except SystemExit:
        return None
    M, W, H = det_state["M"], det_state["W"], det_state["H"]
    accum = np.zeros((H, W), dtype=np.float64)
    n = 0
    for i in range(len(frames)):
        bgr = frames.read(i)
        if bgr is None:
            continue
        warped = cv2.warpPerspective(bgr, M, (W, H))
        accum += cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float64)
        n += 1
    return (accum / max(1, n)).astype(np.uint8) if n else None


def compute_tempdiff_chaos_stats(chaos_folder: Path, det_state: dict,
                                 rest_gray: np.ndarray):
    try:
        frames = FrameSource(chaos_folder)
    except SystemExit:
        return None, None
    n_keys = len(det_state["per_key_mask"])
    accum = np.full((len(frames), n_keys), np.nan, dtype=np.float32)
    M, W, H = det_state["M"], det_state["W"], det_state["H"]
    for fi in range(len(frames)):
        bgr = frames.read(fi)
        if bgr is None:
            continue
        warped = cv2.warpPerspective(bgr, M, (W, H))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sk = skin_mask(warped)
        accum[fi] = channel_temp_diff(gray, sk, det_state, rest_gray)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(accum, axis=0)
        std = np.maximum(np.nanstd(accum, axis=0), 1.0)
    return mean, std


def compute_slope_baseline(rest_folder: Path, det_state: dict):
    """Per-key (mean_angle, std_angle) over all rest frames."""
    try:
        frames = FrameSource(rest_folder)
    except SystemExit:
        return None, None
    n_keys = len(det_state["per_key_mask"])
    accum = np.full((len(frames), n_keys), np.nan, dtype=np.float32)
    M, W, H = det_state["M"], det_state["W"], det_state["H"]
    for fi in range(len(frames)):
        bgr = frames.read(fi)
        if bgr is None:
            continue
        warped = cv2.warpPerspective(bgr, M, (W, H))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sk = skin_mask(warped)
        accum[fi] = channel_slope(gray, sk, det_state)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(accum, axis=0)
        std = np.maximum(np.nanstd(accum, axis=0), 1.0)
    return mean, std


def auto_find_rest_folder(press_folder: Path) -> Path | None:
    """Sibling folder ending in _rest, e.g. recordings/<ts>_rest."""
    parent = press_folder.parent
    for cand in parent.glob("*_rest"):
        if (cand / "cam0").is_dir():
            return cand
    return None


def compute_brightness_baseline(rest_folder: Path, det_state: dict) -> np.ndarray | None:
    """Per-key mean intensity over all rest frames (skin-masked)."""
    try:
        frames = FrameSource(rest_folder)
    except SystemExit:
        return None
    n_keys = len(det_state["per_key_mask"])
    accum = np.zeros((len(frames), n_keys), dtype=np.float32)
    M, W, H = det_state["M"], det_state["W"], det_state["H"]
    for fi in range(len(frames)):
        bgr = frames.read(fi)
        if bgr is None:
            continue
        warped = cv2.warpPerspective(bgr, M, (W, H))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sk = skin_mask(warped)
        accum[fi] = channel_brightness(gray, sk, det_state)
    return np.nanmean(accum, axis=0)


def load_brightness_thresholds(csv_path: Path, n_keys: int, margin: float) -> np.ndarray:
    """B_chaos_max from analyze.py's summary.csv, scaled by margin.
    NaN entries (broken brightness in old analyze runs) → fallback to
    median of valid values (or 15.0 if no valid entries)."""
    raw = np.full(n_keys, np.nan, dtype=np.float32)
    with csv_path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            ki = int(row["key_idx"])
            if ki >= n_keys:
                continue
            try:
                raw[ki] = float(row["B_chaos_max"])
            except ValueError:
                pass
    valid = raw[~np.isnan(raw)]
    fallback = float(np.median(valid)) if len(valid) else 15.0
    raw = np.where(np.isnan(raw), fallback, raw)
    return np.maximum(raw * margin, 1.0)


_LSD_VIZ = cv2.createLineSegmentDetector()


def diagnose_lines(warped_bgr, det_state, skin):
    """Return (per_key_score, viz_image) where viz shows EVERY LSD segment
    color-coded by its classification:
        green  = interior emergent (counts as anomalous toward a key)
        yellow = inside polygon but on boundary band (expected, ignored)
        red    = midpoint on skin pixel (suppressed)
        blue   = outside any polygon
    Skin-masked pixels are also rendered as translucent ORANGE so the
    user can verify hand coverage frame-by-frame.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    g_supp = gray.copy()
    g_supp[skin > 0] = 128
    res = _LSD_VIZ.detect(g_supp)
    n_keys = len(det_state["per_key_mask"])
    scores = np.zeros(n_keys, dtype=np.float32)
    # Start viz with the warped frame, then mark skin pixels: orange
    # fill at higher alpha + bright magenta contour outline so the mask
    # boundary is unambiguous frame-by-frame.
    viz = warped_bgr.copy()
    if skin is not None and np.any(skin):
        orange = np.zeros_like(viz)
        orange[skin > 0] = (0, 140, 255)   # BGR — vivid orange
        viz = cv2.addWeighted(orange, 0.55, viz, 1.0, 0)
        skin_contours, _ = cv2.findContours(
            (skin > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(viz, skin_contours, -1, (255, 0, 255), 2, cv2.LINE_AA)
    if res is None or res[0] is None:
        return scores, viz
    lines = res[0].reshape(-1, 4)
    if lines.size == 0:
        return scores, viz
    H, W = det_state["H"], det_state["W"]
    # Pre-compute gradient magnitude image to filter weak (shadow-like) edges
    # from strong (real geometric) edges. Threshold of 30 keeps real key
    # edges (typically 50-150 magnitude) while dropping shadow gradients
    # (typically <20).
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    GRAD_MIN = 30.0  # tunable; below this is treated as shadow noise
    for x1, y1, x2, y2 in lines:
        mx = int((x1 + x2) * 0.5)
        my = int((y1 + y2) * 0.5)
        if not (0 <= mx < W and 0 <= my < H):
            continue
        if grad_mag[my, mx] < GRAD_MIN:
            continue  # weak gradient → likely shadow, ignore entirely
        length = float(np.hypot(x2 - x1, y2 - y1))
        ki = int(det_state["key_id_map"][my, mx])
        on_boundary = bool(det_state["boundary_band"][my, mx] > 0)
        on_skin = bool(skin[my, mx] > 0)
        if on_skin:
            color = (0, 0, 255)        # red
        elif ki < 0:
            color = (200, 100, 0)      # blue
        elif on_boundary:
            color = (0, 220, 220)      # yellow
        else:
            color = (0, 255, 0)        # green — counts!
            scores[ki] += length
        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA)
    return scores, viz


def load_per_key_thresholds(csv_path: Path, n_keys: int, margin: float = 1.2):
    """Read summary.csv and return per-key threshold = chaos_max * margin
    (or rest_max if larger). Returns array length n_keys.

    Floors:
      - rest_max + 5  (never fire below the silent-keyboard line activity)
      - 0.3 * median(chaos_max)  (edge keys whose chaos_max was abnormally
        low from uneven hand-coverage during chaos capture get protected
        by the global median; keeps them from firing on tiny noise).
    """
    chaos_arr = np.full(n_keys, np.nan, dtype=np.float32)
    rest_arr = np.full(n_keys, np.nan, dtype=np.float32)
    with csv_path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            ki = int(row["key_idx"])
            if ki >= n_keys:
                continue
            try:
                chaos_arr[ki] = float(row["L_chaos_max"])
                rest_arr[ki] = float(row["L_rest_max"])
            except ValueError:
                pass
    valid_chaos = chaos_arr[~np.isnan(chaos_arr)]
    median_chaos_floor = 0.3 * float(np.median(valid_chaos)) if len(valid_chaos) else 5.0
    chaos_arr = np.where(np.isnan(chaos_arr), median_chaos_floor, chaos_arr)
    rest_arr = np.where(np.isnan(rest_arr), 0.0, rest_arr)
    thr = np.maximum.reduce([
        chaos_arr * margin,
        rest_arr + 5.0,
        np.full(n_keys, max(median_chaos_floor, 5.0), dtype=np.float32),
    ])
    return thr.astype(np.float32)


def detect_pressed_per_key(warped_bgr, det_state, thresholds: np.ndarray) -> set[int]:
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    sk = skin_mask(warped_bgr)
    scores = channel_lines(gray, sk, det_state)
    return {int(i) for i, s in enumerate(scores) if s > thresholds[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="recording folder (must contain cam0/)")
    ap.add_argument("--keys", default=None,
                    help="override keys.json path; default = <folder>/cam0_keys.json")
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="global fallback threshold (used if --thresholds not given)")
    ap.add_argument("--thresholds", default=None,
                    help="path to summary.csv from analyze.py for per-key thresholds")
    ap.add_argument("--margin", type=float, default=1.2,
                    help="multiplier on chaos_max when loading per-key thresholds")
    ap.add_argument("--debounce", type=int, default=2,
                    help="frames a key must stay above threshold before flagging press")
    ap.add_argument("--no-skin", action="store_true",
                    help="disable the skin mask entirely (test: is mask the bottleneck?)")
    ap.add_argument("--simple-skin", action="store_true",
                    help="use ONLY the original static YCrCb mask from "
                         "analyze.skin_mask (for A/B comparison vs the new "
                         "motion+color+persistence approach)")
    ap.add_argument("--rest-baseline", default=None,
                    help="rest folder; if given, replaces YCrCb skin mask with motion mask")
    ap.add_argument("--top-crop", type=int, default=0,
                    help="trim N pixels off top of warp; if >0, recalibrate from frame 0")
    ap.add_argument("--margin-black", type=float, default=1.0,
                    help="multiplier on black-key thresholds (raise to require bigger press signal)")
    ap.add_argument("--margin-white", type=float, default=1.0,
                    help="multiplier on white-key thresholds (lower to make whites more sensitive)")
    ap.add_argument("--no-brightness", action="store_true",
                    help="disable brightness channel; line-detection only")
    ap.add_argument("--no-slope", action="store_true",
                    help="disable slope-change channel")
    ap.add_argument("--no-tempdiff", action="store_true",
                    help="disable temporal-difference channel")
    ap.add_argument("--tempdiff-sigma", type=float, default=4.0,
                    help="press = temp_diff > chaos_mean + N*chaos_std (default 4)")
    ap.add_argument("--smooth-window", type=int, default=5,
                    help="rolling-window size for temporal smoothing of all channels")
    ap.add_argument("--bright-margin", type=float, default=1.5,
                    help="multiplier on B_chaos_max for brightness threshold")
    ap.add_argument("--stride", type=int, default=1, help="frame stride")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"missing folder {folder}")
    # FrameSource will raise if neither cam0/*.png nor cam0.mp4 exists.

    keys_path = Path(args.keys) if args.keys else (folder / "cam0_keys.json")
    if not keys_path.exists():
        raise SystemExit(f"missing keys.json at {keys_path}")

    keys_dict = json.loads(keys_path.read_text())
    # If --top-crop is set, recalibrate using frame 0 of the recording
    # so the warp + segmentation match the requested crop.
    if args.top_crop > 0:
        from auto_calibrate import calibrate_frame
        _src = FrameSource(folder)
        first_bgr = _src.read(0)
        if first_bgr is None:
            raise SystemExit("could not read first frame for top-crop recalibration")
        res = calibrate_frame(first_bgr, top_crop=args.top_crop, camera_id="playback")
        if res is None:
            print(f"recalibration with top_crop={args.top_crop} failed; using bundled keys.json")
        else:
            keys_dict, _, _ = res
            print(f"recalibrated from frame 0 with top_crop={args.top_crop}: "
                  f"{len(keys_dict['keys'])} keys")
    polys_src, sbb_src, types = overlay_from_dict(keys_dict)
    det_state = build_detection_state(keys_dict)
    n_keys = len(types)

    # Per-key thresholds from chaos analysis if available; else global default.
    if args.thresholds:
        thresholds = load_per_key_thresholds(Path(args.thresholds), n_keys, args.margin)
        # Per-type margin scaling: blacks usually need higher thresholds
        # (lots of intrinsic line activity from specular/edges), whites
        # usually need lower (they're nearly featureless at rest).
        for i, t in enumerate(types):
            scale = args.margin_black if t == "black" else args.margin_white
            thresholds[i] = thresholds[i] * scale
        print(f"loaded per-key thresholds from {args.thresholds} (margin={args.margin}, "
              f"black-scale={args.margin_black}, white-scale={args.margin_white})")
        print(f"  blacks: min={thresholds[[t=='black' for t in types]].min():.1f}  "
              f"median={np.median(thresholds[[t=='black' for t in types]]):.1f}")
        print(f"  whites: min={thresholds[[t=='white' for t in types]].min():.1f}  "
              f"median={np.median(thresholds[[t=='white' for t in types]]):.1f}")
    else:
        thresholds = np.full(n_keys, args.threshold, dtype=np.float32)
        print(f"using global threshold {args.threshold}")

    # Debounce buffer: count consecutive frames each key was above threshold.
    above_count = np.zeros(n_keys, dtype=np.int32)

    # Brightness channel setup: load chaos-derived thresholds + compute
    # rest-baseline mean per key from the sibling _rest folder.
    bright_baseline = None
    bright_thresholds = None
    if not args.no_brightness and args.thresholds:
        rest_folder = auto_find_rest_folder(folder)
        if rest_folder is not None:
            print(f"computing brightness baseline from {rest_folder}...")
            bright_baseline = compute_brightness_baseline(rest_folder, det_state)
            bright_thresholds = load_brightness_thresholds(
                Path(args.thresholds), n_keys, args.bright_margin
            )
            print(f"  brightness channel ENABLED  median_baseline="
                  f"{np.nanmedian(bright_baseline):.1f}  "
                  f"median_threshold={np.median(bright_thresholds):.1f}")
        else:
            print("  no _rest sibling folder found; brightness channel disabled")

    # Temporal-difference channel: |current_warped - rest_mean_warped|
    # mean per polygon. Simplest possible motion signal.
    rest_mean_frame = None
    tempdiff_chaos_mean = None
    tempdiff_chaos_std = None
    if not args.no_tempdiff:
        rest_folder = auto_find_rest_folder(folder)
        chaos_folder = auto_find_chaos_folder(folder)
        if rest_folder is not None:
            print(f"computing rest mean frame from {rest_folder}...")
            rest_mean_frame = compute_rest_mean_frame(rest_folder, det_state)
            if rest_mean_frame is not None and chaos_folder is not None:
                print(f"computing tempdiff chaos stats from {chaos_folder}...")
                tempdiff_chaos_mean, tempdiff_chaos_std = compute_tempdiff_chaos_stats(
                    chaos_folder, det_state, rest_mean_frame
                )
                print(f"  tempdiff channel ENABLED  "
                      f"median chaos mean={np.nanmedian(tempdiff_chaos_mean):.1f}  "
                      f"std={np.nanmedian(tempdiff_chaos_std):.1f}  "
                      f"sigma_thr={args.tempdiff_sigma}")
            else:
                print("  no _chaos folder; tempdiff channel partially disabled "
                      "(falling back to flat threshold)")
        else:
            print("  no _rest folder; tempdiff channel disabled")

    # Slope channel: weighted-mean angle of LSD lines per polygon. Detects
    # tilt of pre-existing lines (works well on blacks where lines already
    # exist in great number, and on any key with rich edge structure).
    slope_baseline_mean = None
    slope_baseline_std = None
    slope_n_sigma = 4.0  # press triggers when angle drifts > 4σ from baseline
    if not args.no_slope:
        rest_folder = auto_find_rest_folder(folder)
        if rest_folder is not None:
            print(f"computing slope baseline from {rest_folder}...")
            slope_baseline_mean, slope_baseline_std = compute_slope_baseline(
                rest_folder, det_state
            )
            valid_n = int(np.sum(~np.isnan(slope_baseline_mean)))
            print(f"  slope channel ENABLED  valid keys={valid_n}/{n_keys}  "
                  f"median std={np.nanmedian(slope_baseline_std):.1f}°")
        else:
            print("  no _rest sibling folder found; slope channel disabled")

    frames = FrameSource(folder, stride=args.stride)
    print(f"playback: {len(frames)} frames from {folder} (mode={frames.mode})")
    print(f"  speed={args.speed}x  debounce={args.debounce} frames")
    print("  SPACE=pause  -/+=global margin  1/2=black margin -/+  "
          "3/4=white margin -/+  [/]=±1 frame  ,/.=±30 (1s)  </>=±150 (5s)  "
          "0/9=start/end  s=snap  ESC=quit")

    snap_dir = Path("recordings/_snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create both windows at known on-screen positions so they don't
    # spawn off the visible monitor (e.g. after monitor disconnects).
    cv2.namedWindow("playback", cv2.WINDOW_NORMAL)
    cv2.moveWindow("playback", 50, 50)
    cv2.resizeWindow("playback", 1280, 540)
    cv2.namedWindow("warp_lines", cv2.WINDOW_NORMAL)
    cv2.moveWindow("warp_lines", 50, 620)
    cv2.resizeWindow("warp_lines", 1200, 400)

    margin = args.margin
    margin_black = args.margin_black
    margin_white = args.margin_white

    # Rolling-window buffers per channel: (smooth_window, n_keys) ring.
    SMOOTH_W = max(1, args.smooth_window)
    line_buf = np.zeros((SMOOTH_W, n_keys), dtype=np.float32)
    bright_buf = np.full((SMOOTH_W, n_keys), np.nan, dtype=np.float32)
    slope_buf = np.full((SMOOTH_W, n_keys), np.nan, dtype=np.float32)
    tempdiff_buf = np.full((SMOOTH_W, n_keys), np.nan, dtype=np.float32)
    buf_idx = 0
    buf_filled = 0

    # Previous-frame gray for motion-based hand mask. None on first frame.
    prev_gray = None
    REST_DIFF_T = 20    # |current - rest| > this → "foreground" candidate
    MOTION_DIFF_T = 5   # |current - prev| > this → "moving" → hand
    # Persistence map: each pixel that fires the motion+color detector
    # gets stamped with PERSISTENCE_FRAMES; decrements each frame. Pixels
    # with persistence>0 stay classified as hand even when motion stops
    # (e.g., a hand resting on a pressed key), provided they're still
    # foreground (differ from rest baseline).
    hand_persistence = None  # initialized lazily once we have warped shape
    PERSISTENCE_FRAMES = 12
    # HSV loose-skin range (color side of fusion).
    # Hand skin is reddish/orange, has nontrivial saturation, and is bright
    # but not pure-white. Warm-lit white keys are nearly desaturated, so
    # the saturation floor is the main discriminator.
    SKIN_H_LO1, SKIN_H_HI1 = 0, 30      # red-orange band
    SKIN_H_LO2, SKIN_H_HI2 = 165, 180   # wraparound red
    SKIN_S_MIN = 40                     # white keys typically S<30
    SKIN_V_MIN = 50                     # exclude deep shadows/black

    def recompute_thresholds():
        """Rebuild thresholds array from CSV + current per-type margins."""
        if not args.thresholds:
            return np.full(n_keys, args.threshold, dtype=np.float32)
        base = load_per_key_thresholds(Path(args.thresholds), n_keys, margin)
        for i, t in enumerate(types):
            base[i] *= (margin_black if t == "black" else margin_white)
        return base

    thresholds = recompute_thresholds()
    paused = False
    idx = 0
    last_press_set: set[int] = set()

    target_dt = (1.0 / 30.0) / max(0.05, args.speed)

    while True:
        t0 = time.perf_counter()
        if not paused:
            idx = min(idx + 1, len(frames) - 1)

        bgr = frames.read(idx)
        if bgr is None:
            print(f"could not read frame {idx}")
            break

        # Warp + per-key dual-channel scoring + per-type threshold + debounce.
        M = det_state["M"]
        W, H = det_state["W"], det_state["H"]
        warped = cv2.warpPerspective(bgr, M, (W, H))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Motion-based hand mask: pixels that differ from rest baseline
        # AND are moving frame-to-frame. Pressed keys are still after
        # they've dropped, so they pass both tests as NOT-hand. Hands
        # move continuously, so they get masked. Color-independent.
        if args.no_skin:
            sk = np.zeros(warped.shape[:2], dtype=np.uint8)
        elif args.simple_skin:
            # A/B comparison: original static YCrCb mask, no motion / no
            # adaptive color / no persistence / no blob filtering.
            sk = skin_mask(warped)
        elif rest_mean_frame is not None and prev_gray is not None:
            # Motion side: pixels different from rest AND moving frame-to-frame.
            rest_diff = cv2.absdiff(warped_gray, rest_mean_frame)
            motion_diff = cv2.absdiff(warped_gray, prev_gray)
            motion_mask = (rest_diff > REST_DIFF_T) & (motion_diff > MOTION_DIFF_T)
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            # ADAPTIVE color side, BOOTSTRAPPED FROM STATIC SKIN GATE only.
            # Sampling all motion pixels would include moving key-edge
            # pixels (during a press), poisoning the color range to
            # eventually cover keys too. Restrict the sample to pixels
            # that are BOTH moving AND already loosely-skin-colored —
            # i.e., adapt within the skin space, never outside.
            hue_ok = ((h >= SKIN_H_LO1) & (h <= SKIN_H_HI1)) | \
                     ((h >= SKIN_H_LO2) & (h <= SKIN_H_HI2))
            static_skin = hue_ok & (s >= SKIN_S_MIN) & (v >= SKIN_V_MIN)
            seed = motion_mask & static_skin
            seed_hsv = hsv[seed]
            if seed_hsv.shape[0] > 200:
                h_med = float(np.median(seed_hsv[:, 0]))
                s_med = float(np.median(seed_hsv[:, 1]))
                v_med = float(np.median(seed_hsv[:, 2]))
                h_tol, s_tol, v_tol = 12.0, 40.0, 60.0
                h_dist = np.minimum(
                    np.abs(h.astype(np.int16) - h_med),
                    180 - np.abs(h.astype(np.int16) - h_med),
                )
                # Adaptive range AND-ed with static skin so we never
                # match outside the skin envelope even if the median
                # happens to drift.
                adaptive = (
                    (h_dist < h_tol) &
                    (np.abs(s.astype(np.int16) - s_med) < s_tol) &
                    (np.abs(v.astype(np.int16) - v_med) < v_tol)
                )
                color_mask = adaptive & static_skin
            else:
                color_mask = static_skin
            foreground = rest_diff > REST_DIFF_T
            # Detected = (moving OR foreground-vs-rest) AND skin-colored.
            # Hand INTERIOR is foreground+skin even when not moving, so
            # this catches the whole hand body, not just its edges.
            # Pressed-key shadows are foreground but not skin-colored,
            # so the color filter rejects them.
            detected = (motion_mask | foreground) & color_mask
            if hand_persistence is None or hand_persistence.shape != warped_gray.shape:
                hand_persistence = np.zeros(warped_gray.shape, dtype=np.uint8)
            hand_persistence[detected] = PERSISTENCE_FRAMES
            decay = (~detected) & (hand_persistence > 0)
            hand_persistence[decay] -= 1
            recent_hand = hand_persistence > 0
            mask_tight_bool = detected | (recent_hand & foreground)
            mask_tight = (mask_tight_bool).astype(np.uint8) * 255
            # Dilate ONLY for blob grouping (so fingertips merge into a
            # hand blob), but use the original tight mask for the final
            # output so we don't bloat past actual hand pixels.
            mask_grouped = cv2.dilate(
                mask_tight,
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            )
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_grouped, connectivity=8
            )
            # Multi-blob filter — per component, evaluate shape on the
            # TIGHT (pre-dilation) mask, not the dilated one. A thin
            # press-edge sliver dilates into a fat box that would pass a
            # naive thickness check; checking tight pixels catches it.
            MIN_TIGHT_AREA = 100        # pixels of tight mask in this component
            MIN_TIGHT_THICKNESS = 4     # min(tight_bbox_w, tight_bbox_h)
            keep = np.zeros(n_labels, dtype=bool)
            tight_bool = mask_tight > 0
            for comp_idx in range(1, n_labels):  # 0 is background, skip
                tight_in_comp = (labels == comp_idx) & tight_bool
                tight_count = int(tight_in_comp.sum())
                if tight_count < MIN_TIGHT_AREA:
                    continue
                ys, xs = np.where(tight_in_comp)
                tw = int(xs.max() - xs.min() + 1)
                th = int(ys.max() - ys.min() + 1)
                if min(tw, th) < MIN_TIGHT_THICKNESS:
                    continue
                keep[comp_idx] = True
            # Apply the keep-filter to the ORIGINAL tight mask, not the
            # dilated one — final mask hugs actual hand pixels.
            keep_mask = keep[labels]
            sk = np.where(keep_mask & (mask_tight > 0), 255, 0).astype(np.uint8)
            # Hole-fill: 9px close fills gaps in the hand body (between
            # fingers, sub-pixel cracks) without bloating outer boundary.
            sk = cv2.morphologyEx(
                sk, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
            )
        else:
            sk = np.zeros(warped.shape[:2], dtype=np.uint8)
        prev_gray = warped_gray.copy()
        # Channel 1: anomalous-line length (visualization included).
        scores, line_viz = diagnose_lines(warped, det_state, sk)
        line_above = scores > thresholds
        # Channel 2: brightness delta from rest baseline, with global shift.
        bright_above = np.zeros(n_keys, dtype=bool)
        if bright_baseline is not None and bright_thresholds is not None:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            cur_b = channel_brightness(gray, sk, det_state)
            delta = cur_b - bright_baseline
            white_idx_arr = np.array([i for i, t in enumerate(types) if t == "white"])
            if len(white_idx_arr) > 0:
                wd = delta[white_idx_arr]
                if not np.all(np.isnan(wd)):
                    shift = float(np.nanmedian(wd))
                    if not np.isnan(shift):
                        delta = delta - shift
            with np.errstate(invalid="ignore"):
                ba = np.abs(delta) > bright_thresholds
            bright_above = np.where(np.isnan(delta), False, ba).astype(bool)
        # Channel 3: slope (angle) deviation from rest baseline.
        slope_z = np.full(n_keys, np.nan, dtype=np.float32)
        if slope_baseline_mean is not None:
            gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            cur_slope = channel_slope(gray2, sk, det_state)
            slope_delta = cur_slope - slope_baseline_mean
            slope_delta = ((slope_delta + 90.0) % 180.0) - 90.0
            with np.errstate(invalid="ignore"):
                slope_z = np.abs(slope_delta) / np.maximum(slope_baseline_std, 1.0)
        # Channel 4: temporal-difference vs rest mean frame.
        tempdiff_z = np.full(n_keys, np.nan, dtype=np.float32)
        if rest_mean_frame is not None:
            gray3 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            cur_td = channel_temp_diff(gray3, sk, det_state, rest_mean_frame)
            if tempdiff_chaos_mean is not None and tempdiff_chaos_std is not None:
                with np.errstate(invalid="ignore"):
                    tempdiff_z = (cur_td - tempdiff_chaos_mean) / np.maximum(
                        tempdiff_chaos_std, 1.0
                    )
            else:
                # Fallback if no chaos data: just use current value as score.
                tempdiff_z = cur_td

        # ── Roll into smoothing buffers ─────────────────────────────────
        line_buf[buf_idx] = scores
        bright_buf[buf_idx] = (line_above.astype(float))  # not used; kept for shape
        # store actual delta for brightness for smoothing:
        if bright_baseline is not None and bright_thresholds is not None:
            bright_buf[buf_idx] = np.abs(delta) - bright_thresholds  # signed: >0 means above
        slope_buf[buf_idx] = slope_z
        tempdiff_buf[buf_idx] = tempdiff_z
        buf_idx = (buf_idx + 1) % SMOOTH_W
        buf_filled = min(buf_filled + 1, SMOOTH_W)

        # ── Smoothed (rolling-mean) per-channel above-threshold flags ───
        with np.errstate(invalid="ignore"):
            line_smoothed = np.nanmean(line_buf[:buf_filled], axis=0)
            bright_smoothed = np.nanmean(bright_buf[:buf_filled], axis=0)
            slope_smoothed = np.nanmean(slope_buf[:buf_filled], axis=0)
            tempdiff_smoothed = np.nanmean(tempdiff_buf[:buf_filled], axis=0)
        line_above_s = line_smoothed > thresholds
        bright_above_s = np.where(np.isnan(bright_smoothed), False,
                                  bright_smoothed > 0).astype(bool)
        slope_above_s = np.where(np.isnan(slope_smoothed), False,
                                 slope_smoothed > slope_n_sigma).astype(bool)
        tempdiff_above_s = np.where(np.isnan(tempdiff_smoothed), False,
                                    tempdiff_smoothed > args.tempdiff_sigma).astype(bool)

        # Per-type fusion: blacks use line+slope+tempdiff;
        # whites use brightness+slope+tempdiff.
        is_black = np.array([t == "black" for t in types])
        above = np.where(
            is_black,
            line_above_s | slope_above_s | tempdiff_above_s,
            bright_above_s | slope_above_s | tempdiff_above_s,
        )
        above_count = np.where(above, above_count + 1, 0)
        press_set = {int(i) for i, c in enumerate(above_count) if c >= args.debounce}
        last_press_set = press_set
        # Always update the line-diagnosis window (cheap, very useful).
        # Stack: warped bgr top, classified-line viz below.
        diag = np.vstack([warped, line_viz])
        # Annotate top-line scores so we can see which keys are firing.
        if press_set:
            ki_top = max(press_set, key=lambda k: scores[k])
            txt = f"top-press key {ki_top}: score={scores[ki_top]:.0f}  thr={thresholds[ki_top]:.0f}"
            cv2.putText(diag, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(diag, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("warp_lines", diag)

        # Render overlay on the source frame.
        disp = draw_overlay_with_pressed(bgr, polys_src, sbb_src, types, press_set)

        # HUD.
        ph = 540
        pw = max(1, int(disp.shape[1] * (ph / disp.shape[0])))
        disp = cv2.resize(disp, (pw, ph))
        if args.thresholds:
            thr_label = f"global={margin:.1f}  blk={margin_black:.1f}  wht={margin_white:.1f}"
        else:
            thr_label = f"thr={thresholds[0]:.1f}"
        hud = [
            f"frame {idx}/{len(frames)-1}  {thr_label}",
            f"pressed: {sorted(press_set) if press_set else 'none'}",
            ("PAUSED" if paused else f"play {args.speed:.1f}x"),
            "SPACE=pause -/+=global 1/2=blk- /+ 3/4=wht- /+ ,/.=±30f s=snap ESC=quit",
        ]
        for i, t in enumerate(hud):
            y = 24 + i * 22
            cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            color = (0, 0, 255) if t == "PAUSED" or "pressed" in t and press_set else (255, 255, 255)
            cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)
        cv2.imshow("playback", disp)

        elapsed = time.perf_counter() - t0
        wait_ms = max(1, int((target_dt - elapsed) * 1000))
        k = cv2.waitKey(wait_ms) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord(" "):
            paused = not paused
        elif k in (ord("-"), ord("_")):
            margin = max(0.1, margin - 0.1)
            thresholds = recompute_thresholds()
            print(f"global margin: {margin:.2f}  black: {margin_black:.2f}  white: {margin_white:.2f}")
        elif k in (ord("="), ord("+")):
            margin += 0.1
            thresholds = recompute_thresholds()
            print(f"global margin: {margin:.2f}  black: {margin_black:.2f}  white: {margin_white:.2f}")
        elif k == ord("1"):
            margin_black = max(0.1, margin_black - 0.1)
            thresholds = recompute_thresholds()
            print(f"black margin: {margin_black:.2f}  (more sensitive)")
        elif k == ord("2"):
            margin_black += 0.1
            thresholds = recompute_thresholds()
            print(f"black margin: {margin_black:.2f}  (less sensitive)")
        elif k == ord("3"):
            margin_white = max(0.1, margin_white - 0.1)
            thresholds = recompute_thresholds()
            print(f"white margin: {margin_white:.2f}  (more sensitive)")
        elif k == ord("4"):
            margin_white += 0.1
            thresholds = recompute_thresholds()
            print(f"white margin: {margin_white:.2f}  (less sensitive)")
        elif k == ord("["):
            idx = max(0, idx - 2)  # -2 because non-paused branch will +1
            paused = True
        elif k == ord("]"):
            idx = min(len(frames) - 1, idx + 1)
            paused = True
        elif k == ord(","):  # back 30 frames (~1 sec)
            idx = max(0, idx - 31)
            paused = True
        elif k == ord("."):  # fwd 30 frames
            idx = min(len(frames) - 1, idx + 29)
            paused = True
        elif k == ord("<"):  # back 5 sec
            idx = max(0, idx - 151)
            paused = True
        elif k == ord(">"):  # fwd 5 sec
            idx = min(len(frames) - 1, idx + 149)
            paused = True
        elif k == ord("0"):  # jump to start
            idx = 0
            paused = True
        elif k == ord("9"):  # jump to end
            idx = len(frames) - 1
            paused = True
        elif k == ord("s"):
            ts = int(time.time())
            sp = snap_dir / f"playback_{ts}_f{idx:06d}.png"
            cv2.imwrite(str(sp), disp)
            print(f"saved {sp}")

        if not paused and idx >= len(frames) - 1:
            print(f"end. last_press_set={sorted(last_press_set)}")
            paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
