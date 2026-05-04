"""detection.py — unified per-frame press detection.

Single ``Detector`` class consumed by both ``record.py`` (live mode via
the `d` hotkey) and ``playback.py`` (offline replay). Identical algorithm
in both — improvements propagate automatically.

Channels (each gives a per-key score):
    * lines     — anomalous-LSD-line length per polygon
    * brightness — pixel intensity delta vs rest mean (skin-masked,
                   global-illumination corrected)
    * slope     — weighted-mean angle of LSD lines, in σ-units of rest
    * tempdiff  — pixel-wise |current − rest_mean| per polygon, σ vs chaos

Fusion (per key type):
    blacks → lines OR slope OR tempdiff
    whites → brightness OR slope OR tempdiff

Smoothing: rolling mean over ``smooth_window`` frames per channel before
threshold comparison. Plus consecutive-frames debounce on the fused flag.

Hand mask: motion + HSV color (adaptive within static-skin envelope) +
persistence + connected-component blob filter on tight-mask shape.

Workflow:
    det = Detector(keys_dict)
    # set whatever baselines are available; missing ones disable that channel
    det.set_line_thresholds(thr_array)             # required for line channel
    det.set_rest_mean_frame(mean_gray_image)        # for brightness + tempdiff
    det.set_brightness_thresholds(thr_array)
    det.set_tempdiff_chaos_stats(mean_array, std_array)
    det.set_slope_baseline(mean_array, std_array)
    # per frame:
    pressed, viz = det.process(warped_bgr)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from analyze import (
    build_overlays,
    channel_lines,
    channel_brightness,
    channel_slope,
    channel_temp_diff,
    skin_mask as ycrcb_skin_mask,
)
from press_diff import detect_press_regions

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions as _MPBaseOptions
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

_MP_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


# Hand-mask tuning. Per-pixel HSV-decomposition test:
#   hand   = |ΔH| > T_H  OR  ΔS > T_S    (color shifted = skin replacing key)
#   shadow = ΔV < -T_V  AND  |ΔH|<T_HS  AND  |ΔS|<T_SS  (darker but same color)
#   press  = small ΔV with no H/S change (handled by detection channels)
HAND_DELTA_H_T = 10      # hue change threshold for hand
HAND_DELTA_S_T = 8       # saturation increase threshold for hand
SHADOW_DELTA_V_T = 25    # V drop magnitude threshold for shadow
SHADOW_HSV_TOL_H = 5     # max hue change for a "still same color" shadow
SHADOW_HSV_TOL_S = 5     # max saturation change for a "still same color" shadow
HAND_BLOB_MIN_AREA = 300
HAND_BLOB_MIN_THICKNESS = 8
PERSISTENCE_FRAMES = 12
MIN_TIGHT_AREA = 100
MIN_TIGHT_THICKNESS = 4
GRAD_MIN = 30.0      # filter weak-gradient LSD segments (shadow-like)


class Detector:
    """Per-frame press detector consumed by record.py and playback.py."""

    def __init__(
        self,
        keys_dict: dict,
        smooth_window: int = 5,
        debounce: int = 1,
        slope_n_sigma: float = 4.0,
        tempdiff_n_sigma: float = 0.5,
        margin_black: float = 1.5,
        margin_white: float = 0.6,
        use_mediapipe: bool = False,
        use_diff_only: bool = False,
    ):
        self.det_state = build_overlays(keys_dict)
        self.types = [k["type"] for k in keys_dict["keys"]]
        self.n_keys = len(self.types)
        self.is_black = np.array([t == "black" for t in self.types])
        # Camera-far direction (from calibration). Used in diff-mode to
        # break boundary-blob ties by cam side: a blob spanning two
        # adjacent keys goes to the cam-near neighbor (rightmost touched
        # for far_side="left" / right cam; leftmost for far_side="right"
        # / left cam). Replaces area-based argmax assignment.
        self._far_side = str(keys_dict.get("far_side", "right"))

        self.smooth_window = max(1, smooth_window)
        self.debounce = max(1, debounce)
        self.slope_n_sigma = slope_n_sigma
        self.tempdiff_n_sigma = tempdiff_n_sigma
        self.margin_black = margin_black
        self.margin_white = margin_white

        # Diff-only mode: uses press_diff.detect_press_regions
        # (frame-diff threshold + top-frame blob-presence filter) as the
        # raw activation mask, then scores per-key by the fraction of
        # each polygon's safe-mask that the activation covers (after
        # hand-mask exclusion). Bypasses the 4 channels above when True.
        self.use_diff_only = use_diff_only
        # press_diff filter is strict (threshold=75 + BLOB_TOP_THRESH=3 in
        # press_diff.py) AND we have no chaos baseline in this mode — any
        # surviving activation inside a key polygon (after hand-mask
        # exclusion) is treated as a press. Per-key score is the absolute
        # count of activated pixels assigned to that key by key_id_map.
        # Uniform threshold across key types because the signal is
        # whatever survives the strict filter, not a fraction of polygon.
        self._diff_min_blob_area = 1       # 1 = disabled (no CC filter)
        self._diff_press_pixel_count = 20  # per-key activated-pixel threshold
        # Boundary margin (px) for diff-mode pixel→key assignment.
        # ZERO by default: no erosion, no inter-key gap. Boundary-spanning
        # blobs are resolved by the camera-side rule in _process_diff
        # (right cam → rightmost touched key; left cam → leftmost) — the
        # geometric decision is which cam sees the keyboard from which
        # side, so we don't need a buffer ribbon.
        self._diff_boundary_margin = 0
        # Frame-to-frame motion supplement to the MP hand mask. Catches
        # fast sweeps where MP loses tracking. Threshold is on absolute
        # gray-channel delta; press onset is small (<20), hand sweep is
        # large (>50). 0 to disable.
        self._motion_supp_enabled = False
        self._motion_supp_threshold = 80
        self._diff_key_id_map = self._build_diff_key_id_map(
            self._diff_boundary_margin
        )
        self._diff_buf = np.zeros((self.smooth_window, self.n_keys), dtype=np.float32)
        # Per-stage viz attributes for the playback warp_lines panels.
        self._last_diff_raw_mask = None        # press_diff pre-hand-exclusion mask (uint8 0/255)
        self._last_diff_post_hand_mask = None  # post-hand-exclusion mask (uint8 0/255)
        self._last_diff_counted_mask = None    # post-hand AND inside eroded polygon — actually counted
        self._last_hand_viz = None             # warped + orange-skin + magenta contour
        self._last_diff_overlay = None         # per-key overlay with red activation + polys + scores
        self._last_diff_scores = None          # smoothed per-key counts

        # Smoothing buffers (per-channel rolling window).
        self._line_buf = np.zeros((self.smooth_window, self.n_keys), dtype=np.float32)
        self._bright_buf = np.full(
            (self.smooth_window, self.n_keys), np.nan, dtype=np.float32
        )
        self._slope_buf = np.full(
            (self.smooth_window, self.n_keys), np.nan, dtype=np.float32
        )
        self._tempdiff_buf = np.full(
            (self.smooth_window, self.n_keys), np.nan, dtype=np.float32
        )
        self._buf_idx = 0
        self._buf_filled = 0

        # Debounce counter (consecutive frames flagged above threshold per key).
        self._above_count = np.zeros(self.n_keys, dtype=np.int32)

        # Channel state — baselines / thresholds, set by setters.
        self._line_thresholds = None     # per-key threshold for line channel
        self._bright_baseline = None     # per-key rest mean intensity
        self._bright_thresholds = None   # per-key |delta| threshold
        self._slope_baseline_mean = None
        self._slope_baseline_std = None
        self._rest_mean_frame = None     # warped grayscale, for tempdiff + motion
        self._rest_mean_bgr = None       # warped BGR, for color-ratio shadow discriminator
        self._tempdiff_chaos_mean = None
        self._tempdiff_chaos_std = None

        # Hand mask state.
        self._hand_persistence = None
        self._prev_gray = None
        self._lsd = cv2.createLineSegmentDetector()

        # MOG2 background subtractor channel: trained on rest + chaos so
        # rest-state, hand-passes, and shadows are all part of the
        # learned background. After training (frozen via learningRate=0),
        # ONLY truly novel pixel values fire as foreground — i.e. press
        # events. Per-polygon foreground count becomes a press signal
        # robust to shadow + hand variation by construction.
        self._mog2 = None
        self._mog2_trained = False
        # Per-key chaos baseline: an ABSOLUTE press threshold derived
        # from chaos noise — typically the upper percentile of per-key
        # MOG2 scores during the chaos clip (with hand mask applied,
        # same as inference). Set once after train_mog2 via
        # set_chaos_baseline_from_scores. Persistent-flicker regions
        # self-calibrate to a higher threshold and don't fire as presses.
        self._mog_per_key_threshold = None
        # Per-PIXEL persistent-flicker mask. Computed during chaos
        # baseline: pixels that fire as MOG2 foreground (after hand
        # masking) in >MOG_PIXEL_FLICKER_THRESHOLD fraction of chaos
        # frames are zeroed in the MOG2 mask before scoring and viz.
        # Specifically targets the constant-edge-blob phenomenon where
        # the same patch of pixels persistently triggers across the
        # entire clip due to lighting drift / specular / mode-tightness.
        self._mog_persistent_flicker_mask = None
        # Runtime online flicker tracker: per-pixel exponential moving
        # average of foreground state. Catches blobs that are NEW at
        # inference time (lighting drift, etc.) without needing them
        # to have appeared in chaos. Updated each mog2_score_keys call.
        self._mog_runtime_flicker_ema = None
        # Rolling history of recent per-key MOG scores. Used as a
        # temporal-derivative gate: presses have a sharp onset spike;
        # constant-flicker regions have a flat score over time.
        self._mog_score_history = None
        self._mog_score_history_idx = 0
        self._mog_score_history_filled = 0

        # Cache an EXTENDED warp transform: same top corners as the
        # keyboard warp, but bottom corners pushed down into the source
        # frame. The rectified extended view shows keyboard + below in
        # natural perspective — much friendlier to MediaPipe than the
        # raw skewed source frame. Landmarks detected on this extended
        # view are projected back to original-warp coords for the hand
        # mask.
        src_corners = np.array(
            keys_dict["warp"]["corners_tl_tr_br_bl"], dtype=np.float32
        )  # TL, TR, BR, BL
        self._src_corners = src_corners
        W, H = keys_dict["warp"]["out_size"]
        # Extend bottom corners DOWN in source-y by half the keyboard
        # vertical extent (heuristic — captures the area below where
        # hands typically enter for a front-facing camera).
        kb_height = float(
            max(src_corners[2][1], src_corners[3][1])
            - min(src_corners[0][1], src_corners[1][1])
        )
        # Extension below the keyboard for MediaPipe context. 2× kb_height
        # is a reasonable default — captures hand-entry area for most
        # camera placements without going past the source-frame bottom
        # (which just shows black bars). For recordings where the camera
        # captures more below the keyboard, this could be increased; for
        # tighter shots, lower it.
        extension_src_y = kb_height * 2.0
        ext_src = src_corners.copy()
        ext_src[2, 1] += extension_src_y  # BR_y
        ext_src[3, 1] += extension_src_y  # BL_y
        # Extended dst rectangle: keyboard rows preserved at top, below-
        # area appended at proportional resolution.
        H_ext = int(H + extension_src_y * (H / max(kb_height, 1.0)))
        ext_dst = np.array(
            [[0, 0], [W - 1, 0], [W - 1, H_ext - 1], [0, H_ext - 1]],
            dtype=np.float32,
        )
        self._mp_ext_W = W
        self._mp_ext_H = H_ext
        self._mp_ext_M = cv2.getPerspectiveTransform(ext_src, ext_dst)
        # Project from extended-warp coords back to original-warp coords.
        # Pre-compute via: ext_warp → source → original_warp.
        M_orig = cv2.getPerspectiveTransform(
            src_corners, np.array(
                [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                dtype=np.float32,
            ),
        )
        M_ext_inv = np.linalg.inv(self._mp_ext_M)
        self._mp_ext_to_orig = M_orig @ M_ext_inv

        # MediaPipe-based hand mask (color-independent, robust to skin
        # tone, much cleaner than heuristics). Runs on the SOURCE frame
        # (not warped), then projects hand-landmark convex hull into
        # warped space using the calibration's perspective transform M.
        self.use_mediapipe = (
            use_mediapipe and _MP_AVAILABLE and _MP_MODEL_PATH.exists()
        )
        self._mp_landmarker = None
        self._source_frame = None  # caller sets via set_source_frame() each frame
        self._last_mp_ext_viz = None  # extended-warp debug image with MP overlay
        # Unconditional fingertip caps (uint8 0/255 in warped coords).
        # Built per-frame from MP fingertip landmarks; OR'd into the hand
        # mask after the HSV evidence gate so tip pixels are guaranteed
        # to be masked even when V transitions sharply at the
        # finger-on-key boundary fail valid_hue.
        self._last_fingertip_caps = None
        if self.use_mediapipe:
            options = mp_vision.HandLandmarkerOptions(
                base_options=_MPBaseOptions(model_asset_path=str(_MP_MODEL_PATH)),
                num_hands=2,
                min_hand_detection_confidence=0.2,   # was 0.5
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.2,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._mp_landmarker = mp_vision.HandLandmarker.create_from_options(options)
        elif use_mediapipe:
            print("[Detector] mediapipe requested but unavailable: "
                  f"installed={_MP_AVAILABLE}  model_exists={_MP_MODEL_PATH.exists()}")

    # ── Setters / channel enablement ─────────────────────────────────────

    def set_line_thresholds(self, thresholds: np.ndarray):
        """Per-key length-threshold for the line channel (chaos-derived)."""
        self._line_thresholds = thresholds.astype(np.float32)

    def set_rest_mean_frame(self, mean_gray: np.ndarray, mean_bgr: np.ndarray | None = None):
        """Rest-baseline grayscale image for the warped strip. Optionally
        also store the BGR rest baseline for the shadow-vs-color
        discriminator in the hand mask."""
        self._rest_mean_frame = mean_gray
        if mean_bgr is not None:
            self._rest_mean_bgr = mean_bgr

    def set_brightness_baseline(self, baseline: np.ndarray):
        """Per-key rest-mean intensity (skin-masked)."""
        self._bright_baseline = baseline

    def set_brightness_thresholds(self, thresholds: np.ndarray):
        """Per-key |delta| threshold for the brightness channel."""
        self._bright_thresholds = thresholds

    def set_slope_baseline(self, mean: np.ndarray, std: np.ndarray):
        """Per-key rest-mean line angle + std."""
        self._slope_baseline_mean = mean
        self._slope_baseline_std = np.maximum(std, 1.0)

    def set_tempdiff_chaos_stats(self, mean: np.ndarray, std: np.ndarray):
        """Per-key chaos-noise-floor mean + std for tempdiff."""
        self._tempdiff_chaos_mean = mean
        self._tempdiff_chaos_std = np.maximum(std, 1.0)

    def set_margin_black(self, m: float):
        self.margin_black = m

    def set_margin_white(self, m: float):
        self.margin_white = m

    def set_diff_thresholds(self, pixel_count: int, min_blob_area: int):
        """Diff-mode thresholds: per-key activated-pixel count to fire a
        press, and the connected-component area floor for the post-hand
        activation mask (drops sub-pixel noise)."""
        self._diff_press_pixel_count = max(1, int(pixel_count))
        self._diff_min_blob_area = max(1, int(min_blob_area))

    def set_source_frame(self, source_bgr: np.ndarray):
        """Cache the current source (pre-warp) frame for MediaPipe hand
        detection. Caller passes this every frame BEFORE process()."""
        self._source_frame = source_bgr

    # ── MOG2 press-detection hyperparameters ──────────────────────────
    # Connected-component shape filter: reject thin slivers and tiny noise.
    MOG_BLOB_MIN_THICKNESS = 5   # min(width, height) of a foreground blob
    MOG_BLOB_MIN_AREA = 25       # minimum pixel count
    # Edge-of-warp border to zero out (warp-interpolation artifacts).
    MOG_EDGE_BORDER = 5
    # Score history depth for the temporal-derivative gate.
    MOG_HISTORY_LEN = 5
    # Per-key chaos baseline is computed from a HIGH PERCENTILE of chaos
    # scores (mean+std distorts under right-skewed distributions — chaos
    # frames are mostly quiet with occasional hand-mask-leak spikes).
    # Press threshold = max(percentile * MULT, FLOOR).
    MOG_BASELINE_PERCENTILE = 90.0
    MOG_BASELINE_MULT = 1.3
    # Floor used when baseline is zero/tiny (quiet keys never fire in chaos).
    MOG_FLOOR_THRESHOLD = 25.0
    # Required score-rise over rolling-history mean — kills persistent flat
    # flicker that's already baked into baseline.
    MOG_DERIV_DELTA = 8.0
    # Per-pixel persistent-flicker threshold (fraction of chaos frames a
    # pixel fires as foreground after hand masking). Pixels above this
    # are permanently zeroed in inference (kill the persistent blob).
    MOG_PIXEL_FLICKER_THRESHOLD = 0.10
    # Online (runtime) per-pixel flicker EMA. Catches blobs that are NEW
    # in the inference clip (e.g. lighting drift between training and
    # play). Each frame: ema = (1-alpha)*ema + alpha*(fg pixel?), then
    # Gaussian-blurred so REGION-level (not per-pixel) flicker is caught
    # — important when the blob's active pixels jitter frame-to-frame.
    # alpha=0.01 → ~100-frame window (~3.3s at 30fps). Threshold 0.40 on
    # the blurred EMA catches regions where ~40% of nearby pixels fire
    # ~40% of the time. A held key is briefer + more concentrated, so
    # single presses don't get masked.
    MOG_RUNTIME_FLICKER_ALPHA = 0.01
    MOG_RUNTIME_FLICKER_THRESHOLD = 0.40
    MOG_RUNTIME_FLICKER_BLUR_KSIZE = 9

    def train_mog2(self, train_warped_frames):
        """Train MOG2 on a sequence of warped frames (rest + chaos).
        After training, MOG2 has multi-modal background per pixel —
        rest state, hand-passing modes are all 'background'. Subsequent
        applies use learningRate=0 (frozen) so only novel pixel values
        (real presses) fire as foreground.

        Hyperparameter notes:
          - history = exact training length so all training frames count
            equally toward the learned modes (default 500 truncates).
          - nmixtures = 7 (default 5) — wider mode pool absorbs more
            lighting / specular variations as background.
          - detectShadows = True — MOG2's built-in shadow class marks
            shadow pixels as 127 (cyan in viz) so they're trivially
            excluded from press scoring (we filter for ==255 only).
            Without this, shadows fire as full foreground and get
            counted toward presses → shadow false-positives.
        """
        n_frames = max(len(train_warped_frames), 100)
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=n_frames,
            varThreshold=16,
            detectShadows=True,
        )
        self._mog2.setNMixtures(7)
        for fr in train_warped_frames:
            self._mog2.apply(fr)
        self._mog2_trained = True
        # Reset baseline + history; they belong to this model instance.
        self._mog_per_key_threshold = None
        self._mog_persistent_flicker_mask = None
        self._mog_runtime_flicker_ema = None
        self._mog_score_history = None
        self._mog_score_history_idx = 0
        self._mog_score_history_filled = 0

    def mog2_apply(self, warped_bgr):
        """Apply frozen MOG2 to a warped frame. Returns binary mask
        (0=bg, 255=foreground) or None if not trained. Caller should
        call this AT MOST ONCE per frame — each apply consumes a step
        internally even with learningRate=0."""
        if not self._mog2_trained:
            return None
        return self._mog2.apply(warped_bgr, learningRate=0)

    def mog2_score_keys(self, mog_mask, hand_mask=None):
        """Per-polygon shape-filtered foreground area.

        Returns (scores, cleaned_mask):
          scores       — float32 array, per-key sum of pixel area belonging
                         to chunky foreground blobs (thin slivers rejected).
          cleaned_mask — uint8 copy of mog_mask with hand-pixels and
                         edge-border zeroed (suitable for visualization).

        A blob counts toward a polygon's score only if it satisfies
        MOG_BLOB_MIN_THICKNESS and MOG_BLOB_MIN_AREA. Shadow class (127)
        is ignored — only true foreground (==255) counts.
        """
        if mog_mask is None:
            return np.zeros(self.n_keys, dtype=np.float32), None
        cleaned = mog_mask.copy()
        if hand_mask is not None and hand_mask.shape == mog_mask.shape:
            cleaned[hand_mask > 0] = 0
        # Zero permanently-flicker pixels (computed during chaos baseline).
        # This is what visually kills the persistent-edge-blob: those
        # pixels fired in many chaos frames despite hand masking, so
        # they're treated as "always-noisy" and never count toward
        # press scoring.
        if (
            self._mog_persistent_flicker_mask is not None
            and self._mog_persistent_flicker_mask.shape == mog_mask.shape
        ):
            cleaned[self._mog_persistent_flicker_mask > 0] = 0
        b = self.MOG_EDGE_BORDER
        if b > 0:
            cleaned[:b, :] = 0
            cleaned[-b:, :] = 0
            cleaned[:, :b] = 0
            cleaned[:, -b:] = 0
        # Runtime online flicker EMA. Update BEFORE applying the runtime
        # mask, then use the updated EMA to mask pixels for THIS frame.
        # This catches blobs that are NEW at inference (e.g. lighting
        # drift between training and play) by observing per-pixel
        # foreground state over a rolling ~50-frame window. Pixels
        # whose foreground rate exceeds the threshold are zeroed.
        if (
            self._mog_runtime_flicker_ema is None
            or self._mog_runtime_flicker_ema.shape != mog_mask.shape
        ):
            self._mog_runtime_flicker_ema = np.zeros(
                mog_mask.shape, dtype=np.float32
            )
        fg_now = (cleaned == 255).astype(np.float32)
        alpha = self.MOG_RUNTIME_FLICKER_ALPHA
        self._mog_runtime_flicker_ema = (
            (1.0 - alpha) * self._mog_runtime_flicker_ema + alpha * fg_now
        )
        # Gaussian-blur so jittery-but-region-persistent flicker is caught.
        k = self.MOG_RUNTIME_FLICKER_BLUR_KSIZE
        blurred_ema = cv2.GaussianBlur(
            self._mog_runtime_flicker_ema, (k, k), 0
        )
        runtime_flicker = (
            blurred_ema > self.MOG_RUNTIME_FLICKER_THRESHOLD
        )
        if runtime_flicker.any():
            cleaned[runtime_flicker] = 0
        fg = (cleaned == 255).astype(np.uint8)
        scores = np.zeros(self.n_keys, dtype=np.float32)
        min_thick = self.MOG_BLOB_MIN_THICKNESS
        min_area = self.MOG_BLOB_MIN_AREA
        for ki, m in enumerate(self.det_state["per_key_mask"]):
            fg_in = fg & m
            if not fg_in.any():
                continue
            n_lbl, _, st, _ = cv2.connectedComponentsWithStats(
                fg_in, connectivity=8
            )
            s = 0.0
            for ci in range(1, n_lbl):
                _, _, bw, bh, area = st[ci]
                if min(int(bw), int(bh)) < min_thick:
                    continue
                if area < min_area:
                    continue
                s += float(area)
            scores[ki] = s
        return scores, cleaned

    def set_persistent_flicker_mask(self, mask: np.ndarray):
        """Per-pixel mask of always-noisy locations (computed during
        chaos baseline). Pixels with mask>0 will be treated as 'never
        foreground' regardless of MOG2's classification — kills the
        persistent-edge-blob and similar always-firing regions."""
        self._mog_persistent_flicker_mask = mask.astype(np.uint8)

    def set_chaos_baseline_from_scores(self, scores_NK: np.ndarray):
        """Compute per-key absolute press threshold from chaos scores.

        scores_NK shape: (n_chaos_frames, n_keys). For each key, takes
        the configured upper percentile (default 90th) of its scores —
        robust to right-skew from occasional hand-mask leak — and
        multiplies by MOG_BASELINE_MULT (default 1.3) to get the press
        threshold. Floored at MOG_FLOOR_THRESHOLD so quiet keys still
        require enough signal to fire.
        """
        if scores_NK is None or scores_NK.size == 0:
            self._mog_per_key_threshold = None
            return
        pct = float(self.MOG_BASELINE_PERCENTILE)
        per_key_pct = np.percentile(scores_NK, pct, axis=0)
        thr = per_key_pct * float(self.MOG_BASELINE_MULT)
        thr = np.maximum(thr, float(self.MOG_FLOOR_THRESHOLD))
        self._mog_per_key_threshold = thr.astype(np.float32)

    def mog2_press_set(self, scores: np.ndarray) -> set:
        """Convert per-key scores to a press set with two gates:
          1) Absolute threshold = max(baseline_mean + SIGMA*baseline_std,
             FLOOR_THRESHOLD). Adapts to per-key persistent flicker.
          2) Temporal derivative gate: current score must exceed the
             rolling-history mean by DERIV_DELTA. Kills flat-line
             flicker that already shows up in the baseline.
        Both gates must pass."""
        # Lazy-init history buffer (n_keys known at construction time).
        if self._mog_score_history is None:
            self._mog_score_history = np.zeros(
                (self.MOG_HISTORY_LEN, self.n_keys), dtype=np.float32
            )
            self._mog_score_history_idx = 0
            self._mog_score_history_filled = 0

        # Use history BEFORE this frame's score is pushed in.
        if self._mog_score_history_filled > 0:
            n_prev = self._mog_score_history_filled
            prev_mean = self._mog_score_history[:n_prev].mean(axis=0)
        else:
            prev_mean = np.zeros(self.n_keys, dtype=np.float32)
        deriv = scores - prev_mean

        if self._mog_per_key_threshold is not None:
            base_thr = self._mog_per_key_threshold
        else:
            base_thr = np.full(
                self.n_keys, self.MOG_FLOOR_THRESHOLD, dtype=np.float32
            )

        press_mask = (scores > base_thr) & (deriv > self.MOG_DERIV_DELTA)
        press_set = {int(i) for i in np.where(press_mask)[0]}

        # Push current scores into history (ring buffer).
        self._mog_score_history[self._mog_score_history_idx] = scores
        self._mog_score_history_idx = (
            (self._mog_score_history_idx + 1) % self.MOG_HISTORY_LEN
        )
        self._mog_score_history_filled = min(
            self._mog_score_history_filled + 1, self.MOG_HISTORY_LEN
        )
        return press_set

    def mog2_threshold_for(self, ki: int) -> float:
        """Effective per-key absolute threshold (baseline-derived). Useful
        for diagnostics / overlay text."""
        if self._mog_per_key_threshold is None:
            return float(self.MOG_FLOOR_THRESHOLD)
        return float(self._mog_per_key_threshold[ki])

    def reset_smoothing(self):
        self._line_buf[:] = 0
        self._bright_buf[:] = np.nan
        self._slope_buf[:] = np.nan
        self._tempdiff_buf[:] = np.nan
        self._buf_idx = 0
        self._buf_filled = 0
        self._above_count[:] = 0

    # ── Hand mask ────────────────────────────────────────────────────────

    # MediaPipe Hand Landmarker connection topology (21 landmarks per hand).
    _MP_HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # index
        (5, 9), (9, 10), (10, 11), (11, 12),    # middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # ring
        (13, 17), (17, 18), (18, 19), (19, 20), # pinky
        (0, 17),                                # palm wrist-pinky base
    )
    _MP_PALM_INDICES = (0, 1, 5, 9, 13, 17)
    # Thumb-tip, index-tip, middle-tip, ring-tip, pinky-tip. MP often
    # locates these at the nail joint rather than the fleshy fingertip,
    # so the actual press contact pixel can sit a few px past the
    # landmark — we cap each tip with an unconditional circle in the
    # hand mask to guarantee fingertip coverage past MP's reported tip.
    _MP_FINGERTIP_INDICES = (4, 8, 12, 16, 20)
    # Radius of unconditional fingertip cap (no HSV gate). ~25 px covers
    # the fleshy tip past MP's nail-joint landmark on typical warp sizes
    # (640-1280 wide). Tunable; larger = safer at cost of masking more
    # adjacent key-surface pixels (which lose press-signal contribution).
    _FINGERTIP_CAP_RADIUS = 35
    # Filled-circle cap at the wrist landmark (index 0). Roughly 2× the
    # fingertip radius — wrist + forearm extends far past the convex hull
    # of the 21 hand landmarks and routinely leaks through without it.
    _WRIST_CAP_RADIUS = 60
    # Filled-circle cap at every one of the 21 landmarks. Thickens the
    # finger-skeleton beyond just the bone lines + convex hull, so that
    # the silhouette of each finger is reliably masked even at narrow
    # joints / splayed-finger poses.
    _JOINT_CAP_RADIUS = 20
    # Bone-line thickness for the skeleton render. Larger = more finger
    # coverage before final dilation.
    _BONE_THICKNESS = 8

    def _mediapipe_search_region(self, warped_shape):
        """Returns a generous binary search region around hand bones in
        warped space. To improve MediaPipe detection rate, run MP on a
        focused ROI that includes the area BELOW the keyboard (where
        hands enter the frame) plus a strip of keyboard for context —
        far better recall than running on the full source frame where
        the keyboard dominates and hands are small/partial.
        """
        if (
            self._mp_landmarker is None
            or self._source_frame is None
            or self._source_frame.size == 0
        ):
            return None
        h_src, w_src = self._source_frame.shape[:2]
        # Run MP on the EXTENDED RECTIFIED WARP (keyboard + below-area
        # rectified together). This presents hands in natural perspective
        # — much higher detection recall than the raw skewed source view.
        ext_warp = cv2.warpPerspective(
            self._source_frame,
            self._mp_ext_M,
            (self._mp_ext_W, self._mp_ext_H),
        )
        rgb = cv2.cvtColor(ext_warp, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._mp_landmarker.detect(mp_image)
        # Build the debug viz: extended warp with MP overlay drawn on top.
        # The original keyboard's bottom edge is NOT a horizontal line in
        # extended-warp coords (perspective transforms preserve lines but
        # not orientation). Project the source-space keyboard corners
        # through M_ext to get the actual keyboard quad in extended coords.
        ext_viz = ext_warp.copy()
        kb_corners_in_ext = cv2.perspectiveTransform(
            self._src_corners.reshape(-1, 1, 2),
            self._mp_ext_M,
        ).reshape(-1, 2).astype(np.int32)
        cv2.polylines(
            ext_viz, [kb_corners_in_ext], True, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            ext_viz, "keyboard region (red)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            ext_viz, "keyboard region (red)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
        )
        if result.hand_landmarks:
            for lms in result.hand_landmarks:
                pts = [(int(lm.x * self._mp_ext_W), int(lm.y * self._mp_ext_H))
                       for lm in lms]
                for a, b in self._MP_HAND_CONNECTIONS:
                    cv2.line(ext_viz, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(ext_viz, p, 4, (0, 0, 255), -1, cv2.LINE_AA)
        self._last_mp_ext_viz = ext_viz
        if not result.hand_landmarks:
            return None
        H, W = warped_shape
        skeleton = np.zeros((H, W), dtype=np.uint8)
        fingertip_caps = np.zeros((H, W), dtype=np.uint8)
        for hand_landmarks in result.hand_landmarks:
            # MP returns landmarks normalized to extended-warp dims.
            # Multiply by extended-warp size to get extended-warp pixel
            # coords, then apply ext→orig transform to get original-warp
            # coords. Landmarks below the keyboard project to negative
            # y in original-warp coords (out of the strip) and get
            # clipped naturally when drawn.
            pts_ext = np.array(
                [[lm.x * self._mp_ext_W, lm.y * self._mp_ext_H]
                 for lm in hand_landmarks],
                dtype=np.float32,
            )
            pts_warped = cv2.perspectiveTransform(
                pts_ext.reshape(-1, 1, 2), self._mp_ext_to_orig,
            ).reshape(-1, 2).astype(np.int32)
            # Skeleton bones — thick enough that the bones alone cover
            # most of each finger's actual silhouette before any further
            # dilation.
            for a, b in self._MP_HAND_CONNECTIONS:
                cv2.line(
                    skeleton, tuple(pts_warped[a]), tuple(pts_warped[b]),
                    255, thickness=self._BONE_THICKNESS, lineType=cv2.LINE_AA,
                )
            # Joint caps at every one of the 21 landmarks. Filled circles
            # plug any gaps the bone lines + convex hull leave at sharp
            # finger bends and around small joints.
            for ji in range(len(pts_warped)):
                cv2.circle(
                    skeleton, tuple(pts_warped[ji]),
                    self._JOINT_CAP_RADIUS,
                    255, thickness=-1, lineType=cv2.LINE_AA,
                )
            # Convex hull of all 21 landmarks fills the inter-finger webbing
            # and palm interior naturally — strictly more inclusive than
            # the bone-line skeleton + 6-vertex palm polygon, especially
            # for splayed-finger poses where the bone lines miss the
            # space between fingers entirely.
            hull = cv2.convexHull(pts_warped.reshape(-1, 1, 2))
            cv2.fillConvexPoly(skeleton, hull, 255)
            palm_pts = pts_warped[list(self._MP_PALM_INDICES)]
            cv2.fillPoly(skeleton, [palm_pts], 255)
            # Unconditional fingertip caps. Filled circle at each tip
            # landmark so the actual fleshy fingertip (which usually
            # extends a few px past MP's reported tip location at the
            # nail joint) is guaranteed to be masked, even when the HSV
            # evidence gate would have rejected those pixels at the
            # finger/key boundary.
            for ti in self._MP_FINGERTIP_INDICES:
                cv2.circle(
                    fingertip_caps,
                    tuple(pts_warped[ti]),
                    self._FINGERTIP_CAP_RADIUS,
                    255, thickness=-1, lineType=cv2.LINE_AA,
                )
            # Wrist cap. The wrist landmark (index 0) is at the topology
            # edge — convex hull rarely extends much past it, so wrist /
            # forearm pixels routinely leak past the mask. A generous
            # filled circle (~2× fingertip radius) covers the bulk of
            # wrist + lower palm reliably.
            cv2.circle(
                skeleton,
                tuple(pts_warped[0]),
                self._WRIST_CAP_RADIUS,
                255, thickness=-1, lineType=cv2.LINE_AA,
            )
        self._last_fingertip_caps = fingertip_caps
        # Dilate to a generous search region (covers any plausible finger
        # / palm / wrist thickness for the camera angle without committing
        # to a specific number).
        return cv2.dilate(
            skeleton, cv2.getStructuringElement(cv2.MORPH_RECT, (65, 65))
        )

    def _compute_hand_mask(self, warped_bgr: np.ndarray, warped_gray: np.ndarray):
        """Hand mask. When MediaPipe is enabled, trust it: if MP doesn't
        detect hands this frame, NO mask is applied (which is correct —
        no hands present, no need to mask anything). Brief persistence
        (~5 frames) keeps the mask alive across single-frame MP misses
        from low-confidence detection or occlusion.

        When MediaPipe is disabled, falls back to the saturation-increase
        heuristic (kept as a non-ML path; less reliable, color-dependent).
        """
        # ── MediaPipe search-region + per-pixel test (data-driven width) ──
        # MP skeleton provides LOCATION; per-pixel rest-diff + saturation-
        # increase tests determine which pixels are actually hand. Width
        # of the mask comes from real image data, not a hardcoded thickness.
        # When MP misses (e.g., only a fingertip is visible), a fallback
        # finds skin-like blobs of fingertip-plausible size.
        if self.use_mediapipe:
            if (
                self._hand_persistence is None
                or self._hand_persistence.shape != warped_gray.shape
            ):
                self._hand_persistence = np.zeros(warped_gray.shape, dtype=np.uint8)
            self._hand_persistence[self._hand_persistence > 0] -= 1

            # Per-pixel HSV decomposition: classify each foreground pixel
            # as HAND (color shifted) or SHADOW (only V dropped, H/S
            # unchanged). Pressed-key pixels have small ΔV with no color
            # shift → fail both, flow to detection channels normally.
            hand_evidence = None
            if (
                self._rest_mean_frame is not None
                and self._rest_mean_bgr is not None
            ):
                hsv_curr = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
                hsv_rest = cv2.cvtColor(self._rest_mean_bgr, cv2.COLOR_BGR2HSV)
                # Hue is circular; use wraparound distance (0..180 in OpenCV).
                h_diff = np.minimum(
                    np.abs(hsv_curr[..., 0].astype(np.int16) -
                           hsv_rest[..., 0].astype(np.int16)),
                    180 - np.abs(hsv_curr[..., 0].astype(np.int16) -
                                 hsv_rest[..., 0].astype(np.int16)),
                )
                s_diff = (
                    hsv_curr[..., 1].astype(np.int16)
                    - hsv_rest[..., 1].astype(np.int16)
                )
                v_diff = (
                    hsv_curr[..., 2].astype(np.int16)
                    - hsv_rest[..., 2].astype(np.int16)
                )
                # HSV gating: hue is mathematically undefined for dark or
                # unsaturated pixels. Black keys have V<40 + occasional
                # specular highlights up to V≈60 with noisy hue. Tightening
                # both V floor (80) and adding a |ΔV| significance test
                # eliminates black-key flicker — a real hand replacing a
                # key makes V jump by ~100; noise/specular shifts are <10.
                valid_hue = (
                    (hsv_curr[..., 2] > 80) & (hsv_rest[..., 2] > 60)
                    & ((hsv_curr[..., 1] > 25) | (hsv_rest[..., 1] > 25))
                )
                v_significant = np.abs(v_diff) > 30
                # HAND: meaningful color shift in valid hue territory AND
                # significant brightness change.
                hand_evidence = (
                    ((h_diff > HAND_DELTA_H_T) | (s_diff > HAND_DELTA_S_T))
                    & valid_hue
                    & v_significant
                )
                # SHADOW: V dropped a lot AND hue/sat barely changed.
                shadow_evidence = (
                    (v_diff < -SHADOW_DELTA_V_T)
                    & (h_diff < SHADOW_HSV_TOL_H)
                    & (np.abs(s_diff) < SHADOW_HSV_TOL_S)
                )
                # Explicit subtraction: even if a pixel barely passes
                # hand_evidence, force-exclude it if the shadow signature
                # is stronger (uniform darkening).
                hand_evidence = hand_evidence & ~shadow_evidence

            self._last_fingertip_caps = None  # cleared each frame; reset by MP if it detects
            search = self._mediapipe_search_region(warped_gray.shape)
            current = np.zeros(warped_gray.shape, dtype=bool)

            if search is not None and np.any(search):
                # MP detected: keep pixels in search region that pass evidence.
                if hand_evidence is not None:
                    current = (search > 0) & hand_evidence
                else:
                    current = (search > 0)
                # Fingertip caps are unconditional: OR them in AFTER the
                # HSV gate so tip pixels survive even when valid_hue or
                # the |ΔV|>30 test would have rejected the boundary
                # transition between finger and key.
                if self._last_fingertip_caps is not None:
                    current = current | (self._last_fingertip_caps > 0)
            elif hand_evidence is not None:
                # MP missed: fingertip-fallback. Find sat/rest-diff blobs
                # of fingertip-plausible size (50-2000 px) anywhere.
                ev_u8 = hand_evidence.astype(np.uint8) * 255
                ev_dilated = cv2.dilate(
                    ev_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                )
                n_lab, lbl, st, _ = cv2.connectedComponentsWithStats(
                    ev_dilated, connectivity=8
                )
                for ci in range(1, n_lab):
                    area = st[ci, cv2.CC_STAT_AREA]
                    if 50 < area < 2000:
                        current |= (lbl == ci) & hand_evidence
            if np.any(current):
                # Longer TTL bridges MP miss-frames: when MP detects on
                # frame N but misses on N+1..N+5, the mask stays alive.
                self._hand_persistence[current] = 8

            # Motion supplement: catches fast hand sweeps that MP can't
            # track in time. Frame-to-frame |Δgray|>T is hand motion (real
            # press onset is gradual, ~5–15 levels/frame; hand entering/
            # leaving a pixel jumps 50+ levels). Add motion pixels to the
            # persistence map at the same TTL as MP-detected hand. Press
            # signal is preserved because it doesn't trip the threshold;
            # debounce absorbs the 1–2-frame delay during press onset.
            if (
                self._motion_supp_enabled
                and self._prev_gray is not None
                and self._prev_gray.shape == warped_gray.shape
            ):
                motion = cv2.absdiff(warped_gray, self._prev_gray)
                motion_mask = motion > self._motion_supp_threshold
                if motion_mask.any():
                    motion_u8 = motion_mask.astype(np.uint8) * 255
                    motion_u8 = cv2.dilate(
                        motion_u8,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                    )
                    self._hand_persistence = np.maximum(
                        self._hand_persistence,
                        (motion_u8 > 0).astype(np.uint8) * 8,
                    )
            return ((self._hand_persistence > 0).astype(np.uint8) * 255)

        # ── Heuristic fallback path (no ML): saturation-increase ────
        if self._rest_mean_frame is None or self._rest_mean_bgr is None:
            return np.zeros(warped_gray.shape, dtype=np.uint8)
        hsv_curr = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        hsv_rest = cv2.cvtColor(self._rest_mean_bgr, cv2.COLOR_BGR2HSV)
        v_diff = np.abs(
            hsv_curr[..., 2].astype(np.int16) - hsv_rest[..., 2].astype(np.int16)
        )
        sat_increase = (
            hsv_curr[..., 1].astype(np.int16) - hsv_rest[..., 1].astype(np.int16)
        )
        # Per-pixel candidate (catches hands AND press-edges).
        strong_fg = (v_diff > REST_V_DIFF_T) & (sat_increase > SAT_INCREASE_T)
        weak_fg = v_diff > REST_V_DIFF_T

        # BLOB-LEVEL filter BEFORE persistence: classify each connected
        # component as "hand" (large + chunky) or "press-edge" (small or
        # thin). Press-edge blobs are dropped from strong_fg, so they
        # never enter persistence and can't propagate forward as hand.
        sf_u8 = strong_fg.astype(np.uint8) * 255
        sf_dilated = cv2.dilate(
            sf_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        n_pre, lbl_pre, st_pre, _ = cv2.connectedComponentsWithStats(
            sf_dilated, connectivity=8
        )
        hand_blob_mask = np.zeros(n_pre, dtype=bool)
        for ci in range(1, n_pre):
            x, y, w_, h_, area = st_pre[ci]
            if area < HAND_BLOB_MIN_AREA:
                continue
            if min(int(w_), int(h_)) < HAND_BLOB_MIN_THICKNESS:
                continue
            hand_blob_mask[ci] = True
        # Strong_fg, but only pixels that belong to a hand-classified blob.
        strong_fg_filtered = strong_fg & hand_blob_mask[lbl_pre]

        if (self._hand_persistence is None
                or self._hand_persistence.shape != warped_gray.shape):
            self._hand_persistence = np.zeros(warped_gray.shape, dtype=np.uint8)
        self._hand_persistence[strong_fg_filtered] = PERSISTENCE_FRAMES
        decay = (~strong_fg_filtered) & (self._hand_persistence > 0)
        self._hand_persistence[decay] -= 1
        recent_hand = self._hand_persistence > 0
        mask_tight = (
            (strong_fg_filtered | (recent_hand & weak_fg))
            .astype(np.uint8) * 255
        )
        # Connected components on dilated mask, filter by tight-pixel shape.
        mask_grouped = cv2.dilate(
            mask_tight, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        )
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_grouped, connectivity=8
        )
        keep = np.zeros(n_labels, dtype=bool)
        tight_bool = mask_tight > 0
        for ci in range(1, n_labels):
            tight_in_comp = (labels == ci) & tight_bool
            tc = int(tight_in_comp.sum())
            if tc < MIN_TIGHT_AREA:
                continue
            ys, xs = np.where(tight_in_comp)
            tw = int(xs.max() - xs.min() + 1)
            th = int(ys.max() - ys.min() + 1)
            if min(tw, th) < MIN_TIGHT_THICKNESS:
                continue
            keep[ci] = True
        sk = np.where(keep[labels] & tight_bool, 255, 0).astype(np.uint8)
        sk = cv2.morphologyEx(
            sk, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        )
        return sk

    # ── Per-frame processing ─────────────────────────────────────────────

    def _build_diff_key_id_map(self, margin_px: int) -> np.ndarray:
        """Per-pixel key index, with a ``margin_px`` ribbon along every
        polygon boundary set to -1 so press pixels right at a seam don't
        get assigned to either neighbor."""
        src = self.det_state["key_id_map"]
        H, W = src.shape
        if margin_px <= 0:
            return src.copy()
        kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), dtype=np.uint8)
        new_map = np.full((H, W), -1, dtype=np.int32)
        for ki in range(self.n_keys):
            m = (src == ki).astype(np.uint8)
            eroded = cv2.erode(m, kernel)
            new_map[eroded > 0] = ki
        return new_map

    def set_diff_boundary_margin(self, margin_px: int):
        """Re-derive the diff-mode key id map with a new boundary margin."""
        self._diff_boundary_margin = max(0, int(margin_px))
        self._diff_key_id_map = self._build_diff_key_id_map(
            self._diff_boundary_margin
        )

    def _process_diff(self, warped_bgr, warped_gray, hand_mask):
        """Diff-only path. press_diff.detect_press_regions for the raw
        activation mask, MP hand-mask exclusion, then per-key fraction
        of safe_mask activated. Populates per-stage viz attributes for
        playback panels and returns (pressed_set, per_key_overlay_viz)."""
        # Hand-mask-only viz (warped + orange skin fill + magenta contour).
        hand_viz = warped_bgr.copy()
        if hand_mask is not None and np.any(hand_mask):
            orange = np.zeros_like(hand_viz)
            orange[hand_mask > 0] = (0, 140, 255)
            hand_viz = cv2.addWeighted(orange, 0.55, hand_viz, 1.0, 0)
            cnts, _ = cv2.findContours(
                (hand_mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(hand_viz, cnts, -1, (255, 0, 255), 2, cv2.LINE_AA)
        self._last_hand_viz = hand_viz

        if self._rest_mean_bgr is None:
            empty = np.zeros(self.n_keys, dtype=np.float32)
            self._diff_buf[self._buf_idx] = empty
            self._buf_idx = (self._buf_idx + 1) % self.smooth_window
            self._buf_filled = min(self._buf_filled + 1, self.smooth_window)
            self._last_diff_raw_mask = np.zeros_like(warped_gray)
            self._last_diff_post_hand_mask = np.zeros_like(warped_gray)
            self._last_diff_overlay = warped_bgr.copy()
            self._last_diff_scores = empty
            return set(), self._last_diff_overlay

        # press_diff filter: absdiff vs rest-mean → threshold@75 → top-frame blob filter.
        raw_act = detect_press_regions(warped_bgr, self._rest_mean_bgr)
        # Hard-chop everything below the top 15 % of the warp. press_diff's
        # filter_blobs_by_top_presence keeps WHOLE blobs that touch the
        # top band — including pixels below — which can leak press
        # signal into white-key seam areas. This drops everything below
        # the top band regardless of what those pixels are attached to.
        top_band = max(1, int(raw_act.shape[0] * 0.15))
        raw_act[top_band:, :] = 0
        self._last_diff_raw_mask = raw_act.copy()
        act_bool = raw_act > 0
        if hand_mask is not None and hand_mask.size:
            act_bool &= ~(hand_mask > 0)

        # Drop tiny stray components left after hand exclusion (single-
        # pixel speckle, edge slivers from imperfect masking).
        if self._diff_min_blob_area > 1 and act_bool.any():
            act_u8 = act_bool.astype(np.uint8) * 255
            n_lab, lbl, st, _ = cv2.connectedComponentsWithStats(
                act_u8, connectivity=8
            )
            keep = np.zeros(n_lab, dtype=bool)
            for ci in range(1, n_lab):
                if st[ci, cv2.CC_STAT_AREA] >= self._diff_min_blob_area:
                    keep[ci] = True
            act_bool = keep[lbl]

        self._last_diff_post_hand_mask = (act_bool.astype(np.uint8) * 255)

        # ── Per-key score via BLOB-TO-KEY assignment ─────────────────
        # Connected components on the post-hand mask. Each blob is
        # assigned to a SINGLE key by CAMERA-SIDE rule (geometric, not
        # area-based). For a right cam (far_side="left") the blob goes
        # to the rightmost touched key; for a left cam (far_side="right")
        # the leftmost. This leverages each cam's near-side polygon
        # accuracy and gives one press blob → one key, no cross-firing,
        # no lost pixels in any boundary gap.
        key_id = self._diff_key_id_map
        counts = np.zeros(self.n_keys, dtype=np.float32)
        if act_bool.any():
            n_lab, lbl = cv2.connectedComponents(
                act_bool.astype(np.uint8), connectivity=8,
            )
            pick_rightmost = (self._far_side == "left")
            for ci in range(1, n_lab):
                blob_mask = (lbl == ci)
                blob_ids = key_id[blob_mask]
                blob_ids = blob_ids[blob_ids >= 0]
                if blob_ids.size == 0:
                    continue
                per_key_in_blob = np.bincount(blob_ids, minlength=self.n_keys)
                touched = np.where(per_key_in_blob > 0)[0]
                if touched.size == 0:
                    continue
                best = int(touched.max() if pick_rightmost else touched.min())
                counts[best] += float(per_key_in_blob[best])
        # Counted-mask viz: pixels that survive both hand exclusion and
        # the boundary ribbon — the signal feeding the per-key argmax.
        counted_bool = act_bool & (key_id >= 0)
        self._last_diff_counted_mask = (
            counted_bool.astype(np.uint8) * 255
        )

        # Smoothing on the count.
        self._diff_buf[self._buf_idx] = counts
        self._buf_idx = (self._buf_idx + 1) % self.smooth_window
        self._buf_filled = min(self._buf_filled + 1, self.smooth_window)
        smoothed = self._diff_buf[: self._buf_filled].mean(axis=0)
        self._last_diff_scores = smoothed

        thr_const = float(self._diff_press_pixel_count)
        above = smoothed > thr_const
        self._above_count = np.where(above, self._above_count + 1, 0)
        pressed = {
            int(i) for i, c in enumerate(self._above_count) if c >= self.debounce
        }

        # Per-key overlay viz: warped + red activation + polygon outlines
        # + score annotations on top-3 keys.
        viz = warped_bgr.copy()
        red_overlay = np.zeros_like(viz)
        red_overlay[act_bool] = (0, 0, 255)
        viz = cv2.addWeighted(red_overlay, 0.55, viz, 1.0, 0)
        for ki, mask in enumerate(self.det_state["per_key_mask"]):
            cnts2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            col = (0, 200, 0) if self.types[ki] == "white" else (0, 100, 200)
            cv2.drawContours(viz, cnts2, -1, col, 1, cv2.LINE_AA)
        if smoothed.size:
            top3 = np.argsort(-smoothed)[:3]
            for ki in top3:
                if smoothed[ki] < thr_const * 0.5:
                    continue
                ys, xs = (self.det_state["per_key_mask"][ki] > 0).nonzero()
                if xs.size == 0:
                    continue
                x = int(xs.min())
                y = int(ys.min())
                txt = f"k{ki}:{int(smoothed[ki])}/{int(thr_const)}"
                cv2.putText(viz, txt, (x, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(viz, txt, (x, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 0), 1, cv2.LINE_AA)
        self._last_diff_overlay = viz

        self._prev_gray = warped_gray.copy()
        return pressed, viz

    def process(self, warped_bgr: np.ndarray):
        """Run one frame's detection. Returns (pressed_set, line_viz_image)."""
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        sk = self._compute_hand_mask(warped_bgr, warped_gray)

        if self.use_diff_only:
            return self._process_diff(warped_bgr, warped_gray, sk)

        # Line channel + visualization (color-coded LSD segments).
        scores, line_viz = self._diagnose_lines(warped_bgr, warped_gray, sk)
        line_above = (
            scores > self._line_thresholds
            if self._line_thresholds is not None
            else np.zeros(self.n_keys, dtype=bool)
        )

        # Brightness channel.
        bright_score = np.full(self.n_keys, np.nan, dtype=np.float32)
        if (
            self._bright_baseline is not None
            and self._bright_thresholds is not None
        ):
            cur_b = channel_brightness(warped_gray, sk, self.det_state)
            delta = cur_b - self._bright_baseline
            white_idx = np.where(~self.is_black)[0]
            if len(white_idx) > 0:
                wd = delta[white_idx]
                if not np.all(np.isnan(wd)):
                    shift = float(np.nanmedian(wd))
                    if not np.isnan(shift):
                        delta = delta - shift
            bright_score = np.abs(delta) - self._bright_thresholds

        # Slope channel.
        slope_z = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self._slope_baseline_mean is not None:
            cur_slope = channel_slope(warped_gray, sk, self.det_state)
            slope_delta = cur_slope - self._slope_baseline_mean
            slope_delta = ((slope_delta + 90.0) % 180.0) - 90.0
            with np.errstate(invalid="ignore"):
                slope_z = np.abs(slope_delta) / np.maximum(
                    self._slope_baseline_std, 1.0
                )

        # Temp-diff channel.
        tempdiff_z = np.full(self.n_keys, np.nan, dtype=np.float32)
        if self._rest_mean_frame is not None:
            cur_td = channel_temp_diff(
                warped_gray, sk, self.det_state, self._rest_mean_frame
            )
            if (
                self._tempdiff_chaos_mean is not None
                and self._tempdiff_chaos_std is not None
            ):
                with np.errstate(invalid="ignore"):
                    tempdiff_z = (cur_td - self._tempdiff_chaos_mean) / np.maximum(
                        self._tempdiff_chaos_std, 1.0
                    )
            else:
                tempdiff_z = cur_td

        # Roll into smoothing buffers.
        self._line_buf[self._buf_idx] = scores
        self._bright_buf[self._buf_idx] = bright_score
        self._slope_buf[self._buf_idx] = slope_z
        self._tempdiff_buf[self._buf_idx] = tempdiff_z
        self._buf_idx = (self._buf_idx + 1) % self.smooth_window
        self._buf_filled = min(self._buf_filled + 1, self.smooth_window)

        # Smoothed (rolling-mean) per-channel above-threshold flags.
        with np.errstate(invalid="ignore"):
            line_smoothed = np.nanmean(self._line_buf[:self._buf_filled], axis=0)
            bright_smoothed = np.nanmean(self._bright_buf[:self._buf_filled], axis=0)
            slope_smoothed = np.nanmean(self._slope_buf[:self._buf_filled], axis=0)
            tempdiff_smoothed = np.nanmean(
                self._tempdiff_buf[:self._buf_filled], axis=0
            )
        line_above_s = (
            line_smoothed > self._line_thresholds
            if self._line_thresholds is not None
            else np.zeros(self.n_keys, dtype=bool)
        )
        bright_above_s = np.where(
            np.isnan(bright_smoothed), False, bright_smoothed > 0
        ).astype(bool)
        slope_above_s = np.where(
            np.isnan(slope_smoothed), False, slope_smoothed > self.slope_n_sigma
        ).astype(bool)
        tempdiff_above_s = np.where(
            np.isnan(tempdiff_smoothed),
            False,
            tempdiff_smoothed > self.tempdiff_n_sigma,
        ).astype(bool)

        # Per-type fusion.
        above = np.where(
            self.is_black,
            line_above_s | slope_above_s | tempdiff_above_s,
            bright_above_s | slope_above_s | tempdiff_above_s,
        )
        self._above_count = np.where(above, self._above_count + 1, 0)
        pressed = {
            int(i) for i, c in enumerate(self._above_count) if c >= self.debounce
        }

        self._prev_gray = warped_gray.copy()
        return pressed, line_viz

    # ── Line-segment diagnostic visualization ────────────────────────────

    def _diagnose_lines(self, warped_bgr, gray, skin):
        """Compute line-channel scores AND a color-coded segment viz image.
        green  = interior emergent (counts toward press)
        yellow = on polygon-boundary band (expected)
        red    = on skin pixel (suppressed)
        blue   = outside any polygon
        Plus: orange overlay + magenta contour for skin region.
        """
        ov = self.det_state
        n = self.n_keys
        scores = np.zeros(n, dtype=np.float32)
        viz = warped_bgr.copy()
        if skin is not None and np.any(skin):
            orange = np.zeros_like(viz)
            orange[skin > 0] = (0, 140, 255)
            viz = cv2.addWeighted(orange, 0.55, viz, 1.0, 0)
            cnts, _ = cv2.findContours(
                (skin > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(viz, cnts, -1, (255, 0, 255), 2, cv2.LINE_AA)
        # Polygon outlines so segmentation is visible on the same view.
        # Black polys = orange-ish thin line, white polys = green-ish thin line.
        # Drawn FIRST so subsequent LSD segments + skin overlay still read.
        for ki, mask in enumerate(ov["per_key_mask"]):
            cnts2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            col = (0, 200, 0) if self.types[ki] == "white" else (0, 100, 200)
            cv2.drawContours(viz, cnts2, -1, col, 1, cv2.LINE_AA)

        g_supp = gray.copy()
        g_supp[skin > 0] = 128
        res = self._lsd.detect(g_supp)
        if res is None or res[0] is None:
            return scores, viz
        lines = res[0].reshape(-1, 4)
        if lines.size == 0:
            return scores, viz
        H, W = ov["H"], ov["W"]
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        for x1, y1, x2, y2 in lines:
            mx = int((x1 + x2) * 0.5)
            my = int((y1 + y2) * 0.5)
            if not (0 <= mx < W and 0 <= my < H):
                continue
            if grad_mag[my, mx] < GRAD_MIN:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            ki = int(ov["key_id_map"][my, mx])
            on_boundary = bool(ov["boundary_band"][my, mx] > 0)
            on_skin = bool(skin[my, mx] > 0)
            if on_skin:
                color = (0, 0, 255)
            elif ki < 0:
                color = (200, 100, 0)
            elif on_boundary:
                color = (0, 220, 220)
            else:
                color = (0, 255, 0)
                scores[ki] += length
            cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA)
        return scores, viz
