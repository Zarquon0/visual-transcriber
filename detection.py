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

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions as _MPBaseOptions
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

_MP_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


# Hand-mask tuning. Per-pixel candidate test (sat_increase + V change)
# captures hands, but ALSO press-edge sat-jumps (smaller blobs along
# polygon boundaries). Filter happens at the BLOB level — small blobs
# (press-edges) never enter persistence; only large hand-shaped blobs do.
REST_V_DIFF_T = 20      # |V_curr - V_rest| > this counts as foreground
SAT_INCREASE_T = 8      # S_curr - S_rest > this means "pixel got more saturated"
HAND_BLOB_MIN_AREA = 300       # min pixels for a strong_fg blob to be "hand"
HAND_BLOB_MIN_THICKNESS = 8    # min(w, h) of blob bbox — press-edges fail this
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
    ):
        self.det_state = build_overlays(keys_dict)
        self.types = [k["type"] for k in keys_dict["keys"]]
        self.n_keys = len(self.types)
        self.is_black = np.array([t == "black" for t in self.types])

        self.smooth_window = max(1, smooth_window)
        self.debounce = max(1, debounce)
        self.slope_n_sigma = slope_n_sigma
        self.tempdiff_n_sigma = tempdiff_n_sigma
        self.margin_black = margin_black
        self.margin_white = margin_white

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

        # MediaPipe-based hand mask (color-independent, robust to skin
        # tone, much cleaner than heuristics). Runs on the SOURCE frame
        # (not warped), then projects hand-landmark convex hull into
        # warped space using the calibration's perspective transform M.
        self.use_mediapipe = (
            use_mediapipe and _MP_AVAILABLE and _MP_MODEL_PATH.exists()
        )
        self._mp_landmarker = None
        self._source_frame = None  # caller sets via set_source_frame() each frame
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

    def set_source_frame(self, source_bgr: np.ndarray):
        """Cache the current source (pre-warp) frame for MediaPipe hand
        detection. Caller passes this every frame BEFORE process()."""
        self._source_frame = source_bgr

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

    def _mediapipe_search_region(self, warped_shape):
        """Returns a generous binary search region around hand bones in
        warped space (skeleton + dilated). The actual hand mask is then
        the subset of pixels INSIDE this region that pass per-pixel
        rest-diff + saturation-increase tests — width comes from real
        pixel data, not hardcoded bone-thickness."""
        if (
            self._mp_landmarker is None
            or self._source_frame is None
            or self._source_frame.size == 0
        ):
            return None
        rgb = cv2.cvtColor(self._source_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._mp_landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        H, W = warped_shape
        h_src, w_src = self._source_frame.shape[:2]
        M = self.det_state["M"]
        skeleton = np.zeros((H, W), dtype=np.uint8)
        for hand_landmarks in result.hand_landmarks:
            pts_src = np.array(
                [[lm.x * w_src, lm.y * h_src] for lm in hand_landmarks],
                dtype=np.float32,
            )
            pts_warped = cv2.perspectiveTransform(
                pts_src.reshape(-1, 1, 2), M
            ).reshape(-1, 2).astype(np.int32)
            # Thin skeleton (3 px); width comes from search-region dilate
            # below + per-pixel test inside.
            for a, b in self._MP_HAND_CONNECTIONS:
                cv2.line(
                    skeleton, tuple(pts_warped[a]), tuple(pts_warped[b]),
                    255, thickness=3, lineType=cv2.LINE_AA,
                )
            palm_pts = pts_warped[list(self._MP_PALM_INDICES)]
            cv2.fillPoly(skeleton, [palm_pts], 255)
        # Dilate to a generous search region (covers any plausible finger
        # / palm thickness for the camera angle without committing to a
        # specific number).
        return cv2.dilate(
            skeleton, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
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

            # Compute per-pixel hand-evidence (works regardless of MP).
            hand_evidence = None
            if (
                self._rest_mean_frame is not None
                and self._rest_mean_bgr is not None
            ):
                rest_diff = cv2.absdiff(warped_gray, self._rest_mean_frame)
                hsv_curr = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
                hsv_rest = cv2.cvtColor(self._rest_mean_bgr, cv2.COLOR_BGR2HSV)
                sat_inc = (
                    hsv_curr[..., 1].astype(np.int16)
                    - hsv_rest[..., 1].astype(np.int16)
                )
                # Hand evidence: pixel changed AND became more saturated.
                hand_evidence = (rest_diff > 15) & (sat_inc > 5)

            search = self._mediapipe_search_region(warped_gray.shape)
            current = np.zeros(warped_gray.shape, dtype=bool)

            if search is not None and np.any(search):
                # MP detected: keep pixels in search region that pass evidence.
                if hand_evidence is not None:
                    current = (search > 0) & hand_evidence
                else:
                    current = (search > 0)
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
                self._hand_persistence[current] = 1
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

    def process(self, warped_bgr: np.ndarray):
        """Run one frame's detection. Returns (pressed_set, line_viz_image)."""
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        sk = self._compute_hand_mask(warped_bgr, warped_gray)

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
