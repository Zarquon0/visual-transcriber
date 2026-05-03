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


# Hand-mask tuning constants. Mask uses per-pixel **saturation increase**
# vs the rest baseline (computed in HSV space) as the shadow-vs-hand
# discriminator: a hand replaces a key pixel with skin, which has higher
# saturation than the off-white/warm key. A shadow keeps the same key
# pixel just darker — saturation stays similar. This cleanly separates
# the two cases without needing color matching to a fixed skin range.
REST_V_DIFF_T = 20      # |V_curr - V_rest| > this counts as foreground
SAT_INCREASE_T = 8      # S_curr - S_rest > this means "pixel got more saturated" (hand)
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

    def reset_smoothing(self):
        self._line_buf[:] = 0
        self._bright_buf[:] = np.nan
        self._slope_buf[:] = np.nan
        self._tempdiff_buf[:] = np.nan
        self._buf_idx = 0
        self._buf_filled = 0
        self._above_count[:] = 0

    # ── Hand mask ────────────────────────────────────────────────────────

    def _compute_hand_mask(self, warped_bgr: np.ndarray, warped_gray: np.ndarray):
        """HSV-saturation-increase hand mask.

        Per pixel:
          hand = (V changed significantly) AND (S increased vs rest)
        A hand replaces a key with skin → saturation jumps up. A shadow
        keeps the same key, just darker → saturation barely changes. A
        pressed key shifts intensity slightly with no real saturation
        change. Only true hand pixels satisfy both conditions.
        Plus: persistence (keeps held hands masked), connected-component
        blob filter on tight pixels (kills scattered noise).
        """
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
        # Strong hand candidate: significant V change AND saturation jumped up.
        strong_fg = (v_diff > REST_V_DIFF_T) & (sat_increase > SAT_INCREASE_T)
        # Weak: any V change at all (used during persistence decay).
        weak_fg = v_diff > REST_V_DIFF_T
        if (self._hand_persistence is None
                or self._hand_persistence.shape != warped_gray.shape):
            self._hand_persistence = np.zeros(warped_gray.shape, dtype=np.uint8)
        self._hand_persistence[strong_fg] = PERSISTENCE_FRAMES
        decay = (~strong_fg) & (self._hand_persistence > 0)
        self._hand_persistence[decay] -= 1
        recent_hand = self._hand_persistence > 0
        mask_tight = (
            (strong_fg | (recent_hand & weak_fg)).astype(np.uint8) * 255
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
