"""Piano key-press detector built on the Calibration pipeline.

PressDetector takes a loaded Calibration (from calibration.py) and a
short window of quiet-keyboard warped frames, then detects key presses
frame-by-frame by comparing each key's safe-mask mean against the live
baseline.

The live baseline (collected at startup) replaces the single-frame
baseline stored in the JSON, which may reflect different lighting.

Robust piano key-press detector built on the Calibration pipeline.

Main ideas:
- Uses the existing Calibration/RuntimeKey objects.
- Builds a quiet-keyboard baseline from startup frames.
- Computes a per-key press score from:
    1. safe-mask intensity delta
    2. front-region intensity delta
    3. frame-to-frame motion
- Uses per-key adaptive thresholds from baseline noise.
- Uses hysteresis and debounce to avoid flicker.
- Slowly adapts baseline when a key is released and stable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from .calibration import Calibration
from .seg_to_keys import isolate_white

@dataclass
class NoteEvent:
    note: str          # e.g. "C4", "F#3", or "key_07" if note label is unknown
    event: str         # "press" | "release"
    time: float        # time.perf_counter() timestamp
    key_index: int
    score: float
    threshold: float


@dataclass
class KeyDebug:
    note: str
    key_index: int
    is_pressed: bool
    score: float
    press_threshold: float
    release_threshold: float
    safe_delta: float
    front_delta: float
    motion_delta: float


class PressDetector:
    """Per-key temporal detector using pre-calibrated masks.

    Workflow:
        1. Collect N quiet-keyboard warped frames.
        2. detector = PressDetector(calibration, baseline_frames)
        3. events = detector.update(warped)
        4. overlay = detector.draw_overlay(warped)

    This class intentionally does not know about cameras, windows, or CLI args.
    """
    MIN_NORMALIZED_PRESS = 0.70
    LOCAL_PEAK_RADIUS = 1
    MAX_NEW_PRESSES_PER_FRAME = 3
    ENABLE_BLACK_KEYS = False

    N_SIGMA = 2.0
    MIN_PRESS_THRESH = 2.5
    RELEASE_RATIO = 0.40

    PRESS_DEBOUNCE = 1
    RELEASE_DEBOUNCE = 3

    # Shadow rejection
    LOCAL_MEDIAN_RADIUS = 3
    LOCAL_MEDIAN_WEIGHT = 0.75

    # Score weights
    SAFE_WEIGHT = 1.0
    FRONT_WEIGHT = 0.65
    MOTION_WEIGHT = 0.75

    # Baseline adaptation: only when released and stable
    BASELINE_ADAPT_RATE = 0.004

    # Preprocessing
    BLUR_KERNEL = (5, 5)

    def __init__(
        self,
        calibration: Calibration,
        baseline_frames: list[np.ndarray],
        *,
        n_sigma: float | None = None,
        min_press_thresh: float | None = None,
        release_ratio: float | None = None,
        press_debounce: int | None = None,
        release_debounce: int | None = None,
    ):
        if not baseline_frames:
            raise ValueError("PressDetector requires at least one baseline frame.")

        self._calib = calibration
        self._keys = list(calibration.keys)
        self._n = len(self._keys)

        self.n_sigma = float(self.N_SIGMA if n_sigma is None else n_sigma)
        self.min_press_thresh = float(
            self.MIN_PRESS_THRESH if min_press_thresh is None else min_press_thresh
        )
        self.release_ratio = float(
            self.RELEASE_RATIO if release_ratio is None else release_ratio
        )
        self.press_debounce = int(
            self.PRESS_DEBOUNCE if press_debounce is None else press_debounce
        )
        self.release_debounce = int(
            self.RELEASE_DEBOUNCE if release_debounce is None else release_debounce
        )

        self._safe_masks = [self._clean_mask(k.safe_mask) for k in self._keys]
        self._front_masks = [self._make_front_mask(k.safe_bbox, k.safe_mask.shape, k.type) for k in self._keys]

        grays = [self._match_mask_shape(self._preprocess(f)) for f in baseline_frames]

        self._safe_baselines = np.zeros(self._n, dtype=np.float32)
        self._front_baselines = np.zeros(self._n, dtype=np.float32)

        for i in range(self._n):
            self._safe_baselines[i] = self._masked_mean_stack(grays, self._safe_masks[i])
            self._front_baselines[i] = self._masked_mean_stack(grays, self._front_masks[i])

        # Estimate quiet-frame score distribution for each key.
        quiet_scores = np.zeros((len(grays), self._n), dtype=np.float32)
        prev_gray: Optional[np.ndarray] = None
        for fi, gray in enumerate(grays):
            for ki in range(self._n):
                quiet_scores[fi, ki] = self._score_key(ki, gray, prev_gray)[0]
            prev_gray = gray

        noise_mean = quiet_scores.mean(axis=0)
        noise_std = np.maximum(quiet_scores.std(axis=0), 0.75)

        raw_press = np.maximum(
            self.min_press_thresh,
            noise_mean + self.n_sigma * noise_std,
        )

        # Lower confidence keys get more conservative thresholds.
        confidences = np.array(
            [max(0.25, float(getattr(k, "confidence", 1.0))) for k in self._keys],
            dtype=np.float32,
        )
        self._press_thresholds = (raw_press / confidences).astype(np.float32)
        self._release_thresholds = (self._press_thresholds * self.release_ratio).astype(np.float32)

        self._is_pressed = [False] * self._n
        self._press_pending = [0] * self._n
        self._release_pending = [0] * self._n
        self._last_gray: Optional[np.ndarray] = None

        self._last_debug: list[KeyDebug] = [
            KeyDebug(
                note=self._display_note(i),
                key_index=i,
                is_pressed=False,
                score=0.0,
                press_threshold=float(self._press_thresholds[i]),
                release_threshold=float(self._release_thresholds[i]),
                safe_delta=0.0,
                front_delta=0.0,
                motion_delta=0.0,
            )
            for i in range(self._n)
        ]

    def _match_mask_shape(self, gray: np.ndarray) -> np.ndarray:
        """Crop/pad gray image to exactly match calibration mask shape.

        Needed because warp_to_piano can drift by a few pixels frame-to-frame.
        """
        target_h, target_w = self._safe_masks[0].shape
        h, w = gray.shape[:2]

        gray = gray[:target_h, :target_w]

        h2, w2 = gray.shape[:2]
        if h2 == target_h and w2 == target_w:
            return gray

        out = np.zeros((target_h, target_w), dtype=gray.dtype)
        out[:h2, :w2] = gray
        return out
    
    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update(self, warped: np.ndarray) -> list[NoteEvent]:
        """Update detector with one warped BGR frame and return new events.

        Candidate-gated raw mode:
        - compute all scores first
        - normalize score by threshold
        - suppress non-local peaks
        - cap how many new keys can press in one frame
        - optionally ignore black keys while debugging
        """
        events: list[NoteEvent] = []

        if warped is None or warped.size == 0:
            return events

        gray = self._match_mask_shape(self._preprocess(warped))
        now = time.perf_counter()

        raw_scores = np.zeros(self._n, dtype=np.float32)
        safe_deltas = np.zeros(self._n, dtype=np.float32)
        front_deltas = np.zeros(self._n, dtype=np.float32)
        motion_deltas = np.zeros(self._n, dtype=np.float32)

        # Global illumination correction: compute median shift across white keys.
        # Subtracting this cancels lighting changes that affect all keys together
        # (e.g. room dimming, camera gain drift) without masking real presses.
        white_cur, white_base = [], []
        for ki, key in enumerate(self._keys):
            if key.type == "white":
                m = self._masked_mean(gray, self._safe_masks[ki])
                white_cur.append(m)
                white_base.append(float(self._safe_baselines[ki]))
        global_shift = (
            float(np.median(white_cur)) - float(np.median(white_base))
            if white_cur else 0.0
        )

        # First pass: score every key.
        for ki, key in enumerate(self._keys):
            if not self.ENABLE_BLACK_KEYS and key.type == "black":
                raw_scores[ki] = 0.0
                continue

            score, safe_delta, front_delta, motion_delta = self._score_key(
                ki,
                gray,
                self._last_gray,
                white_mask=None,
                global_shift=global_shift,
            )

            raw_scores[ki] = float(score)
            safe_deltas[ki] = float(safe_delta)
            front_deltas[ki] = float(front_delta)
            motion_deltas[ki] = float(motion_delta)

        press_thresholds = np.maximum(self._press_thresholds, 1e-6)
        release_thresholds = np.maximum(self._release_thresholds, 1e-6)

        normalized = raw_scores / press_thresholds

        # Second pass: identify local-peak press candidates.
        candidate_indices: list[int] = []

        for ki, key in enumerate(self._keys):
            if not self.ENABLE_BLACK_KEYS and key.type == "black":
                continue

            if self._is_pressed[ki]:
                continue

            if normalized[ki] < self.MIN_NORMALIZED_PRESS:
                continue

            r = self.LOCAL_PEAK_RADIUS
            lo = max(0, ki - r)
            hi = min(self._n, ki + r + 1)

            local_max = float(np.max(normalized[lo:hi]))
            if normalized[ki] >= local_max - 1e-6:
                candidate_indices.append(ki)

        # If a shadow/hand event triggers too many keys, keep only strongest few.
        candidate_indices.sort(key=lambda idx: normalized[idx], reverse=True)
        allowed_new_presses = set(candidate_indices[: self.MAX_NEW_PRESSES_PER_FRAME])

        debug_rows: list[KeyDebug] = []

        # Third pass: state machine.
        for ki, key in enumerate(self._keys):
            score = float(raw_scores[ki])
            press_thr = float(self._press_thresholds[ki])
            release_thr = float(self._release_thresholds[ki])

            if not self._is_pressed[ki]:
                if ki in allowed_new_presses:
                    self._press_pending[ki] += 1
                else:
                    self._press_pending[ki] = 0

                self._release_pending[ki] = 0

                if self._press_pending[ki] >= self.press_debounce:
                    self._is_pressed[ki] = True
                    self._press_pending[ki] = 0
                    events.append(
                        NoteEvent(
                            note=self._display_note(ki),
                            event="press",
                            time=now,
                            key_index=ki,
                            score=score,
                            threshold=press_thr,
                        )
                    )
                else:
                    if score < release_thr:
                        self._adapt_baseline(ki, gray, global_shift)

            else:
                # For release, use raw score. Pressed keys should release when
                # their own evidence falls back down.
                if score <= release_thr:
                    self._release_pending[ki] += 1
                else:
                    self._release_pending[ki] = 0

                self._press_pending[ki] = 0

                if self._release_pending[ki] >= self.release_debounce:
                    self._is_pressed[ki] = False
                    self._release_pending[ki] = 0
                    events.append(
                        NoteEvent(
                            note=self._display_note(ki),
                            event="release",
                            time=now,
                            key_index=ki,
                            score=score,
                            threshold=release_thr,
                        )
                    )

            debug_rows.append(
                KeyDebug(
                    note=self._display_note(ki),
                    key_index=ki,
                    is_pressed=self._is_pressed[ki],
                    score=float(normalized[ki]),  # display normalized score
                    press_threshold=1.0,
                    release_threshold=float(release_thresholds[ki] / press_thresholds[ki]),
                    safe_delta=float(safe_deltas[ki]),
                    front_delta=float(front_deltas[ki]),
                    motion_delta=float(motion_deltas[ki]),
                )
            )

        self._last_gray = gray
        self._last_debug = debug_rows
        return events

    def active_notes(self) -> list[str]:
        """Return currently pressed note names."""
        return [
            self._display_note(i)
            for i, state in enumerate(self._is_pressed)
            if state
        ]

    def debug_rows(self) -> list[KeyDebug]:
        """Return per-key debug data from the most recent update."""
        return list(self._last_debug)

    def draw_overlay(self, img: np.ndarray, *, show_scores: bool = False) -> np.ndarray:
        """Return a copy with pressed-key outlines and optional scores."""
        out = img.copy()

        for i, (key, state) in enumerate(zip(self._keys, self._is_pressed)):
            if not state:
                continue

            cv2.drawContours(out, [key.polygon], -1, (0, 255, 0), 3)

            if show_scores:
                x, y, w, h = key.bbox
                dbg = self._last_debug[i]
                label = f"{self._display_note(i)} {dbg.score:.1f}"
                self._put_text(out, label, (x + 3, max(14, y + 18)))

        return out

    # ---------------------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------------------

    def _score_key(
        self,
        key_index: int,
        gray: np.ndarray,
        prev_gray: Optional[np.ndarray],
        white_mask: Optional[np.ndarray] = None,
        global_shift: float = 0.0,
    ) -> tuple[float, float, float, float] | None:
        raw_safe = self._safe_masks[key_index]
        raw_front = self._front_masks[key_index]

        # For white keys, restrict measurement to pixels that are visibly white-key
        # surface (not covered by a hand). If <30% of the safe region is still white,
        # the hand is occluding the key — return None to signal skip.
        if white_mask is not None and self._keys[key_index].type == "white":
            safe_mask = raw_safe & white_mask
            coverage = safe_mask.sum() / max(1, raw_safe.sum())
            if coverage < 0.3:
                return None
            front_mask = raw_front & white_mask
        else:
            safe_mask = raw_safe
            front_mask = raw_front

        safe_mean = self._masked_mean(gray, safe_mask)
        front_mean = self._masked_mean(gray, front_mask)

        safe_baseline = float(self._safe_baselines[key_index])
        front_baseline = float(self._front_baselines[key_index])

        if self._keys[key_index].type == "white":
            # Global-shift-corrected means: cancel lighting drift common to all keys.
            # Signed (darkening only): a finger covering the key makes it darker;
            # specular reflections (brighter than baseline) are NOT a press.
            safe_delta = max(0.0, safe_baseline - (safe_mean - global_shift))
            front_delta = max(0.0, front_baseline - (front_mean - global_shift))
        else:
            safe_delta = abs(safe_mean - safe_baseline)
            front_delta = abs(front_mean - front_baseline)

        motion_delta = 0.0
        if prev_gray is not None and prev_gray.shape == gray.shape:
            curr_px = gray[safe_mask]
            prev_px = prev_gray[safe_mask]
            if curr_px.size and prev_px.size == curr_px.size:
                motion_delta = float(np.mean(np.abs(curr_px - prev_px)))

        score = (
            self.SAFE_WEIGHT * safe_delta
            + self.FRONT_WEIGHT * front_delta
            + self.MOTION_WEIGHT * motion_delta
        )

        return float(score), float(safe_delta), float(front_delta), float(motion_delta)

    def _adapt_baseline(self, key_index: int, gray: np.ndarray, global_shift: float = 0.0) -> None:
        # Adapt toward the global-shift-corrected mean so the baseline stays
        # anchored to key-surface appearance rather than drifting with lighting.
        safe_mean = self._masked_mean(gray, self._safe_masks[key_index]) - global_shift
        front_mean = self._masked_mean(gray, self._front_masks[key_index]) - global_shift

        a = self.BASELINE_ADAPT_RATE
        self._safe_baselines[key_index] = (
            (1.0 - a) * self._safe_baselines[key_index] + a * safe_mean
        )
        self._front_baselines[key_index] = (
            (1.0 - a) * self._front_baselines[key_index] + a * front_mean
        )

    # ---------------------------------------------------------------------
    # Masks / preprocessing
    # ---------------------------------------------------------------------

    @staticmethod
    def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.GaussianBlur(gray, PressDetector.BLUR_KERNEL, 0)

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        return np.asarray(mask, dtype=np.bool_)

    @staticmethod
    def _make_front_mask(
        safe_bbox: tuple[int, int, int, int],
        shape: tuple[int, int],
        key_type: str,
    ) -> np.ndarray:
        """Create a smaller lower/front region inside the existing safe box.

        In the warped view, larger y usually means closer/front side of keys.
        White keys get a lower band. Black keys get a slightly larger lower band.
        """
        h_img, w_img = shape
        x, y, w, h = [int(v) for v in safe_bbox]

        x0 = max(0, x)
        x1 = min(w_img, x + w)
        y0 = max(0, y)
        y1 = min(h_img, y + h)

        mask = np.zeros(shape, dtype=np.bool_)
        if x1 <= x0 or y1 <= y0:
            return mask

        frac = 0.55 if key_type == "black" else 0.45
        fy0 = int(round(y0 + (y1 - y0) * frac))
        fy0 = min(max(y0, fy0), y1 - 1)

        mask[fy0:y1, x0:x1] = True
        return mask

    @staticmethod
    def _masked_mean(img: np.ndarray, mask: np.ndarray) -> float:
        px = img[mask]
        if px.size == 0:
            return 0.0
        return float(px.mean())

    @classmethod
    def _masked_mean_stack(cls, imgs: list[np.ndarray], mask: np.ndarray) -> float:
        vals = [cls._masked_mean(img, mask) for img in imgs]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    # ---------------------------------------------------------------------
    # Display helpers
    # ---------------------------------------------------------------------

    def _display_note(self, key_index: int) -> str:
        raw = str(getattr(self._keys[key_index], "note", "") or "")
        if raw and raw != "?":
            return raw
        return f"key_{key_index:02d}"

    @staticmethod
    def _put_text(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
        for thickness, color in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
            cv2.putText(
                img,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                thickness,
                cv2.LINE_AA,
            )