"""Piano key-press detector built on the Calibration pipeline.

PressDetector takes a loaded Calibration (from calibration.py) and a
short window of quiet-keyboard warped frames, then detects key presses
frame-by-frame by comparing each key's safe-mask mean against the live
baseline.

The live baseline (collected at startup) replaces the single-frame
baseline stored in the JSON, which may reflect different lighting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Calibration


@dataclass
class NoteEvent:
    note:  str    # e.g. "C4", "F#3"
    event: str    # "press" | "release"
    time:  float  # time.perf_counter() timestamp


class PressDetector:
    """Frame-diff key-press detector using pre-calibrated safe-region masks.

    Workflow:
        1. Collect N quiet-keyboard warped frames; pass to the constructor.
        2. Call update(warped) each frame; returns a (possibly empty) list
           of NoteEvents for keys that crossed the press/release threshold.
        3. Call draw_overlay(warped) to get a copy annotated with green
           polygon outlines on currently-pressed keys.
    """

    N_SIGMA    = 3.0   # threshold = noise_mean + N_SIGMA * noise_std per key
    DEBOUNCE   = 3     # consecutive frames required to flip key state
    MIN_THRESH = 4.0   # absolute-diff floor in gray units

    def __init__(self, calibration: Calibration, baseline_frames: list[np.ndarray]):
        self._calib = calibration
        keys = calibration.keys
        n    = len(keys)

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                 for f in baseline_frames]

        # Per-key baseline mean from the quiet window
        self._baselines = np.zeros(n, dtype=np.float32)
        for fg in grays:
            for k, key in enumerate(keys):
                px = fg[key.safe_mask]
                if px.size:
                    self._baselines[k] += float(px.mean())
        self._baselines /= max(1, len(grays))

        # Per-key threshold derived from noise distribution across baseline frames
        diffs = np.zeros((len(grays), n), dtype=np.float32)
        for fi, fg in enumerate(grays):
            for k, key in enumerate(keys):
                px = fg[key.safe_mask]
                if px.size:
                    diffs[fi, k] = abs(float(px.mean()) - self._baselines[k])

        noise_mean   = diffs.mean(axis=0)
        noise_std    = np.maximum(diffs.std(axis=0), 0.5)
        raw_thr      = np.maximum(self.MIN_THRESH,
                                  noise_mean + self.N_SIGMA * noise_std)
        # High-confidence keys can flag presses on smaller deltas
        confidences      = np.array([k.confidence for k in keys], dtype=np.float32)
        self._thresholds = (raw_thr / confidences).astype(np.float32)

        self._key_state = [False] * n
        self._pending   = [0]     * n
        self._keys      = keys

    def update(self, warped: np.ndarray) -> list[NoteEvent]:
        """Compare warped against the live baseline; return any new NoteEvents."""
        events: list[NoteEvent] = []
        if warped is None or warped.size == 0:
            return events
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        now  = time.perf_counter()

        for k, key in enumerate(self._keys):
            px = gray[key.safe_mask]
            if not px.size:
                continue
            diff   = abs(float(px.mean()) - float(self._baselines[k]))
            active = diff > float(self._thresholds[k])

            if active == self._key_state[k]:
                self._pending[k] = 0
            else:
                self._pending[k] += 1
                if self._pending[k] >= self.DEBOUNCE:
                    self._key_state[k] = active
                    self._pending[k]   = 0
                    events.append(NoteEvent(
                        note=key.note,
                        event="press" if active else "release",
                        time=now,
                    ))
        return events

    def active_notes(self) -> list[str]:
        """Return note names of all currently-pressed keys."""
        return [k.note for k, s in zip(self._keys, self._key_state) if s]

    def draw_overlay(self, img: np.ndarray) -> np.ndarray:
        """Return a copy of img with a green polygon outline on each pressed key."""
        out = img.copy()
        for key, state in zip(self._keys, self._key_state):
            if state:
                cv2.drawContours(out, [key.polygon], -1, (0, 255, 0), 3)
        return out
