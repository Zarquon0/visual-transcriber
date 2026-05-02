"""MediaPipe hand landmark extractor + calibration-aware candidate key gate.

Stage 1 (no calibration):
    gate = HandGate()
    tips = gate.fingertips(raw_bgr)      # raw-frame pixel coords
    debug = gate.draw(raw_bgr, tips)

Stage 2 (with calibration):
    gate = HandGate(calibration)
    candidates = gate.candidate_keys(raw_bgr)   # set[int] of key indices
    det.update(warped, candidate_key_indices=candidates)
    raw_dbg    = gate.draw(raw_bgr, gate._last_raw_tips)
    warped_dbg = gate.draw_warped_debug(warped)

MediaPipe is called exactly once per frame (inside candidate_keys → fingertips).

Landmark index reference (MediaPipe Hands):
  4  THUMB_TIP       8  INDEX_FINGER_TIP
  12 MIDDLE_FINGER_TIP  16 RING_FINGER_TIP  20 PINKY_TIP
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

FINGERTIP_IDS = (4, 8, 12, 16, 20)

_HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
_DEFAULT_MODEL = Path(__file__).parent.parent / "hand_landmarker.task"


class HandGate:
    """MediaPipe hand landmarker with optional calibration-aware key gating.

    Parameters
    ----------
    calibration:
        A loaded ``Calibration`` object.  When provided, ``candidate_keys()``
        perspective-transforms fingertips into warped keyboard coords and maps
        them to key polygons.  When ``None``, only raw-frame fingertip
        detection is available.
    model_path:
        Path to hand_landmarker.task.
    max_hands:
        Maximum simultaneous hands to detect.
    min_detection_confidence / min_tracking_confidence:
        MediaPipe thresholds.
    candidate_ttl_frames:
        After a fingertip leaves a key, keep that key in the candidate set for
        this many additional frames (absorbs brief tracking dropout).
    include_neighbors:
        Also add this many adjacent key indices on each side of the matched
        key to the candidate set (accounts for fingertip positional error and
        wide finger tips).
    """

    _INFER_WIDTH = 640   # downsample to this width before running MediaPipe

    def __init__(
        self,
        calibration=None,
        model_path: str | Path = _DEFAULT_MODEL,
        max_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        candidate_ttl_frames: int = 5,
        include_neighbors: int = 1,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"hand_landmarker.task not found at {model_path}.\n"
                "Download it with:\n"
                "  curl -L -o hand_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self._last_result = None
        self._frame_ts_ms: int = 0

        self._calibration = calibration
        self._candidate_ttl_frames = int(candidate_ttl_frames)
        self._include_neighbors = int(include_neighbors)
        self._candidate_ttl: dict[int, int] = {}   # key_index → frames remaining
        self._last_raw_tips: list[tuple[int, int]] = []
        self._last_warped_tips: list[tuple[int, int]] = []

    # ── MediaPipe inference ──────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray):
        """Run MediaPipe on one BGR frame (VIDEO running mode).

        Downsamples to _INFER_WIDTH before inference; landmark coordinates
        are always returned in the original frame's pixel space because
        MediaPipe normalises to [0, 1].  Advances an internal monotonic
        timestamp by 1 ms per call.
        """
        h, w = frame_bgr.shape[:2]
        if w > self._INFER_WIDTH:
            scale = self._INFER_WIDTH / w
            small = cv2.resize(frame_bgr, (self._INFER_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self._last_result = result
        return result

    # ── Raw-frame landmark extraction ────────────────────────────────────────

    def fingertips(self, frame_bgr: np.ndarray) -> list[tuple[int, int]]:
        """Pixel (x, y) for every fingertip in the raw frame.

        Calls process() internally — call at most once per frame.
        """
        result = self.process(frame_bgr)
        h, w = frame_bgr.shape[:2]
        tips: list[tuple[int, int]] = []
        if not result.hand_landmarks:
            return tips
        for hand_lms in result.hand_landmarks:
            for tip_id in FINGERTIP_IDS:
                lm = hand_lms[tip_id]
                tips.append((int(lm.x * w), int(lm.y * h)))
        return tips

    def all_landmarks(self, frame_bgr: np.ndarray) -> list[list[tuple[int, int]]]:
        """All 21 landmarks per hand as pixel (x, y) lists (index = landmark id).

        Calls process() internally.
        """
        result = self.process(frame_bgr)
        h, w = frame_bgr.shape[:2]
        hands_out: list[list[tuple[int, int]]] = []
        if not result.hand_landmarks:
            return hands_out
        for hand_lms in result.hand_landmarks:
            hands_out.append([(int(lm.x * w), int(lm.y * h)) for lm in hand_lms])
        return hands_out

    # ── Calibration-aware stage 2 methods ────────────────────────────────────

    def warped_fingertips(self, frame_bgr: np.ndarray) -> list[tuple[int, int]]:
        """Fingertip points perspective-transformed into warped keyboard coords.

        Calls fingertips() (which calls process()) — do not also call
        fingertips() on the same frame.
        """
        tips = self.fingertips(frame_bgr)
        self._last_raw_tips = tips

        if self._calibration is None or not tips:
            self._last_warped_tips = []
            return []

        pts = np.array(tips, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self._calibration._M)
        self._last_warped_tips = [(int(x), int(y)) for [[x, y]] in warped]
        return self._last_warped_tips

    def _key_at_point(self, x: float, y: float) -> int | None:
        """Key index whose polygon contains (x, y) in warped coords.

        Checks black keys first (they visually sit above whites and polygons
        may overlap).
        """
        if self._calibration is None:
            return None

        for i, key in enumerate(self._calibration.keys):
            if key.type != "black":
                continue
            if cv2.pointPolygonTest(key.polygon.astype(np.float32),
                                    (float(x), float(y)), False) >= 0:
                return i

        for i, key in enumerate(self._calibration.keys):
            if key.type != "white":
                continue
            if cv2.pointPolygonTest(key.polygon.astype(np.float32),
                                    (float(x), float(y)), False) >= 0:
                return i

        return None

    def candidate_keys(self, frame_bgr: np.ndarray) -> set[int]:
        """Return key indices allowed to start new presses this frame.

        Runs MediaPipe once, transforms fingertips into warped coords, maps
        to key polygons, expands by include_neighbors, and keeps candidates
        alive for candidate_ttl_frames after fingertip dropout.

        Returns an empty set when no calibration is set.
        """
        if self._calibration is None:
            # Still run inference so draw() stays up-to-date.
            self._last_raw_tips = self.fingertips(frame_bgr)
            self._last_warped_tips = []
            return set()

        # Decay existing candidates.
        for k in list(self._candidate_ttl):
            self._candidate_ttl[k] -= 1
            if self._candidate_ttl[k] <= 0:
                del self._candidate_ttl[k]

        warped_tips = self.warped_fingertips(frame_bgr)   # calls process() once
        n_keys = len(self._calibration.keys)

        for x, y in warped_tips:
            ki = self._key_at_point(x, y)
            if ki is None:
                continue
            lo = max(0, ki - self._include_neighbors)
            hi = min(n_keys, ki + self._include_neighbors + 1)
            for j in range(lo, hi):
                self._candidate_ttl[j] = self._candidate_ttl_frames

        return {k for k, ttl in self._candidate_ttl.items() if ttl > 0}

    # ── Debug drawing ────────────────────────────────────────────────────────

    def draw(
        self,
        frame_bgr: np.ndarray,
        tips: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Annotated copy of frame_bgr: hand skeleton + fingertip circles.

        Uses the result from the most recent process() call; does NOT call
        MediaPipe again.
        """
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        if self._last_result and self._last_result.hand_landmarks:
            for hand_lms in self._last_result.hand_landmarks:
                pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                for conn in _HAND_CONNECTIONS:
                    cv2.line(out, pts_px[conn.start], pts_px[conn.end],
                             (200, 200, 200), 2, cv2.LINE_AA)
                for pt in pts_px:
                    cv2.circle(out, pt, 4, (255, 255, 255), -1)

        for x, y in (tips or []):
            cv2.circle(out, (x, y), 9, (0, 255, 255), -1)
            cv2.circle(out, (x, y), 9, (0, 0, 0), 2)

        return out

    def draw_warped_debug(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Annotated copy of warped_bgr: candidate key outlines + warped tips.

        Uses cached state from the most recent candidate_keys() call.
        """
        out = warped_bgr.copy()

        if self._calibration is None:
            return out

        for ki, ttl in self._candidate_ttl.items():
            if ki < 0 or ki >= len(self._calibration.keys):
                continue
            key = self._calibration.keys[ki]
            cv2.polylines(out, [key.polygon.astype(np.int32)], True,
                          (0, 255, 255), 2, cv2.LINE_AA)
            x, y, _, _ = key.bbox
            cv2.putText(out, key.note, (int(x), int(y) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        for x, y in self._last_warped_tips:
            cv2.circle(out, (int(x), int(y)), 7, (0, 255, 255), -1)
            cv2.circle(out, (int(x), int(y)), 7, (0, 0, 0), 2)

        return out

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
