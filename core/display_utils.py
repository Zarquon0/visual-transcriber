"""Shared display utilities for live OpenCV windows."""

from __future__ import annotations

import cv2
import numpy as np

CORNER_REFRESH = 45  # re-run warp_to_piano every N frames


def _corner_overlay(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    pts = corners.astype(int)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for (x, y), lbl in zip(pts, ["TL", "TR", "BR", "BL"]):
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(vis, lbl, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def _status(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    for wt, c in [(3, (0, 0, 0)), (1, (255, 255, 255))]:
        cv2.putText(frame, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, wt, cv2.LINE_AA)


def _side_by_side(left: np.ndarray, right: np.ndarray, h: int = 360) -> np.ndarray:
    def fit(img: np.ndarray) -> np.ndarray:
        ih, iw = img.shape[:2]
        s = h / ih
        return cv2.resize(img, (max(1, int(iw * s)), h))
    return np.hstack([fit(left), fit(right)])
