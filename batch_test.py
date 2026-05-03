"""Run the top-down key-detection pipeline on multiple photos and save a grid.

Each row = one image, columns = original+bbox, warped, warped+labels.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from key_extractor2 import (
    load_image,
    find_keyboard_bbox,
    warp_to_bbox,
    draw_labels_tight_crop,
)


CELL_W = 640
CELL_H = 360


def _fit_cell(img: np.ndarray, label: str) -> np.ndarray:
    """Scale image into CELL_W x CELL_H preserving aspect, letterbox, label."""
    if img is None or img.size == 0:
        img = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(CELL_W / w, CELL_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    y0 = (CELL_H - nh) // 2
    x0 = (CELL_W - nw) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = resized
    cv2.putText(cell, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(cell, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def process_one(path: str):
    frame = load_image(path)
    bbox, _, _ = find_keyboard_bbox(frame, debug=True)
    name = Path(path).stem
    if bbox is None:
        blank = np.zeros_like(frame)
        return name, (blank, blank, blank), None

    x0, y0, x1, y1 = bbox
    bbox_vis = frame.copy()
    cv2.rectangle(bbox_vis, (x0, y0), (x1, y1), (0, 255, 0), 8)

    warped = warp_to_bbox(frame, bbox)
    labeled = draw_labels_tight_crop(warped)
    return name, (bbox_vis, warped, labeled), bbox


def main(paths):
    rows = []
    for p in paths:
        name, (a, b, c), bbox = process_one(p)
        size = f"{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}" if bbox else "NONE"
        print(f"{name}: bbox={bbox} size={size}")
        row = np.hstack([
            _fit_cell(a, f"{name} | original + bbox"),
            _fit_cell(b, f"{name} | warped"),
            _fit_cell(c, f"{name} | warped + labels"),
        ])
        rows.append(row)
    grid = np.vstack(rows)
    out = Path("batch_result.png")
    cv2.imwrite(str(out), grid)
    print(f"wrote {out.resolve()} — {grid.shape[1]}x{grid.shape[0]}")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else [
        "piano_photos/IMG_9064.jpg",
        "piano_photos/IMG_9066.jpg",
        "piano_photos/IMG_9073.jpg",
    ])
