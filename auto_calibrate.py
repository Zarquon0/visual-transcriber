"""auto_calibrate.py — auto-detect keyboard corners + build keys.json.

Uses warp_to_piano from seg_to_keys.py (LSD + RANSAC corner detection)
on a snapshot, then runs the same calibration.build_calibration_data
pipeline that manual_calibrate.py runs after the 4 clicks.

Usage:
    uv run python auto_calibrate.py path/to/snapshot.png

Outputs next to the input:
    <stem>_warped.png    — auto-warped strip
    <stem>_labeled.png   — with note labels
    <stem>_calib.json    — corners only
    <stem>_keys.json     — full per-key calibration

Falls back with a clear error if auto-detection fails — drop back to
manual_calibrate.py in that case.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from seg_to_keys import warp_to_piano, load_image
from key_labeler import draw_labels_tight_crop
from calibration import build_calibration_data, save_calibration


def calibrate_frame(
    frame: np.ndarray,
    top_crop: int = 0,
    camera_id: str = "live",
    far_side: str = "right",
):
    """Auto-detect corners → warp → build keys.json dict (in-process, no IO).

    Returns (keys_dict, warped_img, corners) or None if detection failed.
    top_crop trims pixels off the top of the warped strip first; the
    returned corners are recomputed so the dict's warp matches the trim.

    far_side: which direction the camera looks AWAY from the keyboard.
    Side-view cams from the left of the keyboard see camera-far on the
    right ("right", default); cams on the right see camera-far on the
    left ("left"). Affects which side's outer-piece contour gets
    projected onto inner pieces in merged-blob splitting.
    """
    warped, warp_trans, corners = warp_to_piano(frame, debug=False)
    if warp_trans is None or corners is None or warped.size == 0:
        return None
    corners = np.asarray(corners, dtype=np.float32)
    if 0 < top_crop < warped.shape[0]:
        H_orig, W_orig = warped.shape[:2]
        dst_full = np.array(
            [[0, 0], [W_orig - 1, 0], [W_orig - 1, H_orig - 1], [0, H_orig - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(dst_full, corners)
        new_top = cv2.perspectiveTransform(
            np.array([[[0, top_crop]], [[W_orig - 1, top_crop]]], dtype=np.float32), M
        ).reshape(2, 2)
        corners = np.array(
            [new_top[0], new_top[1], corners[2], corners[3]], dtype=np.float32
        )
        warped = warped[top_crop:]
    keys_dict = build_calibration_data(
        warped, corners, far_side="right", camera_id=camera_id
    )
    return keys_dict, warped, corners


def main():
    if len(sys.argv) < 2:
        print("usage: uv run python auto_calibrate.py path/to/snapshot.png [--top-crop N]")
        raise SystemExit(2)

    img_path = Path(sys.argv[1])
    top_crop = 0
    if "--top-crop" in sys.argv:
        i = sys.argv.index("--top-crop")
        top_crop = int(sys.argv[i + 1])

    img = load_image(str(img_path))
    result = calibrate_frame(img, top_crop=top_crop, camera_id=img_path.stem)
    if result is None:
        print("auto-detection failed (no keyboard blob found).")
        print("fall back to: uv run python manual_calibrate.py", img_path)
        raise SystemExit(1)
    keys_data, warped, corners = result

    print("auto-detected corners (TL, TR, BR, BL) in original coords:")
    for label, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"  {label}: ({x:.1f}, {y:.1f})")

    out_stem = img_path.with_suffix("")
    warped_path = Path(f"{out_stem}_warped.png")
    labeled_path = Path(f"{out_stem}_labeled.png")
    calib_path = Path(f"{out_stem}_calib.json")
    keys_path = Path(f"{out_stem}_keys.json")

    labeled = draw_labels_tight_crop(warped)
    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(labeled_path), labeled)
    calib_path.write_text(json.dumps({
        "image": str(img_path),
        "corners_tl_tr_br_bl": corners.tolist(),
    }, indent=2))
    save_calibration(keys_data, keys_path)

    print(
        f"wrote: {warped_path}\n       {labeled_path}\n"
        f"       {calib_path}\n       {keys_path}"
    )

    stack = np.vstack([warped, labeled])
    cv2.imshow("auto-calibrate result — any key to close", stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
