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
    layout=None,
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

    layout: KeyboardLayout instance controlling keyboard span (default
    LAYOUT_61KEY = C2-C7, 25 b / 36 w). Pass LAYOUT_25KEY for a
    25-key C4-C6 mini.
    """
    warped, warp_trans, corners = warp_to_piano(frame, debug=False)
    if warp_trans is None or corners is None or warped.size == 0:
        return None
    corners = np.asarray(corners, dtype=np.float32)
    # warp_to_piano can pick the keyboard's short edges as rails when
    # the source frame has the keyboard at a steep diagonal (e.g. an
    # MPK Mini sitting on a stool, photographed off-axis). The output
    # then comes out portrait — taller than wide — which breaks all
    # downstream segmentation (it assumes horizontal blacks-above-whites
    # layout). A keyboard is always wider than it is deep, so a portrait
    # warp means we picked the wrong axis. Rotate 90° clockwise to
    # restore the horizontal orientation, and rotate the corner
    # ordering accordingly (TL→TR→BR→BL ⇒ BL→TL→TR→BR).
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        corners = np.array(
            [corners[3], corners[0], corners[1], corners[2]], dtype=np.float32,
        )
    # After landscape orientation, check if the warp is upside-down
    # (blacks at bottom, whites at top). The top half of a correct warp
    # is dominated by black-key bodies (dark pixels) while the bottom
    # half is white-key surface (bright pixels). If the inverse is true,
    # rotate 180° and reorder corners (TL→TR→BR→BL ⇒ BR→BL→TL→TR).
    h_half = warped.shape[0] // 2
    if h_half > 0:
        gray_check = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        top_mean = float(gray_check[:h_half].mean())
        bot_mean = float(gray_check[h_half:].mean())
        if top_mean > bot_mean:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
            corners = np.array(
                [corners[2], corners[3], corners[0], corners[1]], dtype=np.float32,
            )
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
        warped, corners, far_side=far_side, camera_id=camera_id, layout=layout,
    )
    return keys_dict, warped, corners


def main():
    if len(sys.argv) < 2:
        print("usage: uv run python auto_calibrate.py path/to/snapshot.png "
              "[--top-crop N] [--layout 61|25] [--far-side right|left]")
        raise SystemExit(2)

    img_path = Path(sys.argv[1])
    top_crop = 0
    if "--top-crop" in sys.argv:
        i = sys.argv.index("--top-crop")
        top_crop = int(sys.argv[i + 1])

    from key_labeler import LAYOUT_61KEY, LAYOUT_25KEY
    layout = LAYOUT_61KEY
    if "--layout" in sys.argv:
        i = sys.argv.index("--layout")
        v = sys.argv[i + 1]
        layout = LAYOUT_25KEY if v == "25" else LAYOUT_61KEY

    far_side = "right"
    if "--far-side" in sys.argv:
        i = sys.argv.index("--far-side")
        far_side = sys.argv[i + 1]

    img = load_image(str(img_path))
    result = calibrate_frame(
        img, top_crop=top_crop, camera_id=img_path.stem,
        far_side=far_side, layout=layout,
    )
    if result is None:
        print("auto-detection failed (no keyboard blob found).")
        print("fall back to: uv run python manual_calibrate.py", img_path)
        raise SystemExit(1)
    keys_data, warped, corners = result

    print("auto-detected corners (TL, TR, BR, BL) in original coords:")
    for label, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"  {label}: ({x:.1f}, {y:.1f})")
    n_total = len(keys_data["keys"])
    n_blacks = sum(1 for k in keys_data["keys"] if k["type"] == "black")
    n_whites = n_total - n_blacks
    n_labeled = sum(1 for k in keys_data["keys"] if k.get("note", "?") not in ("?", ""))
    print(f"layout: n_octaves={layout.n_octaves} start_octave={layout.start_octave} "
          f"(expected {layout.n_blacks}b + {layout.n_whites}w = {layout.n_blacks + layout.n_whites})")
    print(f"detected: {n_total} keys ({n_blacks} black + {n_whites} white), "
          f"{n_labeled} labeled, {n_total - n_labeled} '?'")

    out_stem = img_path.with_suffix("")
    warped_path = Path(f"{out_stem}_warped.png")
    labeled_path = Path(f"{out_stem}_labeled.png")
    calib_path = Path(f"{out_stem}_calib.json")
    keys_path = Path(f"{out_stem}_keys.json")

    labeled = draw_labels_tight_crop(warped, far_side=far_side, layout=layout)
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
