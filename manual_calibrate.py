"""Manual 4-corner calibration tool.

Usage:
    uv run python manual_calibrate.py path/to/photo.jpg

Click 4 corners in order: top-left, top-right, bottom-right, bottom-left
of the key surface region (just the key tops, no body, no front face).
Press 'r' to reset, ESC to abort. On the 4th click, the script warps the
image and runs key detection, saving the result and the calibration.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from key_labeler import load_image, draw_labels_tight_crop


MAX_DISPLAY = 1400  # max dimension for the display window


def pick_corners(img: np.ndarray) -> np.ndarray:
    """Show scaled image, collect 4 clicks (in original-image coords)."""
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DISPLAY / max(h, w))
    display = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img.copy()
    base = display.copy()

    pts_disp: list[tuple[int, int]] = []
    labels = ["TL", "TR", "BR", "BL"]

    def redraw():
        display[:] = base
        for i, (x, y) in enumerate(pts_disp):
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display, labels[i], (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if len(pts_disp) > 1:
            closed = pts_disp + ([pts_disp[0]] if len(pts_disp) == 4 else [])
            for a, b in zip(closed, closed[1:]):
                cv2.line(display, a, b, (0, 255, 0), 2)
        hint = f"click: {labels[len(pts_disp)] if len(pts_disp) < 4 else 'DONE — any key to continue'}"
        cv2.putText(display, hint, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, hint, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, "r=reset  ESC=abort", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, "r=reset  ESC=abort", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_disp) < 4:
            pts_disp.append((x, y))
            redraw()

    win = "calibrate — click TL, TR, BR, BL"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        cv2.imshow(win, display)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow(win)
            raise SystemExit("aborted")
        if k == ord("r"):
            pts_disp.clear()
            redraw()
        if len(pts_disp) == 4 and k != 255:
            break

    cv2.destroyWindow(win)
    pts_orig = np.array([(x / scale, y / scale) for (x, y) in pts_disp], dtype=np.float32)
    return pts_orig


def warp_from_corners(img: np.ndarray, corners: np.ndarray,
                       out_height: int = 220) -> np.ndarray:
    """Warp so the quad becomes a clean rectangle. Output width scaled to
    preserve the corners' top-edge length at out_height scale."""
    tl, tr, br, bl = corners
    top_len = float(np.linalg.norm(tr - tl))
    bot_len = float(np.linalg.norm(br - bl))
    avg_w = (top_len + bot_len) / 2
    left_len = float(np.linalg.norm(bl - tl))
    right_len = float(np.linalg.norm(br - tr))
    avg_h = (left_len + right_len) / 2
    out_w = max(100, int(avg_w * out_height / max(1.0, avg_h)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_height - 1],
                    [0, out_height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_height))


def main():
    if len(sys.argv) < 2:
        print("usage: uv run python manual_calibrate.py path/to/photo.jpg")
        raise SystemExit(2)

    img_path = Path(sys.argv[1])
    img = load_image(str(img_path))

    corners = pick_corners(img)
    print(f"corners (TL,TR,BR,BL) in original coords:")
    for label, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"  {label}: ({x:.1f}, {y:.1f})")

    warped = warp_from_corners(img, corners, out_height=220)
    labeled = draw_labels_tight_crop(warped)

    out_stem = img_path.with_suffix("")
    warped_path = Path(f"{out_stem}_warped.png")
    labeled_path = Path(f"{out_stem}_labeled.png")
    calib_path = Path(f"{out_stem}_calib.json")

    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(labeled_path), labeled)
    calib_path.write_text(json.dumps({
        "image": str(img_path),
        "corners_tl_tr_br_bl": corners.tolist(),
    }, indent=2))

    print(f"wrote: {warped_path}\n       {labeled_path}\n       {calib_path}")

    # Display result
    stack = np.vstack([warped, labeled])
    cv2.imshow("result — any key to close", stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
