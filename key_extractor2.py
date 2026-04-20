# /Users/billyan/Documents/2026/CSCI1430/uv run key_extractor2.py --mode image --image test_piano.avif
# for a static image
# /Users/billyan/Documents/2026/CSCI1430/uv run key_extractor2.py --mode live
# for a live videofeed

import argparse
import cv2
import numpy as np

from test_hough import (
    load_image,
    annotate_frame,
    make_mosaic,
    stream_hough,
    open_canon_streams,
)


# ── Key detector helpers ─────────────────────────────────────────────────────

def find_keyboard_y_bounds(warped: np.ndarray) -> tuple[int, int, int]:
    """
    Returns:
        y_black_top, y_black_bottom, y_white_bottom
    """
    if warped is None or warped.size == 0:
        return 0, 0, 0

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, _ = gray.shape

    row_mean = gray.mean(axis=1)
    grad = np.abs(np.diff(row_mean))
    if grad.size == 0:
        return 0, max(1, h // 5), max(2, h // 3)

    search_bottom = max(10, int(0.55 * h))
    grad_roi = grad[:search_bottom]

    candidate_rows = np.argsort(grad_roi)[::-1][:40]
    candidate_rows = np.sort(candidate_rows)

    y_black_top = int(0.02 * h)
    y_black_bottom = int(0.18 * h)
    y_white_bottom = int(0.32 * h)

    for r in candidate_rows:
        if 0 <= r <= int(0.10 * h):
            y_black_top = int(r)
            break

    for r in candidate_rows:
        if int(0.08 * h) <= r <= int(0.24 * h):
            y_black_bottom = int(r)
            break

    for r in candidate_rows:
        if int(0.18 * h) <= r <= int(0.45 * h):
            y_white_bottom = int(r)
            break

    y_black_top = max(0, min(h - 3, y_black_top))
    y_black_bottom = max(y_black_top + 2, min(h - 2, y_black_bottom))
    y_white_bottom = max(y_black_bottom + 2, min(h - 1, y_white_bottom))

    return y_black_top, y_black_bottom, y_white_bottom


def detect_keyboard_x_bounds(warped: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    if warped is None or warped.size == 0:
        return 0, 0

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    band = gray[y0:y1, :]
    if band.size == 0:
        return 0, w - 1

    col_mean = band.mean(axis=0)
    thresh = max(140, int(np.percentile(col_mean, 60)))
    bright = (col_mean >= thresh).astype(np.uint8)

    kernel = np.ones(15, dtype=np.uint8)
    smooth = np.convolve(bright, kernel, mode="same")
    mask = smooth >= 8

    xs = np.where(mask)[0]
    if len(xs) == 0:
        return 0, w - 1

    return int(xs[0]), int(xs[-1])


def detect_black_key_boxes(warped: np.ndarray, y_black_top: int, y_black_bottom: int) -> list[tuple[int, int, int, int]]:
    if warped is None or warped.size == 0:
        return []

    roi = warped[y_black_top:y_black_bottom, :]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_h, roi_w = gray.shape

    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bh / max(bw, 1)

        if area < 120:
            continue
        if bh < 0.45 * roi_h:
            continue
        if bw < 4:
            continue
        if bw > 0.06 * roi_w:
            continue
        if aspect < 1.6:
            continue

        boxes.append((int(x), int(y + y_black_top), int(bw), int(bh)))

    boxes.sort(key=lambda b: b[0] + b[2] / 2.0)

    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        px, py, pw, ph = merged[-1]
        pcx = px + pw / 2.0
        cx = box[0] + box[2] / 2.0

        if abs(cx - pcx) < 0.6 * max(pw, box[2]):
            prev_area = pw * ph
            curr_area = box[2] * box[3]
            merged[-1] = box if curr_area > prev_area else merged[-1]
        else:
            merged.append(box)

    return merged


def moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
    if x.ndim != 1:
        x = x.reshape(-1)
    if k <= 1:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def merge_close_positions(xs: list[int], min_sep: int) -> list[int]:
    if not xs:
        return []
    xs = sorted(xs)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= min_sep:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [int(round(np.mean(g))) for g in groups]

def draw_warped_key_lines_only(warped: np.ndarray) -> np.ndarray:
    """
    Clean visualization:
    only the warped image with detected white-key boundaries in yellow.
    """
    if warped is None or warped.size == 0:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    out = warped.copy()
    h, w = out.shape[:2]

    y_black_top, y_black_bottom, y_white_bottom = find_keyboard_y_bounds(out)

    white_band_top = min(h - 1, y_black_bottom + 2)
    white_band_bottom = min(h, max(white_band_top + 1, y_white_bottom - 2))
    x0, x1 = detect_keyboard_x_bounds(out, white_band_top, white_band_bottom)

    if x1 <= x0:
        return out

    white_h = y_white_bottom - y_black_bottom
    edge_y0 = y_black_bottom + int(0.55 * white_h)
    edge_y1 = y_black_bottom + int(0.95 * white_h)

    boundaries, _ = detect_white_key_boundaries_from_edges(
        out,
        x0=x0,
        x1=x1,
        y0=edge_y0,
        y1=edge_y1,
    )

    for bx in boundaries:
        cv2.line(out, (bx, 0), (bx, h - 1), (0, 255, 255), 2)

    return out

def detect_white_key_boundaries_from_edges(
    warped: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> tuple[list[int], np.ndarray]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))

    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        dbg = np.zeros((80, max(1, x1 - x0), 3), dtype=np.uint8)
        return [x0, x1], dbg

    sobel_x = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    edge_strength = np.mean(np.abs(sobel_x), axis=0)
    edge_strength = moving_average_1d(edge_strength, 9)

    med = float(np.median(edge_strength))
    p90 = float(np.percentile(edge_strength, 90))
    thresh = med + 0.25 * max(1.0, (p90 - med))

    peaks = []
    for i in range(1, len(edge_strength) - 1):
        c = edge_strength[i]
        if c >= thresh and c >= edge_strength[i - 1] and c >= edge_strength[i + 1]:
            peaks.append(i + x0)

    peaks = merge_close_positions(peaks, min_sep=6)

    if len(peaks) >= 3:
        gaps = np.diff(peaks)
        typical = float(np.median(gaps))
        filtered = [peaks[0]]
        for p in peaks[1:]:
            gap = p - filtered[-1]
            if gap >= max(4, int(0.45 * typical)):
                filtered.append(p)
        peaks = filtered

    boundaries = merge_close_positions([x0] + peaks + [x1], min_sep=4)

    dbg_h = 80
    dbg_w = max(1, x1 - x0)
    dbg = np.zeros((dbg_h, dbg_w, 3), dtype=np.uint8)

    sig = edge_strength.copy()
    if sig.size > 0 and float(sig.max()) > 0:
        sig = sig / float(sig.max())

    for i in range(len(sig) - 1):
        y_a = dbg_h - 1 - int(sig[i] * (dbg_h - 10))
        y_b = dbg_h - 1 - int(sig[i + 1] * (dbg_h - 10))
        cv2.line(dbg, (i, y_a), (i + 1, y_b), (0, 255, 255), 1)

    max_strength = float(edge_strength.max()) if edge_strength.size > 0 else 0.0
    if max_strength > 0:
        thr_norm = min(1.0, thresh / max_strength)
        y_thr = dbg_h - 1 - int(thr_norm * (dbg_h - 10))
        cv2.line(dbg, (0, y_thr), (dbg_w - 1, y_thr), (0, 0, 255), 1)

    for b in boundaries:
        bx = b - x0
        if 0 <= bx < dbg_w:
            cv2.line(dbg, (bx, 0), (bx, dbg_h - 1), (255, 255, 0), 1)

    return boundaries, dbg


def draw_warped_key_detector(warped: np.ndarray) -> np.ndarray:
    if warped is None or warped.size == 0:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    out = warped.copy()
    h, w = out.shape[:2]

    y_black_top, y_black_bottom, y_white_bottom = find_keyboard_y_bounds(out)

    white_band_top = min(h - 1, y_black_bottom + 2)
    white_band_bottom = min(h, max(white_band_top + 1, y_white_bottom - 2))
    x0, x1 = detect_keyboard_x_bounds(out, white_band_top, white_band_bottom)

    black_boxes = detect_black_key_boxes(out, y_black_top, y_black_bottom)

    cv2.line(out, (0, y_black_top), (w - 1, y_black_top), (0, 0, 255), 2)
    cv2.line(out, (0, y_black_bottom), (w - 1, y_black_bottom), (0, 0, 255), 2)
    cv2.line(out, (0, y_white_bottom), (w - 1, y_white_bottom), (0, 0, 255), 2)

    if x1 <= x0:
        return out

    white_h = y_white_bottom - y_black_bottom
    edge_y0 = y_black_bottom + int(0.55 * white_h)
    edge_y1 = y_black_bottom + int(0.95 * white_h)

    boundaries, signal_dbg = detect_white_key_boundaries_from_edges(
        out,
        x0=x0,
        x1=x1,
        y0=edge_y0,
        y1=edge_y1,
    )

    for bx in boundaries:
        cv2.line(out, (bx, y_black_bottom), (bx, y_white_bottom), (0, 255, 255), 2)

    for x, y, bw, bh in black_boxes:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

    inset_h, inset_w = signal_dbg.shape[:2]
    if inset_w > 10 and inset_h > 10:
        inset_w = min(inset_w, w - 10)
        signal_dbg = cv2.resize(signal_dbg, (inset_w, inset_h))
        y_start = min(h - inset_h - 5, y_white_bottom + 5)
        x_start = 5
        if y_start >= 0 and y_start + inset_h <= h and x_start + inset_w <= w:
            out[y_start:y_start + inset_h, x_start:x_start + inset_w] = signal_dbg
            cv2.rectangle(out, (x_start, y_start), (x_start + inset_w, y_start + inset_h), (255, 255, 255), 1)

    return out


# ── Wrapper modes ────────────────────────────────────────────────────────────

def run_image_mode(image_path: str, window_name: str = "hough_image") -> None:
    frame = load_image(image_path)

    threshed, blurred, edges, output, key_output, warped = annotate_frame(frame)
    warped_labeled = draw_warped_key_detector(warped)
    warped_lines_only = draw_warped_key_lines_only(warped)

    mosaic = make_mosaic([
        ("threshed", threshed),
        ("blurred", blurred),
        ("edges", edges),
        ("all lines", output),
        ("key lines", key_output),
        ("warped", warped),
        ("warped + key labels", warped_labeled),
        ("warped + yellow lines only", warped_lines_only),
    ])

    cv2.imshow(window_name, mosaic)
    print("Showing static image result. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live_mode(window_name: str = "hough_stream") -> None:
    streams = open_canon_streams(silent=False)
    if not streams:
        print("No camera streams found.")
        return

    for stream in streams:
        stream.start()
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                print("Failed to read from camera")
                break

            threshed, blurred, edges, output, key_output, warped = annotate_frame(frame)
            warped_labeled = draw_warped_key_detector(warped)

            mosaic = make_mosaic([
                ("threshed", threshed),
                ("blurred", blurred),
                ("edges", edges),
                ("all lines", output),
                ("key lines", key_output),
                ("warped", warped),
                ("warped + key labels", warped_labeled),
            ])

            cv2.imshow(window_name, mosaic)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        stream.stop()

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Key detector wrapper for original test_hough.py")
    parser.add_argument(
        "--mode",
        choices=["image", "live"],
        default="live",
        help="Run on a static image or live camera stream.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a static image when using --mode image",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default=None,
        help="Optional OpenCV window name.",
    )
    args = parser.parse_args()

    if args.mode == "image":
        if not args.image:
            raise ValueError("In image mode, provide --image PATH_TO_IMAGE")
        run_image_mode(
            image_path=args.image,
            window_name=args.window_name or "hough_image",
        )
    else:
        run_live_mode(
            window_name=args.window_name or "hough_stream",
        )


if __name__ == "__main__":
    main()