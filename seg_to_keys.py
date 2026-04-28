import cv2
import numpy as np
from PIL import Image
import sys
from stream_webcams import CanonStream, open_canon_streams

#
# Hyperparameters
#

#BRIGHTNESS_THRESHOLD = 0.6 # brightness threshold for white detection
WHITE_PEAK_TOLERANCE = 30  # ±pixel value tolerance around each channel's brightest histogram peak

GAUSSIAN_KERNEL   = (5, 5)   # blur kernel size (must be odd)
GAUSSIAN_SIGMA    = 1.0      # blur sigma

MERGE_DIST = 5 # distance to merge blobs before isolating one

HOUGH_RHO         = 1        # distance resolution (pixels)
HOUGH_THETA       = np.pi / 180  # angle resolution (radians)
HOUGH_THRES_MULT  = 0.1       # minimum accumulator votes to report a line (* im_width)
HOUGH_ML_MULT     = 0.5       # minimum segment length (pixels) (* im_width)
HOUGH_GAP_MULT    = 0.05       # maximum gap to bridge within a segment (pixels) (* im_width)

N_LINES           = 20       # number of longest lines to accept

RANSAC_THRESH = 0.2 # minimum percent of input points a ransac line must claim as inliers to be accepted

PARALLEL_TOL = 5 # degrees off the heading of two lines can be to be considered parallel
CROP_PADDING = 0 # pixels of padding to add to warped crop

# Debug params
LINE_COLOR        = (0, 0, 255)   # BGR
LINE_COLOR2       = (0, 255, 0)
LINE_THICKNESS    = 2
MOSAIC_COLS       = 3            # tiles per row in the debug mosaic
MOSAIC_CELL_W     = 640          # width of each tile (pixels)
MOSAIC_CELL_H     = 360          # height of each tile (pixels)

# 
# Debug Helpers
#

def _draw_hist_debug(smoothed: np.ndarray, peak: int) -> np.ndarray:
    """Render the L-channel smoothed histogram with the selected peak marked."""
    W, H = 512, 256
    pad_b, pad_t = 24, 12
    vis = np.full((H, W, 3), 20, dtype=np.uint8)
    plot_h = H - pad_b - pad_t
    max_val = smoothed.max()
    if max_val == 0:
        return vis
    pts = np.array([
        (int(round(i * (W - 1) / 255)),
         pad_t + plot_h - int(round(smoothed[i] / max_val * plot_h)))
        for i in range(256)
    ], dtype=np.int32)
    cv2.polylines(vis, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
    px = int(round(peak * (W - 1) / 255))
    cv2.line(vis, (px, pad_t), (px, H - pad_b), (0, 255, 255), 2)
    label = f'L peak: {peak}  tol: ±{WHITE_PEAK_TOLERANCE}'
    cv2.putText(vis, label, (min(px + 4, W - 180), pad_t + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (min(px + 4, W - 180), pad_t + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    # X-axis
    cv2.line(vis, (0, H - pad_b), (W - 1, H - pad_b), (160, 160, 160), 1)
    for v in [0, 64, 128, 192, 255]:
        x = int(round(v * (W - 1) / 255))
        cv2.line(vis, (x, H - pad_b), (x, H - pad_b + 4), (160, 160, 160), 1)
        cv2.putText(vis, str(v), (x - 8, H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    return vis

def make_mosaic(named_frames: list[tuple[str, np.ndarray]]) -> np.ndarray:
    cells = []
    for label, img in named_frames:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cell = cv2.resize(img, (MOSAIC_CELL_W, MOSAIC_CELL_H))
        cv2.putText(cell, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    blank = np.zeros((MOSAIC_CELL_H, MOSAIC_CELL_W, 3), dtype=np.uint8)
    while len(cells) % MOSAIC_COLS != 0:
        cells.append(blank)
    rows = [np.hstack(cells[i:i + MOSAIC_COLS])
            for i in range(0, len(cells), MOSAIC_COLS)]
    return np.vstack(rows)

#
# Geometry Helpers
#

def segment_length(seg):
        x1, y1, x2, y2 = seg[0]
        return np.hypot(x2 - x1, y2 - y1)

def segment_angle(seg):
    x1, y1, x2, y2 = seg[0]
    return np.arctan2(y2 - y1, x2 - x1)

def angle_diff(a1, a2):
    diff = abs(a1 - a2) % np.pi
    return min(diff, np.pi - diff)

def point_to_line_dist(px, py, seg):
    x1, y1, x2, y2 = seg[0]
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return np.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length

def intersect(l_h: tuple[float], l_v: tuple[float]):
    m_h, b_h = l_h
    m_v, b_v = l_v
    # y = m_h*x + b_h   and   x = m_v*y + b_v
    # => x = m_v*(m_h*x + b_h) + b_v
    # => x*(1 - m_v*m_h) = m_v*b_h + b_v
    denom = 1.0 - m_v * m_h
    if abs(denom) < 1e-9:
        return None
    x = (m_v * b_h + b_v) / denom
    y = m_h * x + b_h
    return np.array([x, y])

#
# Main Helpers
#

def isolate_white(frame: np.ndarray) -> np.ndarray:
    """Return a white-key mask (BGR, white=key, black=background).

    Converts to LAB, finds the rightmost peak in the L-channel histogram,
    and masks pixels whose L value falls within WHITE_PEAK_TOLERANCE of it.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    n_pixels = frame.shape[0] * frame.shape[1]
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256]).flatten()
    smoothed = np.convolve(hist, np.ones(11) / 11, mode='same')
    min_height = n_pixels * 0.0015
    peak_idxs = [
        i for i in range(1, 255)
        if smoothed[i] > smoothed[i - 1]
        and smoothed[i] > smoothed[i + 1]
        and smoothed[i] >= min_height
    ]
    peak = peak_idxs[-1] if peak_idxs else int(np.argmax(smoothed))
    l = l_channel.astype(np.int16)
    mask = (l >= peak - WHITE_PEAK_TOLERANCE) & (l <= peak + WHITE_PEAK_TOLERANCE)
    result = np.where(np.stack([mask] * 3, axis=2), 255, 0).astype(np.uint8)
    # cv2.imshow("L-channel histogram", _draw_hist_debug(smoothed, peak))
    # cv2.waitKey(0)
    return result

def isolate_key_blob(frame: np.ndarray) -> np.ndarray:
    #smeared = cv2.dilate(frame, np.ones((1, 20), np.uint8), iterations=1)
    smeared = frame
    # Merge contours within MERGE_DISTpx of each other by dilating, then pick the largest.
    merged = cv2.dilate(smeared, np.ones((MERGE_DIST * 2 + 1, MERGE_DIST * 2 + 1), np.uint8))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(frame)
    contour = max(contours, key=cv2.contourArea)
    # Fill the merged contour on the un-dilated smeared mask to preserve geometry.
    blob = np.zeros_like(frame)
    cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)
    blob = cv2.bitwise_and(blob, frame)
    return blob

def extrema(mask: np.ndarray):
    """Per-column top/bottom y and per-row left/right x extrema, or -1 where empty."""
    h, w = mask.shape
    present = mask > 0
    # Per-column: topmost y (first True) and bottommost y (last True).
    top_idx = np.where(present.any(axis=0), present.argmax(axis=0), -1)
    flipped_v = present[::-1]
    bot_idx = np.where(present.any(axis=0), h - 1 - flipped_v.argmax(axis=0), -1)
    xs = np.arange(w)
    top_pts = np.stack([xs[top_idx >= 0], top_idx[top_idx >= 0]], axis=1)
    bot_pts = np.stack([xs[bot_idx >= 0], bot_idx[bot_idx >= 0]], axis=1)
    # Per-row: leftmost x (first True) and rightmost x (last True).
    left_idx = np.where(present.any(axis=1), present.argmax(axis=1), -1)
    flipped_h = present[:, ::-1]
    right_idx = np.where(present.any(axis=1), w - 1 - flipped_h.argmax(axis=1), -1)
    ys = np.arange(h)
    left_pts  = np.stack([ys[left_idx >= 0],  left_idx[left_idx >= 0]],  axis=1)
    right_pts = np.stack([ys[right_idx >= 0], right_idx[right_idx >= 0]], axis=1)
    return top_pts, bot_pts, left_pts, right_pts

def find_multiple_lines(points: np.ndarray, num_lines=2, inlier_tol=3.0):
    remaining_points = points.copy()
    minimum_acceptable_inliers = len(points)*RANSAC_THRESH
    detected_lines = []
    for _ in range(num_lines):
        result = ransac_line(remaining_points, inlier_tol=inlier_tol)
        if result is None or len(result[2]) < minimum_acceptable_inliers: break
        detected_lines.append(result)
        remaining_points = remaining_points[~result[2]]
        if len(remaining_points) < minimum_acceptable_inliers: break
            
    return detected_lines

def ransac_line(points: np.ndarray, iters: int = 200, inlier_tol: float = 3.0, rng=None):
    """Fit a line y = m*x + b to 2D points via RANSAC. Returns (m, b, inliers_mask)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(points)
    if n < 2:
        return None
    best_inliers = None
    best_count = 0
    xs = points[:, 0]
    ys = points[:, 1]
    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if i == j: continue
        xi, yi = points[i]
        xj, yj = points[j]
        if xj == xi: continue
        m = (yj - yi) / (xj - xi)
        b = yi - m * xi
        dist = np.abs(ys - (m * xs + b))
        inliers = dist <= inlier_tol
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 2: return None
    # Least-squares refit on inliers.
    xin = xs[best_inliers]
    yin = ys[best_inliers]
    A = np.vstack([xin, np.ones_like(xin)]).T
    m, b = np.linalg.lstsq(A, yin, rcond=None)[0]
    return float(m), float(b), best_inliers

def warp_key_lines(frame: np.ndarray, line_a: np.ndarray, line_b: np.ndarray, padding: int = CROP_PADDING) -> np.ndarray:
    """Perspective-warp frame so that line_a and line_b form a rectangle and crop to this (padded) rectangle"""
    ax1, ay1, ax2, ay2 = line_a.astype(float)
    bx1, by1, bx2, by2 = line_b.astype(float)

    # Unit vector along line_a
    dax, day = ax2 - ax1, ay2 - ay1
    len_a = np.hypot(dax, day)
    if len_a == 0:
        return np.zeros((2 * padding, 2 * padding, 3), dtype=np.uint8)
    ux, uy = dax / len_a, day / len_a

    # Project line_b endpoints onto line_a direction for consistent left/right ordering
    tb1 = (bx1 - ax1) * ux + (by1 - ay1) * uy
    tb2 = (bx2 - ax1) * ux + (by2 - ay1) * uy
    if tb1 <= tb2:
        b_left, b_right = np.float32([bx1, by1]), np.float32([bx2, by2])
    else:
        b_left, b_right = np.float32([bx2, by2]), np.float32([bx1, by1])

    a_left  = np.float32([ax1, ay1])
    a_right = np.float32([ax2, ay2])

    # Order into TL, TR, BR, BL by which line sits higher (smaller mean y)
    if (ay1 + ay2) / 2 <= (by1 + by2) / 2:
        src = np.float32([a_left, a_right, b_right, b_left])
    else:
        src = np.float32([b_left, b_right, a_right, a_left])

    # Destination rectangle: width = len_a, height = perp distance between lines
    mid_bx, mid_by = (bx1 + bx2) / 2, (by1 + by2) / 2
    h = int(max(1, abs(day * (mid_bx - ax1) - dax * (mid_by - ay1)) / len_a))
    w = int(len_a)

    dst = np.float32([
        [padding,         padding    ],
        [padding + w,     padding    ],
        [padding + w,     padding + h],
        [padding,         padding + h],
    ])

    # Apply perspective warp and return
    M = cv2.getPerspectiveTransform(src, dst)
    return M, cv2.warpPerspective(frame, M, (w + 2 * padding, h + 2 * padding))

def warp_to_piano(frame: np.ndarray, debug=False) -> np.ndarray:
    """
    Takes a more or less top down shot of a piano and warps + crops it to just its keys.

    Isolate white pixels → Grayscale → Gaussian blur → Horizontal dilation (smears white keys) 
    → Canny edges → Bidirectional dilation (thicken lines) → Hough lines (find lines from edges)
    → Isolate keyboard bounding lines → Projective warp.
    """
    threshed = isolate_white(frame) # Isolate white keys
    gray = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY) # Grayscale (prepare for edge detection)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA) # Gaussian blur (remove noise)
    #smeared = cv2.dilate(blurred, np.ones((1, 20), np.uint8), iterations=1) # Horizontal dilate (smear white keys into a more uniform brick)
    blob = isolate_key_blob(blurred)
    # edges = cv2.Canny(blob, 50, 150) # Canny edge detection
    # thick_edges = cv2.dilate(edges, np.ones((45, 1), np.uint8), iterations=1) # Bidirectional dilate (thicken detected edges for hough)

    extreme_idcs = extrema(blob)
    extreme_lines = []
    for idx, idcs in enumerate(extreme_idcs):
        lines = find_multiple_lines(idcs, num_lines=3)
        if len(lines) == 0:
            return (frame, threshed, blob, lines_vis, corners_vis, np.zeros_like(frame)) if debug else np.zeros_like(frame)
        elif len(lines) == 1:
            extreme_lines.append(lines[0][:2])
        else:
            line1_inliers = lines[0][2]
            line2_inliers = lines[1][2]
            line1_avg = np.mean(extreme_idcs[idx][line1_inliers], axis=0)
            line2_avg = np.mean(extreme_idcs[idx][~line1_inliers][line2_inliers], axis=0)
            # col 1 is y for top/bot pts (x,y) and x for left/right pts (row,x)
            if idx == 0:   # top:   smallest avg y
                selected_line = lines[0] if line1_avg[1] <= line2_avg[1] else lines[1]
            elif idx == 1: # bot:   greatest avg y
                selected_line = lines[0] if line1_avg[1] >= line2_avg[1] else lines[1]
            elif idx == 2: # left:  smallest avg x
                selected_line = lines[0] if line1_avg[1] <= line2_avg[1] else lines[1]
            else:          # right: greatest avg x
                selected_line = lines[0] if line1_avg[1] >= line2_avg[1] else lines[1]
            extreme_lines.append(selected_line[:2])
    corners = [
        intersect(extreme_lines[0], extreme_lines[2]), #tl
        intersect(extreme_lines[0], extreme_lines[3]), #tr
        intersect(extreme_lines[1], extreme_lines[2]), #bl
        intersect(extreme_lines[1], extreme_lines[3])  #br
    ]
    lines_vis = np.zeros_like(frame)
    corners_vis = np.zeros_like(frame)
    if debug:
        H_f, W_f = frame.shape[:2]
        lines_vis = frame.copy()
        line_colors = [(0, 255, 255), (0, 165, 255), (255, 0, 255), (0, 255, 128)]
        line_labels = ["top", "bot", "left", "right"]
        for (m, b), color, label in zip(extreme_lines, line_colors, line_labels):
            if label in ("top", "bot"):
                # y = m*x + b — draw across full width
                p1 = (0, int(round(b)))
                p2 = (W_f - 1, int(round(m * (W_f - 1) + b)))
            else:
                # x = m*y + b — draw across full height
                p1 = (int(round(b)), 0)
                p2 = (int(round(m * (H_f - 1) + b)), H_f - 1)
            cv2.line(lines_vis, p1, p2, color, 2, cv2.LINE_AA)
            cv2.putText(lines_vis, label, ((p1[0] + p2[0]) // 2 + 5, (p1[1] + p2[1]) // 2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        corners_vis = frame.copy()
        pts = [(int(round(x)), int(round(y))) for x, y in corners]
        for pt in pts:
            cv2.circle(corners_vis, pt, 10, (0, 255, 0), -1)
        for label, pt in zip(["TL", "TR", "BL", "BR"], pts):
            cv2.putText(corners_vis, label, (pt[0] + 12, pt[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.polylines(corners_vis, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 0), 2)                                                                                                                        

    first_rail = np.concatenate([corners[0], corners[1]])
    second_rail = np.concatenate([corners[2], corners[3]])
    warp_trans, warped = warp_key_lines(frame, first_rail, second_rail)
        
    return frame, threshed, blob, lines_vis, corners_vis, warped if debug else warped


def stream_to_piano(stream: CanonStream, window_name: str = "keyboard_stream"):
    """Display a CanonStream with Hough line annotations until ESC is pressed."""
    stream.start()
    while True:
        grabbed, frame = stream.read()
        if not grabbed or frame is None:
            print("Failed to read from camera")
            break
        original, threshed, blob, lines_vis, corners_vis, warped = warp_to_piano(frame, debug=True)
        mosaic = make_mosaic([
            ("original", original),
            ("threshed", threshed),
            ("blob", blob),
            ("lines", lines_vis),
            ("corners", corners_vis),
            ("warped", warped),
        ])
        cv2.imshow(window_name, mosaic)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    stream.stop()
    cv2.destroyAllWindows()

def pics_to_piano(paths: list[str], window_name: str = "keyboard_stream"):
    for path in paths:
        img = load_image(path)
        original, threshed, blob, lines_vis, corners_vis, warped = warp_to_piano(img, debug=True)
        mosaic = make_mosaic([
            ("original", original),
            ("threshed", threshed),
            ("blob", blob),
            ("lines", lines_vis),
            ("corners", corners_vis),
            ("warped", warped),
        ])
        cv2.imshow(window_name, mosaic)
        cv2.waitKey(0)

if __name__ == "__main__":
    streams = open_canon_streams(allow_iphone=True, silent=False)
    for stream in streams:
        stream_to_piano(stream)
        
def load_image(path: str) -> np.ndarray:
    """Load an image by path. Falls back to PIL for formats OpenCV may not
    ship with (e.g. AVIF on some builds)."""
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image

    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

# if __name__ == "__main__":
#     pics_to_piano(sys.argv[1:] if len(sys.argv) > 1 else [
#         "piano_photos/IMG_9064.jpg",
#         "piano_photos/IMG_9066.jpg",
#         "piano_photos/IMG_9073.jpg",
#     ])
