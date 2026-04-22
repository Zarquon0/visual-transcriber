import cv2
import numpy as np
from PIL import Image
from stream_webcams import CanonStream, open_canon_streams

#
# Hyperparameters
#

BRIGHTNESS_THRESHOLD = 0.6 # brightness threshold for white detection

GAUSSIAN_KERNEL   = (5, 5)   # blur kernel size (must be odd)
GAUSSIAN_SIGMA    = 1.0      # blur sigma

HOUGH_RHO         = 1        # distance resolution (pixels)
HOUGH_THETA       = np.pi / 180  # angle resolution (radians)
HOUGH_THRES_MULT  = 0.1       # minimum accumulator votes to report a line (* im_width)
HOUGH_ML_MULT     = 0.5       # minimum segment length (pixels) (* im_width)
HOUGH_GAP_MULT    = 0.05       # maximum gap to bridge within a segment (pixels) (* im_width)

N_LINES           = 20       # number of longest lines to accept

PARALLEL_TOL = 5 # degrees off the heading of two lines can be to be considered parallel
CROP_PADDING = 20 # pixels of padding to add to warped crop

# Debug params
LINE_COLOR        = (0, 0, 255)   # BGR
LINE_COLOR2       = (0, 255, 0)
LINE_THICKNESS    = 2
MOSAIC_COLS       = 3            # tiles per row in the debug mosaic
MOSAIC_CELL_W     = 640          # width of each tile (pixels)
MOSAIC_CELL_H     = 360          # height of each tile (pixels)

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

#
# Main Helpers
#

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
    return cv2.warpPerspective(frame, M, (w + 2 * padding, h + 2 * padding))


def isolate_white(frame: np.ndarray) -> np.ndarray:
    img = frame.astype(np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = ch.min(), ch.max()
        img[:, :, c] = (ch - lo) / (hi - lo) if hi > lo else np.zeros_like(ch)
    thresh = BRIGHTNESS_THRESHOLD
    mask = (img[:, :, 0] >= thresh) & (img[:, :, 1] >= thresh) & (img[:, :, 2] >= thresh)
    mask = np.stack([mask, mask, mask], axis=2)
    img = np.where(mask, 255, 0).astype(np.uint8)
    return img

def find_other_rail(first_rail, other_segs: list[np.ndarray]):
    # Filter out non-parallel segments
    longest_angle = segment_angle(first_rail)
    parallel_tol_rad = PARALLEL_TOL * np.pi / 180
    parallel = [
        s for s in other_segs
        if angle_diff(segment_angle(s), longest_angle) <= parallel_tol_rad
    ]
    # Find the farthest of away of the parallel segments (by midpoint)
    if parallel:
        farthest = max(parallel, key=lambda s: point_to_line_dist(
            (s[0][0] + s[0][2]) / 2, (s[0][1] + s[0][3]) / 2, first_rail))
        other_rail = farthest
    else:
        other_rail = first_rail # Failure mode! Don't want to error though to end livestream

    return other_rail

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
    smeared = cv2.dilate(blurred, np.ones((1, 20), np.uint8), iterations=1) # Horizontal dilate (smear white keys into a more uniform brick)
    edges = cv2.Canny(smeared, 50, 150) # Canny edge detection
    thick_edges = cv2.dilate(edges, np.ones((15, 1), np.uint8), iterations=1) # Bidirectional dilate (thicken detected edges for hough)
    
    # Hough transform (find lines from edges)
    _, W = thick_edges.shape
    segments = cv2.HoughLinesP(
        thick_edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=int(W * HOUGH_THRES_MULT),
        minLineLength=int(W * HOUGH_ML_MULT),
        maxLineGap=int(W * HOUGH_GAP_MULT),
    )

    all_lines = frame.copy()
    keys = frame.copy()
    warped = np.zeros_like(frame)
    if segments is not None and len(segments) > 0:
        # Keep only near-horizontal segments so Hough can't pick vertical
        # key-separator edges as a "rail".
        horiz_tol_rad = 30 * np.pi / 180
        horizontal = [s for s in segments
                      if angle_diff(segment_angle(s), 0.0) <= horiz_tol_rad]
        segments = horizontal if horizontal else list(segments)
        # Isolate keyboard bounding lines
        top_segments = sorted(segments, key=segment_length, reverse=True)[:N_LINES]
        first_rail = top_segments[0]
        other_rail = find_other_rail(first_rail, top_segments[1:])
        # Apply projective warp and crop
        warped = warp_key_lines(frame, first_rail[0], other_rail[0])

        if debug:
            # all_lines lines
            for seg in top_segments:
                x1, y1, x2, y2 = seg[0]
                cv2.line(all_lines, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)
            # keys lines
            r1x1, r1y1, r1x2, r1y2 = first_rail[0]
            r2x1, r2y1, r2x2, r2y2 = other_rail[0]
            cv2.line(keys, (r1x1, r1y1), (r1x2, r1y2), LINE_COLOR, LINE_THICKNESS)
            cv2.line(keys, (r2x1, r2y1), (r2x2, r2y2), LINE_COLOR2, LINE_THICKNESS)
        
    return threshed, blurred, thick_edges, all_lines, keys, warped if debug else warped


def stream_to_piano(stream: CanonStream, window_name: str = "keyboard_stream"):
    """Display a CanonStream with Hough line annotations until ESC is pressed."""
    stream.start()
    while True:
        grabbed, frame = stream.read()
        if not grabbed or frame is None:
            print("Failed to read from camera")
            break
        threshed, blurred, edges, output, key_output, warped = warp_to_piano(frame, debug=True)
        mosaic = make_mosaic([
            ("threshed", threshed),
            ("blurred", blurred),
            ("edges", edges),
            ("all lines", output),
            ("key lines", key_output),
            ("warped", warped),
        ])
        cv2.imshow(window_name, mosaic)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    streams = open_canon_streams(silent=False)
    for stream in streams:
        stream_to_piano(stream)
