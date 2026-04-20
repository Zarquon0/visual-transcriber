import cv2
import numpy as np
from PIL import Image
from stream_webcams import CanonStream, open_canon_streams

# ── Hyperparameters ──────────────────────────────────────────────────────────
BRIGHTNESS_THRESHOLD = 0.5 # brightness threshold for white detection

GAUSSIAN_KERNEL   = (5, 5)   # blur kernel size (must be odd)
GAUSSIAN_SIGMA    = 1.0      # blur sigma

SOBEL_KSIZE       = 3        # Sobel kernel size

HOUGH_RHO         = 1        # distance resolution (pixels)
HOUGH_THETA       = np.pi / 180  # angle resolution (radians)
HOUGH_THRESHOLD   = 50       # minimum accumulator votes to report a line
HOUGH_MIN_LENGTH  = 50       # minimum segment length (pixels)
HOUGH_MAX_GAP     = 10       # maximum gap to bridge within a segment (pixels)

N_LINES           = 20       # number of longest lines to overlay
LINE_COLOR        = (0, 0, 255)   # BGR
LINE_THICKNESS    = 2

OUTPUT_PATH       = "test_hough_output.png"

MOSAIC_COLS       = 3            # tiles per row in the debug mosaic
MOSAIC_CELL_W     = 640          # width of each tile (pixels)
MOSAIC_CELL_H     = 360          # height of each tile (pixels)
# ─────────────────────────────────────────────────────────────────────────────


def load_image(path: str) -> np.ndarray:
    """Load any image (including AVIF) as a BGR numpy array."""
    pil_img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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


def normalize_and_threshold(frame: np.ndarray) -> np.ndarray:
    img = frame.astype(np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = ch.min(), ch.max()
        img[:, :, c] = (ch - lo) / (hi - lo) if hi > lo else np.zeros_like(ch)
    thresh = 0.6
    mask = (img[:, :, 0] >= thresh) & (img[:, :, 1] >= thresh) & (img[:, :, 2] >= thresh)
    mask = np.stack([mask, mask, mask], axis=2)
    img = np.where(mask, 255, 0).astype(np.uint8)
    return img

def annotate_frame(frame: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur → Canny edges → HoughLinesP and return frame with top lines drawn."""
    threshed = normalize_and_threshold(frame)
    gray = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    kernel = np.ones((1, 20), np.uint8) # A horizontal kernel
    blurred = cv2.dilate(blurred, kernel, iterations=1)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((10, 1), np.uint8) 
    edges = cv2.dilate(edges, kernel, iterations=1)

    H, W = edges.shape
    segments = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=int(W * 0.1),
        minLineLength=int(W * 0.5),
        maxLineGap=int(W * 0.05),
    )

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

    output = frame.copy()
    key_output = frame.copy()

    if segments is not None and len(segments) > 0:
        top_segments = sorted(segments, key=segment_length, reverse=True)[:N_LINES]
        for seg in top_segments:
            x1, y1, x2, y2 = seg[0]
            cv2.line(output, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)

        longest = top_segments[0]
        longest_angle = segment_angle(longest)
        longest_len = segment_length(longest)
        PARALLEL_TOL = 5 * np.pi / 180

        parallel = [s for s in top_segments[1:]
                    if angle_diff(segment_angle(s), longest_angle) <= PARALLEL_TOL]

        if parallel:
            farthest = max(parallel, key=lambda s: point_to_line_dist(
                (s[0][0] + s[0][2]) / 2, (s[0][1] + s[0][3]) / 2, longest))
            #print(farthest)

            # mx = (farthest[0][0] + farthest[0][2]) / 2
            # my = (farthest[0][1] + farthest[0][3]) / 2
            # half = longest_len / 2
            # ca, sa = np.cos(longest_angle), np.sin(longest_angle)
            # new_seg = (int(mx - half * ca), int(my - half * sa),
            #            int(mx + half * ca), int(my + half * sa))

            lx1, ly1, lx2, ly2 = longest[0]
            new_seg = farthest[0]
            cv2.line(key_output, (lx1, ly1), (lx2, ly2), LINE_COLOR, LINE_THICKNESS)
            cv2.line(key_output, new_seg[:2], new_seg[2:], (0, 255, 0), LINE_THICKNESS)


    return threshed, blurred, edges, output, key_output


def stream_hough(stream: CanonStream, window_name: str = "hough_stream"):
    """Display a CanonStream with Hough line annotations until ESC is pressed."""
    stream.start()
    while True:
        grabbed, frame = stream.read()
        if not grabbed or frame is None:
            print("Failed to read from camera")
            break
        threshed, blurred, edges, output, key_output = annotate_frame(frame)
        mosaic = make_mosaic([
            ("threshed", threshed),
            ("blurred", blurred),
            ("edges", edges),
            ("all lines", output),
            ("key lines", key_output),
        ])
        cv2.imshow(window_name, mosaic)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    stream.stop()
    cv2.destroyAllWindows()


# def find_hough(img):
#     original = load_image("test_piano.avif")
#     gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

#     blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
#     # edges = sobel_magnitude(blurred)

#     # 1. Reduce the blur significantly
#     # We want to remove noise, not destroy the edges!
#     # blurred = cv2.GaussianBlur(gray, (25, 25), 0)

#     # 2. Use Canny instead of manual Sobel magnitude
#     # Canny performs non-maximum suppression (makes lines 1-pixel thin)
#     # and binarization (makes pixels strictly 0 or 255)
#     edges = cv2.Canny(blurred, 50, 150)

#     # 3. CRITICAL: Clean the borders 
#     # This removes the "frame" lines that Hough loves to detect
#     # border_size = 5
#     # edges[:border_size, :] = 0 # Top
#     # edges[-border_size:, :] = 0 # Bottom
#     # edges[:, :border_size] = 0 # Left
#     # edges[:, -border_size:] = 0 # Right

#     H, W = edges.shape
#     segments = cv2.HoughLinesP(
#         edges,
#         rho=HOUGH_RHO,
#         theta=HOUGH_THETA,
#         threshold=int(W*0.1),#HOUGH_THRESHOLD,
#         minLineLength=int(W*0.5),#HOUGH_MIN_LENGTH,
#         maxLineGap=int(W*0.05)#HOUGH_MAX_GAP,
#     )

#     if segments is None or len(segments) == 0:
#         print("No lines detected — try loosening HOUGH_THRESHOLD or HOUGH_MIN_LENGTH")
#         raise SystemExit(1)

#     def segment_length(seg):
#         x1, y1, x2, y2 = seg[0]
#         return np.hypot(x2 - x1, y2 - y1)

#     top_segments = sorted(segments, key=segment_length, reverse=True)[:N_LINES]
#     print(f"Detected {len(segments)} segments, overlaying top {len(top_segments)}")

#     output = original.copy()
#     for seg in top_segments:
#         x1, y1, x2, y2 = seg[0]
#         cv2.line(output, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)

#     cv2.imwrite(OUTPUT_PATH, output)
#     print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    # original = load_image("test_piano.avif")
    # find_hough(original)
    streams = open_canon_streams(silent=False)
    for stream in streams:
        stream_hough(stream)
