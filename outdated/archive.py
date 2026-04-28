"""
Old code that may become useful at some point in the future...
"""


"""
def load_image(path: str) -> np.ndarray:
    \"""Load any image (including AVIF) as a BGR numpy array.\"""
    pil_img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

SOBEL_KSIZE       = 3        # Sobel kernel size
def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    
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
    
### Old warp_to_piano \w Hough lines code:

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

    
"""