import cv2
import numpy as np
from stream_webcams import CanonStream, open_canon_streams
from seg_to_keys import warp_to_piano, warp_key_lines, WHITE_PEAK_TOLERANCE

#DIFF_PEAK_MIN_L  = 0     # peaks at or below this L value are ignored as too-dark-to-matter
DIFF_TRESH = 75
BLOB_TOP_FRAC    = 0.15  # fraction of image height that counts as "top"
BLOB_TOP_THRESH  = 3     # minimum pixels a blob must have in the top region to be kept

def filter_blobs_by_top_presence(mask: np.ndarray) -> np.ndarray:
    """Keep only blobs that have at least BLOB_TOP_THRESH pixels in the top BLOB_TOP_FRAC of the image."""
    top_cutoff = int(mask.shape[0] * BLOB_TOP_FRAC)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    for contour in contours:
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)
        if np.count_nonzero(blob[:top_cutoff]) >= BLOB_TOP_THRESH:
            result = cv2.bitwise_or(result, blob)
    return result

def isolate_diff_peak(gray: np.ndarray) -> np.ndarray:
    """Like isolate_white() but for a grayscale diff image.

    Currently, applies simple, constant value thresholding to input grayscale image.
    """
    # n_pixels = gray.shape[0] * gray.shape[1]
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    # smoothed = np.convolve(hist, np.ones(11) / 11, mode='same')
    # min_height = n_pixels * 0.0015
    # peak_idxs = [
    #     i for i in range(1, 255)
    #     if smoothed[i] > smoothed[i - 1]
    #     and smoothed[i] > smoothed[i + 1]
    #     and smoothed[i] >= min_height
    #     and i > DIFF_PEAK_MIN_L
    # ]
    # if not peak_idxs:
    #     return np.zeros_like(gray)
    # peak = peak_idxs[-1]
    g = gray.astype(np.int16)
    #mask = (g >= peak - WHITE_PEAK_TOLERANCE) & (g <= peak + WHITE_PEAK_TOLERANCE)
    mask = (g >= DIFF_TRESH)
    return np.where(mask, 255, 0).astype(np.uint8)

def detect_press_regions(live_warped: np.ndarray, stored_warped: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(
        cv2.cvtColor(stored_warped, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(live_warped,        cv2.COLOR_BGR2GRAY),
    )
    isolated = isolate_diff_peak(diff)
    return filter_blobs_by_top_presence(isolated)



def stream_diff(stream: CanonStream, window_name: str = "diff_stream"):
    stream.start()
    stored_corners = None
    stored_warped  = None
    while True:
        grabbed, frame = stream.read()
        if not grabbed or frame is None:
            print("Failed to read from camera")
            break

        if stored_corners is None:
            warped, _, corners = warp_to_piano(frame, debug=True)
            cv2.imshow("warped", warped)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32 and corners is not None:
                stored_corners = corners
                stored_warped  = warped
        else:
            first_rail = np.concatenate([corners[0], corners[1]])
            second_rail = np.concatenate([corners[2], corners[3]])
            _, warped = warp_key_lines(frame, first_rail, second_rail)
            cv2.imshow(window_name, detect_press_regions(warped, stored_warped))
            cv2.imshow("warped", warped)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    streams = open_canon_streams(allow_iphone=False, silent=False)
    for stream in streams:
        stream_diff(stream)