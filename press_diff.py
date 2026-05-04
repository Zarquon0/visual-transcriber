import cv2
from seg_to_keys import load_image
import numpy as np
from stream_webcams import CanonStream, open_canon_streams
from seg_to_keys import warp_to_piano, warp_key_lines, WHITE_PEAK_TOLERANCE
from calibration import build_calibration_data

DIFF_PEAK_MIN_L  = 0     # peaks at or below this L value are ignored as too-dark-to-matter
BLOB_TOP_FRAC    = 0.15  # fraction of image height that counts as "top"
BLOB_TOP_THRESH  = 3     # minimum pixels a blob must have in the top region to be kept

# press    = load_image("piano_photos/press.png")
# no_press = load_image("piano_photos/no_press.png")

# ph, pw = press.shape[:2]
# nh, nw = no_press.shape[:2]

# if (pw, ph) != (nw, nh):
#     if pw * ph > nw * nh:
#         press = cv2.resize(press, (nw, nh), interpolation=cv2.INTER_AREA)
#     else:
#         no_press = cv2.resize(no_press, (pw, ph), interpolation=cv2.INTER_AREA)

# press_gray    = cv2.cvtColor(press,    cv2.COLOR_BGR2GRAY)
# no_press_gray = cv2.cvtColor(no_press, cv2.COLOR_BGR2GRAY)

# diff = cv2.absdiff(press_gray, no_press_gray)

# cv2.imshow("press - no_press diff", diff)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def isolate_white_all_peaks(frame: np.ndarray) -> None:
#     """Display a masked image for every L-channel histogram peak, plus an annotated histogram.

#     Generates the same histogram as isolate_white(), then produces one mask per peak
#     (rather than just the rightmost one) and displays them all in a mosaic. Also calls
#     _draw_hist_debug_all_peaks() to show the histogram with each peak marked by index.
#     """
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l_channel = lab[:, :, 0]
#     n_pixels = frame.shape[0] * frame.shape[1]
#     hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256]).flatten()
#     smoothed = np.convolve(hist, np.ones(35) / 35, mode='same')
#     min_height = n_pixels * 0.0015
#     peak_idxs = [
#         i for i in range(1, 255)
#         if smoothed[i] >= min_height
#         and all(
#             smoothed[i] > smoothed[j]
#             for j in range(max(0, i - PEAK_NEIGHBORHOOD), min(256, i + PEAK_NEIGHBORHOOD + 1))
#             if j != i
#         )
#     ]
#     if not peak_idxs:
#         peak_idxs = [int(np.argmax(smoothed))]
#     l = l_channel.astype(np.int16)
#     named_frames = []
#     for idx, peak in enumerate(peak_idxs):
#         mask = (l >= peak - WHITE_PEAK_TOLERANCE) & (l <= peak + WHITE_PEAK_TOLERANCE)
#         result = np.where(np.stack([mask] * 3, axis=2), 255, 0).astype(np.uint8)
#         named_frames.append((f'peak {idx}  L={peak}', result))
#     cv2.imshow("L-channel histogram (all peaks)", _draw_hist_debug_all_peaks(smoothed, peak_idxs))
#     cv2.imshow("All peak masks", make_mosaic(named_frames))
#     cv2.waitKey(0)

# def white_balance(frame: np.ndarray) -> np.ndarray:
#     """Correct the white balance of frame using isolated white pixels as the reference.

#     Applies isolate_white() to find white-key pixels, computes their mean BGR in the
#     original image, then scales each channel so that mean maps to 255.
#     """
#     white_mask_bgr = isolate_white(frame)
#     mask = white_mask_bgr[:, :, 0] > 0
#     if not mask.any():
#         return frame.copy()
#     mean_bgr = frame[mask].astype(np.float32).mean(axis=0)
#     scale = np.where(mean_bgr > 0, 255.0 / mean_bgr, 1.0)
#     return np.clip(frame.astype(np.float32) * scale[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)

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

    Treats grayscale values directly as L. Finds the rightmost histogram peak
    above DIFF_PEAK_MIN_L and masks pixels within WHITE_PEAK_TOLERANCE of it.
    Returns a black image if no qualifying peak exists.
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
    mask = (g >= 75)
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