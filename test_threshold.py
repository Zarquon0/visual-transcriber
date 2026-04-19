import cv2
import numpy as np
from stream_webcams import open_canon_streams


def normalize_and_threshold(frame: np.ndarray) -> np.ndarray:
    img = frame.astype(np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = ch.min(), ch.max()
        img[:, :, c] = (ch - lo) / (hi - lo) if hi > lo else np.zeros_like(ch)
    thresh = 0.7
    mask = (img[:, :, 0] >= thresh) & (img[:, :, 1] >= thresh) & (img[:, :, 2] >= thresh)
    img = np.where(mask, 1.0, 0.0).astype(np.float32)
    return img


if __name__ == "__main__":
    streams = open_canon_streams(silent=False)
    for stream in streams:
        stream.start()

    while True:
        frames = [stream.read() for stream in streams]
        if not all(ok for ok, _ in frames):
            print("Failed to read from one or more cameras")
            break

        for i, (_, frame) in enumerate(frames):
            cv2.imshow(f"cam{i}", normalize_and_threshold(frame))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()
