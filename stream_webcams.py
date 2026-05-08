import subprocess
from pathlib import Path
import time
from collections import deque
import cv2
from cv2.typing import MatLike
import yaml
import threading


_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_canon_indices(allow_iphone=False) -> list[int]:
    """Return OpenCV VideoCapture indices for all connected Canon cameras on macOS.

    Uses Swift to enumerate AVFoundation video devices by name, then maps them
    to OpenCV indices. OpenCV's AVFoundation backend orders external (USB /
    non-built-in) cameras before built-in cameras, preserving each group's
    relative AVFoundation order within itself.
    """
    swift_code = r"""
import AVFoundation
let devices = AVCaptureDevice.devices(for: .video)
for d in devices {
    let flag = d.deviceType == .builtInWideAngleCamera ? "1" : "0"
    print("\(d.localizedName)|\(flag)")
}
"""
    try:
        result = subprocess.run(
            ["swift", "-e", swift_code],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        raise RuntimeError("'swift' not found — install Xcode Command Line Tools")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Swift device enumeration timed out")

    external: list[str] = []
    builtin: list[str] = []
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        name, flag = line.rsplit("|", 1)
        (builtin if flag.strip() == "1" else external).append(name.strip())

    opencv_order = external + builtin
    # Prefer "EOS Webcam Utility" (the working video pipe) over "Canon
    # Digital Camera" (raw still-mode, often crashes on capture). Both
    # show up when the camera is plugged in via EOS Webcam Utility app.
    if allow_iphone:
        is_target = lambda name: "EOS" in name or "Canon" in name or "iPhone" in name
    else:
        is_target = lambda name: "EOS" in name or "Canon" in name
    matches = [(i, name) for i, name in enumerate(opencv_order) if is_target(name)]
    # Within matches, sort so EOS comes first, then Canon, then iPhone.
    def rank(name):
        if "EOS" in name: return 0
        if "Canon" in name: return 1
        return 2
    matches.sort(key=lambda im: rank(im[1]))
    # Drop "Canon Digital Camera" if "EOS Webcam Utility" is present
    # (same physical device, only EOS variant works for video).
    has_eos = any("EOS" in name for _, name in matches)
    if has_eos:
        matches = [(i, n) for i, n in matches if "EOS" in n or "iPhone" in n]
    return [i for i, _ in matches]

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
class CanonStream():
    """
    Wrapper around a cv2 VideoCapture stream that decouples reading an image from the camera to python
    from reading an image from memory for further processing, pipelining the process and reducing latency
    """
    def __init__(self, src: int, cfg: dict = None, show_stats: bool = False):
        # Make capture object
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)

        # Read configuration and set resolution/frame rate
        if cfg:
            width = cfg.get("resolution", {}).get("width", DEFAULT_WIDTH)
            height = cfg.get("resolution", {}).get("height", DEFAULT_HEIGHT)
            fps = cfg.get("fps")
        else:
            width = DEFAULT_WIDTH
            height = DEFAULT_HEIGHT
            fps = None

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Get rid of buffer to eliminate buffer latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize other state
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.height = height
        self.width = width

        # Stats tracking (enabled only when show_stats=True)
        self._show_stats = show_stats
        if show_stats:
            self._frame_times: deque[float] = deque()
            self._measured_fps: float = 0.0
            self._measured_res: tuple[int, int] = (0, 0)

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not (frame is None): #On continuity cam, frame is sometimes None - in that case, just don't update
                if frame.shape[0] != self.height or frame.shape[1] != self.width:
                    # Manual resize if not receiving requested image resolution
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                if self._show_stats and grabbed and frame is not None:
                    # Update image stream stats, if desired
                    now = time.perf_counter()
                    self._frame_times.append(now)
                    cutoff = now - 1.0
                    while self._frame_times and self._frame_times[0] < cutoff:
                        self._frame_times.popleft()
                    self._measured_fps = len(self._frame_times)
                    h, w = frame.shape[:2]
                    self._measured_res = (w, h)
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame

    def read(self) -> tuple[bool, MatLike]:
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        if self._show_stats and frame is not None:
            # Display image stream stats, if desired
            w, h = self._measured_res
            fps = self._measured_fps
            for i, text in enumerate([f"FPS: {fps:.1f}", f"Res: {w}x{h}"]):
                y = 30 + i * 30
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return grabbed, frame

    def stop(self):
        self.started = False
        self.cap.release()
        self.thread.join(timeout=2.0)

_DUAL_BUFFER_SIZE = 5

class DualCanonStream:
    """
    Synchronized stream from two Canon R50 cameras. Each camera runs its own
    reader thread that fills a rolling buffer of (timestamp, frame) pairs.
    read() returns the best-matched pair across both buffers.
    """
    def __init__(self, src0: int, src1: int, cfg: dict = None, show_stats: bool = False):
        self._show_stats = show_stats

        if cfg:
            width = cfg.get("resolution", {}).get("width", DEFAULT_WIDTH)
            height = cfg.get("resolution", {}).get("height", DEFAULT_HEIGHT)
            fps = cfg.get("fps")
        else:
            width = DEFAULT_WIDTH
            height = DEFAULT_HEIGHT
            fps = None

        self._caps: list[cv2.VideoCapture] = []
        for src in (src0, src1):
            cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None:
                cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._caps.append(cap)

        self.width = width
        self.height = height

        # Each buffer stores (timestamp, frame) tuples
        self._buffers: list[deque[tuple[float, MatLike]]] = [
            deque(maxlen=_DUAL_BUFFER_SIZE),
            deque(maxlen=_DUAL_BUFFER_SIZE),
        ]
        self._locks = [threading.Lock(), threading.Lock()]
        self.started = False
        self._threads: list[threading.Thread] = []

    def start(self):
        if self.started:
            return
        self.started = True
        for i in range(2):
            t = threading.Thread(target=self._update, args=(i,), daemon=True)
            self._threads.append(t)
            t.start()

    def _update(self, cam_idx: int):
        cap = self._caps[cam_idx]
        buf = self._buffers[cam_idx]
        lock = self._locks[cam_idx]
        while self.started:
            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                continue
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            ts = time.perf_counter()
            with lock:
                buf.append((ts, frame))

    def read(self) -> tuple[MatLike | None, MatLike | None]:
        with self._locks[0]:
            buf0 = list(self._buffers[0])
        with self._locks[1]:
            buf1 = list(self._buffers[1])

        if not buf0 or not buf1:
            return None, None

        ts0, frame0 = buf0[-1]
        ts1, frame1 = buf1[-1]

        # The older most-recent frame is the reference; find nearest match in the other buffer
        if ts0 <= ts1:
            ref_ts, ref_frame = ts0, frame0
            match_ts, match_frame = min(buf1, key=lambda item: abs(item[0] - ts0))
            f0, f1 = ref_frame.copy(), match_frame.copy()
            t0, t1 = ref_ts, match_ts
        else:
            ref_ts, ref_frame = ts1, frame1
            match_ts, match_frame = min(buf0, key=lambda item: abs(item[0] - ts1))
            f0, f1 = match_frame.copy(), ref_frame.copy()
            t0, t1 = match_ts, ref_ts

        if self._show_stats:
            dt_ms = abs(t1 - t0) * 1000
            for frame, ts in ((f0, t0), (f1, t1)):
                for i, text in enumerate([f"t={ts:.3f}s", f"dt={dt_ms:.1f}ms"]):
                    y = 30 + i * 30
                    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return f0, f1

    def stop(self):
        self.started = False
        for t in self._threads:
            t.join(timeout=2.0)
        for cap in self._caps:
            cap.release()


def open_canon_streams(config_path: Path = _CONFIG_PATH, allow_iphone=False, silent = True) -> list[CanonStream]:
    """Detect all Canon cameras and return a list of opened VideoCapture objects.

    Resolution and frame rate are applied from the config yaml. Raises
    RuntimeError if no Canon cameras are found.
    """

    indices = find_canon_indices(allow_iphone=allow_iphone)
    if not indices:
        raise RuntimeError("No Canon cameras detected")

    if not silent:
        print(f"Detected {len(indices)} Canon camera(s) at OpenCV indices: {indices}")

    cfg = _load_config(config_path)
    streams = []
    for idx in indices:
        stream = CanonStream(idx, cfg, show_stats=False)
        streams.append(stream)
        if not silent:
            print(f"  cam{idx} opened: {stream.cap.isOpened()}")

    return streams


def open_dual_canon_stream(config_path: Path = _CONFIG_PATH, show_stats: bool = False) -> DualCanonStream:
    """Detect exactly two Canon cameras and return an opened DualCanonStream."""
    indices = find_canon_indices()
    if len(indices) < 2:
        raise RuntimeError(f"Expected 2 Canon cameras, found {len(indices)}")
    if len(indices) > 2:
        print(f"Warning: found {len(indices)} Canon cameras, using indices {indices[0]} and {indices[1]}")
    cfg = _load_config(config_path)
    return DualCanonStream(indices[0], indices[1], cfg, show_stats=show_stats)

if __name__ == "__main__":
    # DEMO: streams all connected webcams until escaped
    streams = open_canon_streams(silent=False)
    
    for stream in streams: stream.start()
    while True:
        frames = [stream.read() for stream in streams]
        if not all(ok for ok, _ in frames):
            print("Failed to read from one or more cameras")
            break

        for i, (_, frame) in enumerate(frames):
            cv2.imshow(f"cam{i}", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            # NOTE: this doesn't seem to be working at the moment - just ^C twice to quit
            break

    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     stream = open_dual_canon_stream(show_stats=True)
#     stream.start()

#     frozen = False

#     while True:
#         if not frozen:
#             f0, f1 = stream.read()
#             if f0 is not None and f1 is not None:
#                 cv2.imshow("Canon 0", f0)
#                 cv2.imshow("Canon 1", f1)

#         key = cv2.waitKey(1) & 0xFF
#         if key == 32:  # space — freeze on last displayed frames
#             frozen = True
#         elif key == 27:  # ESC — quit
#             break

#     stream.stop()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     stream = open_dual_canon_stream(show_stats=True)
#     stream.start()

#     while True:
#         f0, f1 = stream.read()
#         if f0 is None or f1 is None:
#             continue

#         cv2.imshow("Canon 0", f0)
#         cv2.imshow("Canon 1", f1)

#         if cv2.waitKey(1) & 0xFF == 27:  # ESC
#             break

#     stream.stop()
#     cv2.destroyAllWindows()

