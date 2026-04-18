import subprocess
from pathlib import Path
import time
from collections import deque
import cv2
from cv2.typing import MatLike
import yaml
import threading


_CONFIG_PATH = Path(__file__).parent / "camera_config.yaml"


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_canon_indices() -> list[int]:
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
    return [i for i, name in enumerate(opencv_order) if "Canon" in name]

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

def open_canon_streams(config_path: Path = _CONFIG_PATH, silent = True) -> list[CanonStream]:
    """Detect all Canon cameras and return a list of opened VideoCapture objects.

    Resolution and frame rate are applied from the config yaml. Raises
    RuntimeError if no Canon cameras are found.
    """

    indices = find_canon_indices()
    if not indices:
        raise RuntimeError("No Canon cameras detected")

    if not silent:
        print(f"Detected {len(indices)} Canon camera(s) at OpenCV indices: {indices}")

    cfg = _load_config(config_path)
    streams = []
    for idx in indices:
        stream = CanonStream(idx, cfg, show_stats=True)
        streams.append(stream)
        if not silent:
            print(f"  cam{idx} opened: {stream.cap.isOpened()}")

    return streams


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
