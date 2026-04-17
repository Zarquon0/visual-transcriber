import subprocess
import sys
from pathlib import Path
import cv2
import yaml


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


def open_canon_captures(config_path: Path = _CONFIG_PATH, silent = True) -> list[cv2.VideoCapture]:
    """Detect all Canon cameras and return a list of opened VideoCapture objects.

    Resolution and frame rate are applied from the config yaml. Raises
    RuntimeError if no Canon cameras are found.
    """
    cfg = _load_config(config_path)
    width = cfg.get("resolution", {}).get("width", 1280)
    height = cfg.get("resolution", {}).get("height", 720)
    fps = cfg.get("fps")

    indices = find_canon_indices()
    if not indices:
        raise RuntimeError("No Canon cameras detected")

    if not silent:
        print(f"Detected {len(indices)} Canon camera(s) at OpenCV indices: {indices}")

    captures = []
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, fps)
        captures.append(cap)
        if not silent:
            print(f"  cam{idx} opened: {cap.isOpened()}")

    return captures


if __name__ == "__main__":
    # DEMO: streams all connected webcams until escaped
    caps = open_canon_captures(silent=False)

    while True:
        frames = [cap.read() for cap in caps]
        if not all(ok for ok, _ in frames):
            print("Failed to read from one or more cameras")
            break

        for i, (_, frame) in enumerate(frames):
            cv2.imshow(f"cam{i}", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
