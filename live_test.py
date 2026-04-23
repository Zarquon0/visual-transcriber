"""Quick live test: capture a frame from the connected Canon (or iPhone),
run the auto-calibration + labeling pipeline on it, show the result.

Usage:
    uv run python live_test.py
Preview window opens — position the camera, then:
    s   → save current frame and run detection
    SPACE → same as s
    ESC → quit without capture
The captured frame is saved to piano_photos/live_TIMESTAMP.jpg, and the
pipeline result is saved to piano_photos/live_TIMESTAMP_result.png and
auto-opened.
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

import cv2

from stream_webcams import open_canon_streams, CanonStream, _load_config


def _find_specific_camera_index(prefer: tuple[str, ...]) -> int | None:
    """Ask Swift/AVFoundation for device names in OpenCV's ordering, then
    pick the first device whose name contains any substring in `prefer`.
    Returns the OpenCV VideoCapture index, or None if not found.
    """
    import subprocess
    swift_code = (
        'import AVFoundation; '
        'for d in AVCaptureDevice.devices(for: .video) { '
        'let flag = d.deviceType == .builtInWideAngleCamera ? "1" : "0"; '
        'print("\\(d.localizedName)|\\(flag)") }'
    )
    r = subprocess.run(
        ["swift", "-e", swift_code], capture_output=True, text=True, timeout=15
    )
    external, builtin = [], []
    for line in r.stdout.splitlines():
        if "|" not in line:
            continue
        name, flag = line.rsplit("|", 1)
        (builtin if flag.strip() == "1" else external).append(name.strip())
    opencv_order = external + builtin
    for i, name in enumerate(opencv_order):
        if any(p.lower() in name.lower() for p in prefer):
            print(f"using device index {i}: {name}")
            return i
    return None


def main() -> None:
    # Prefer the Canon (EOS Webcam Utility) specifically. Fall back to
    # direct Canon Digital Camera if EOS-utility isn't active, then iPhone.
    import cv2
    idx = _find_specific_camera_index(("EOS", "Canon"))
    if idx is None:
        print("no Canon camera found — falling back to first available stream.")
        streams = open_canon_streams(silent=False)
        if not streams:
            print("no stream at all.")
            sys.exit(1)
        stream = streams[0]
    else:
        # Open directly at the chosen index and wrap with CanonStream for
        # threaded reads / consistent API.
        cfg = _load_config()
        stream = CanonStream(idx, cfg, show_stats=False)
        if not stream.cap.isOpened():
            print(f"failed to open device index {idx}.")
            sys.exit(1)
    stream.start()
    print("preview running. s/SPACE = capture, ESC = quit")

    out_dir = Path("piano_photos")
    out_dir.mkdir(exist_ok=True)
    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                continue
            # Rotate 180° so an inverted-mounted camera still produces an
            # upright frame for both preview and downstream detection.
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            # Resize preview for screen visibility.
            preview_h = 720
            preview_w = int(frame.shape[1] * (preview_h / frame.shape[0]))
            preview = cv2.resize(frame, (preview_w, preview_h))
            cv2.putText(
                preview, "s/SPACE: capture   ESC: quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                preview, "s/SPACE: capture   ESC: quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow("live preview", preview)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                return
            if k in (ord("s"), ord(" ")):
                ts = int(time.time())
                photo_path = out_dir / f"live_{ts}.jpg"
                cv2.imwrite(str(photo_path), frame)
                print(f"saved {photo_path}")
                cv2.destroyAllWindows()
                # Run the pipeline on it.
                subprocess.run(
                    ["uv", "run", "python", "auto_calibrate.py", str(photo_path)],
                    check=False,
                )
                out = Path("auto_calib_result.png")
                if out.exists():
                    subprocess.run(["open", str(out)], check=False)
                else:
                    print("no result image produced")
                return
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
