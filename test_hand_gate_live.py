"""Quick live test for HandGate — shows hand landmarks + fingertips in real time.

Usage:
    uv run python test_hand_gate_live.py          # webcam index 0
    uv run python test_hand_gate_live.py --cam 1  # specific index
    uv run python test_hand_gate_live.py --canon  # auto-detect Canon camera

Controls: ESC or q to quit.
"""

import argparse
import cv2
from core.hand_gate import HandGate
from core.stream_webcams import open_canon_streams


def run(cam_index: int | None, use_canon: bool) -> None:
    gate = HandGate()

    if use_canon:
        streams = open_canon_streams(silent=False)
        stream = streams[0]
        stream.start()
        read = stream.read
        stop = stream.stop
    else:
        idx = cam_index if cam_index is not None else 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {idx}")
        read = lambda: cap.read()
        stop = cap.release

    print("Running — ESC or q to quit")
    try:
        while True:
            ok, frame = read()
            if not ok or frame is None:
                continue

            tips = gate.fingertips(frame)
            out  = gate.draw(frame, tips)

            # Overlay tip count
            label = f"{len(tips)} fingertip(s)"
            cv2.putText(out, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(out, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

            cv2.imshow("HandGate live", out)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        stop()
        gate.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=None, help="OpenCV camera index")
    p.add_argument("--canon", action="store_true", help="Auto-detect Canon camera")
    args = p.parse_args()
    run(args.cam, args.canon)
