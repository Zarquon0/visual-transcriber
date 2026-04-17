import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    opened = cap.isOpened()
    ok, frame = (False, None)
    if opened:
        ok, frame = cap.read()
    shape = frame.shape if ok else None
    print(f"index {i}: opened={opened}, read_ok={ok}, shape={shape}")
    cap.release()
