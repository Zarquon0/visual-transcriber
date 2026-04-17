import cv2
import sys

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"index {idx}: opened={cap.isOpened()}")

while True:
    ok, frame = cap.read()
    if not ok:
        print("read failed")
        break
    cv2.putText(frame, f"index {idx}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow(f"cam index {idx}", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
