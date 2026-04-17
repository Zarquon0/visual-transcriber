import cv2

cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(2)

for cap in (cap0, cap1):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("cam0 (index 1) opened:", cap0.isOpened())
print("cam1 (index 2) opened:", cap1.isOpened())

while True:
    ok0, frame0 = cap0.read()
    ok1, frame1 = cap1.read()

    if not ok0 or not ok1:
        print("Failed to read one of the cameras")
        break

    cv2.imshow("cam0", frame0)
    cv2.imshow("cam1", frame1)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()