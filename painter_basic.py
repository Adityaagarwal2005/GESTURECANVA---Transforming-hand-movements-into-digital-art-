# Basic Virtual Painter
# Simple drawing with index finger only
# No palette, eraser, or saving functionality
import cv2

cap = cv2.VideoCapture(0)

for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ret, frame = cap.read()
    if ret:
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Requested: {w}x{h}, Got: {int(actual_w)}x{int(actual_h)}")

cap.release()
