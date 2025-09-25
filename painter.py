import cv2
import mediapipe as mp
import numpy as np

# ----------------- MediaPipe Hands Setup -----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ----------------- Webcam Setup -----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow("ADITYA'S WEBCAM", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ADITYA'S WEBCAM", 640, 480)

# ----------------- Canvas Setup -----------------
canvas = None  # Will create after getting first frame
draw_color = (0,255,0,128)  # green color
brush_thickness = 2
prev_points = [(0, 0), (0, 0)]  # Previous fingertip coordinates for 2 hands

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from camera.")
        break

    frame = cv2.flip(frame, 1)  # Mirror

    if canvas is None:
        canvas = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)  # Same size as frame
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, c = frame.shape
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            prev_x, prev_y = prev_points[i]
            if prev_x == 0 and prev_y == 0:
                prev_points[i] = (x, y)
            else:
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
                prev_points[i] = (x, y)
    else:
        prev_points = [(0, 0), (0, 0)]  # Reset when no hands detected

    # Merge canvas with frame
    alpha_canvas = canvas[:, :, 3] / 255.0  # Alpha channel (0-1)
    for c in range(3):  # For B, G, R channels
        frame_bgra[:, :, c] = (1 - alpha_canvas) * frame_bgra[:, :, c] + alpha_canvas * canvas[:, :, c]

    cv2.imshow("ADITYA'S WEBCAM", cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Ask user if they want to save
        choice = input("Do you want to save your drawing? (y/n): ").lower()
        if choice == 'y':
            image_name = input("Enter a name for your drawing (without extension): ")
            image_name = image_name + ".png"
            cv2.imwrite(image_name, canvas)
            print(f"✅ Saved your drawing as {image_name}")
        else:
            print(" Drawing discarded.")
        break
    elif key == ord('f'):  # Toggle fullscreen
        cv2.setWindowProperty("ADITYA'S WEBCAM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()
