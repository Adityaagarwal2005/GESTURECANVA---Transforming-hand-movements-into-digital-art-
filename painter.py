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
draw_color = (0,255,0,128)  # Green color
brush_thickness = 2
prev_points = {"Left": (0, 0), "Right": (0, 0)}  # Previous fingertip coordinates for 2 hands
drawing_enabled = False  # 🆕 NEW: Track whether drawing is active or paused

colors = [(0,255,0,128), (0,0,255,128), (255,0,0,128), (0,255,255,128)]  # Green, Red, Blue, Yellow
color_index = 0  # current color


# 🆕 NEW FUNCTION: Count how many fingers are up
def count_fingers(hand_landmarks, h, w):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    fingers = []

    # Thumb check (horizontal direction)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (vertical direction)
    for tip_id in tips[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

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
        if results.multi_hand_landmarks:
            any_draw = False   # 🆕 NEW: track if any hand shows draw gesture
            any_pause = False  # 🆕 NEW: track if any hand shows pause gesture

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, c = frame.shape
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

                    # 🆕 NEW: Detect gesture based on number of fingers
            fingers_up = count_fingers(hand_landmarks, h, w)

                # 🆕 NEW: Just record what each hand is doing
            if fingers_up == 1:
                any_draw = True
            if fingers_up == 5:
                any_pause = True
    
            # 🆕 Change color with 2 fingers
            if fingers_up == 2:
                color_index = (color_index + 1) % len(colors)
                draw_color = colors[color_index]
            
            # 🆕 Increase brush thickness with 3 fingers
            if fingers_up == 3:
                brush_thickness += 1
                if brush_thickness > 10:  # max limit
                    brush_thickness = 10
            
            # 🆕 Decrease brush thickness with 4 fingers
            if fingers_up == 4:
                brush_thickness -= 1
                if brush_thickness < 1:  # min limit
                    brush_thickness = 1
            
            # 🆕 NEW: Only draw if drawing_enabled is True
            if drawing_enabled:
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                prev_x, prev_y = prev_points[hand_label]
    
                if prev_x == 0 and prev_y == 0:
                    prev_points[hand_label] = (x, y)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
                    prev_points[hand_label] = (x, y)
            else:
                # 🆕 NEW: Reset previous points to prevent unwanted connecting lines
                prev_points["Left"] = (0, 0)
                prev_points["Right"] = (0, 0)
        if any_pause:
            drawing_enabled = False
            cv2.putText(frame, "✋ Paused", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif any_draw:
            drawing_enabled = True
            cv2.putText(frame, "✏️ Drawing ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    


    else:
        # Reset when no hands detected
        prev_points["Left"] = (0, 0)
        prev_points["Right"] = (0, 0)

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
            image_name = input("Enter a name for your drawing (without extension): ") + ".png"
            cv2.imwrite(image_name, canvas)
            print(f"✅ Saved your drawing as {image_name}")
        else:
            print(" Drawing discarded.")
        break
    elif key == ord('f'):  # Toggle fullscreen
        cv2.setWindowProperty("ADITYA'S WEBCAM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()
