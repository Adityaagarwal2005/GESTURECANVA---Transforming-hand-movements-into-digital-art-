GESTURECANVAS-->Real-Time Hand Detection & Drawing with Fingertip

FingerDraw is a real-time computer vision project that uses Google’s pre-trained MediaPipe Hands model and OpenCV to let you draw on the screen using just your fingertip 🖌️ — no mouse, no touchscreen required!

The project tracks your hand from a webcam feed, identifies the index fingertip among 21 hand landmarks, and draws wherever you point — like a virtual pen in the air.

--> Features

1-Real-time hand detection & tracking using a pre-trained ML model

2-Draw on the screen with your index fingertip

3-Access 21 key landmarks on the hand for gesture-based control

4 Works live on webcam feed
 
5Easy to customize for gesture recognition, air-writing, or virtual whiteboards
 

-->Tech Stack

Python 3.x

OpenCV – for video capture and drawing

MediaPipe – for hand detection and landmark estimation

NumPy – for numerical operations (optional, used for arrays/drawing logic)

-->project features

Hand Gesture Drawing: Draw with 1 finger (index); shows Drawing ON / Paused dynamically.

Pause / Resume: Open palm (5 fingers) pauses drawing; resumes when index finger is shown.

Custom Canvas: Transparent overlay on webcam; lines follow fingertip smoothly.

Finger-Based Controls:

1 finger → Draw

2 fingers → Change color

(Can extend for brush size, clear canvas, etc.)

Persistent Drawing: Lines are stored on a separate canvas, preventing unwanted connections.

Save Drawing: Press q to save as .png.

Fullscreen Toggle: Press f for fullscreen webcam view.