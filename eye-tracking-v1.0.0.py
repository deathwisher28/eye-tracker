"""
MVP Prompt for AI:
Write a Python application using MediaPipe, OpenCV, and PyAutoGUI that:
1. Continuously captures webcam frames.
2. Uses MediaPipe Hands to detect an **open fist** (all five fingers extended) to activate eye tracking, and a **closed fist** (no fingers visible) to deactivate; if the closed-fist gesture is held for 2 seconds, exit the app.
3. When eye tracking is active, uses MediaPipe Face Mesh to locate iris centers and move the cursor. Provide a sensitivity parameter (`--sensitivity`).
4. Detects winks via **eye aspect ratio (EAR)**: left wink → left-click; right wink → right-click; double wink → double-click. Blinks (both eyes simultaneously) are ignored.
5. Quit via 'q' key or prolonged closed-fist gesture.
6. Modular, commented code with easy dependency install instructions.
"""

# =========================================
# eye_cursor_control.py - Updated Implementation
# =========================================

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import argparse
import sys

# ------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Eye-tracking cursor control with gesture commands.")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=13.0,
        help="Cursor movement sensitivity multiplier (default=1.0)"
    )
    return parser.parse_args()

# ------------------------------------------------
# Constants
EYE_AR_THRESH = 0.15       # threshold for wink detection
EYE_AR_CONSEC_FRAMES = 3   # frames below threshold
DOUBLE_WINK_INTERVAL = 1.0 # seconds for double wink\ n
GESTURE_HOLD_TIME = 2.0    # seconds to exit on closed-fist

# ------------------------------------------------
# Init pyautogui and screen size
dpy = pyautogui
dpy.FAILSAFE = False
screen_w, screen_h = dpy.size()

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Landmarks for EAR calculation (MediaPipe FaceMesh)
LEFT_EYE_VERT = [159, 145]
LEFT_EYE_HORIZ = [33, 133]
RIGHT_EYE_VERT = [386, 374]
RIGHT_EYE_HORIZ = [362, 263]

# ------------------------------------------------
def eye_aspect_ratio(landmarks, vert_idxs, horiz_idxs):
    v = np.linalg.norm(np.array(landmarks[vert_idxs[0]]) - np.array(landmarks[vert_idxs[1]]))
    h = np.linalg.norm(np.array(landmarks[horiz_idxs[0]]) - np.array(landmarks[horiz_idxs[1]]))
    return v / h if h > 0 else 0

# ------------------------------------------------
def is_open_fist(hand_landmarks):
    """Return True if all five fingers are extended"""
    tips = [4, 8, 12, 16, 20]
    return all(hand_landmarks.landmark[t].y < hand_landmarks.landmark[t-2].y for t in tips)

# ------------------------------------------------
def is_closed_fist(hand_landmarks):
    """Return True if all five fingertips are folded"""
    tips = [4, 8, 12, 16, 20]
    return all(hand_landmarks.landmark[t].y > hand_landmarks.landmark[t-2].y for t in tips)

# ------------------------------------------------
def main():
    args = parse_args()
    sens = args.sensitivity

    cap = cv2.VideoCapture(0)
    tracking = False
    closed_start = None

    wink_count = {'left': 0, 'right': 0}
    last_double = 0

    print(f"Starting... sensitivity={sens}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Hand gestures: open fist activates, closed fist deactivates/exits ---
        hands_res = hands.process(rgb)
        if hands_res.multi_hand_landmarks:
            for h in hands_res.multi_hand_landmarks:
                if is_open_fist(h):
                    tracking = True
                    closed_start = None
                elif is_closed_fist(h):
                    # deactivate immediately
                    tracking = False
                    if closed_start is None:
                        closed_start = time.time()
                    elif time.time() - closed_start > GESTURE_HOLD_TIME:
                        print("Closed fist held; exiting.")
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)
                else:
                    closed_start = None
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        # --- Eye tracking & click detection ---
        if tracking:
            face_res = face_mesh.process(rgb)
            if face_res.multi_face_landmarks:
                pts = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_res.multi_face_landmarks[0].landmark]

                # Move cursor by iris center (average of two eye centers)
                li = np.array(pts[LEFT_EYE_HORIZ[0]])
                ri = np.array(pts[RIGHT_EYE_HORIZ[1]])
                cx, cy = np.mean([li, ri], axis=0)
                sx = screen_w/2 + (cx - frame.shape[1]/2) * sens
                sy = screen_h/2 + (cy - frame.shape[0]/2) * sens
                dpy.moveTo(sx, sy)

                # Compute EARs
                ear_l = eye_aspect_ratio(pts, LEFT_EYE_VERT, LEFT_EYE_HORIZ)
                ear_r = eye_aspect_ratio(pts, RIGHT_EYE_VERT, RIGHT_EYE_HORIZ)

                # Wink detection
                if ear_l < EYE_AR_THRESH <= ear_r:
                    wink_count['left'] += 1
                else:
                    wink_count['left'] = 0
                if ear_r < EYE_AR_THRESH <= ear_l:
                    wink_count['right'] += 1
                else:
                    wink_count['right'] = 0

                # Single clicks
                if wink_count['left'] >= EYE_AR_CONSEC_FRAMES:
                    dpy.click(button='left')
                    wink_count['left'] = 0
                if wink_count['right'] >= EYE_AR_CONSEC_FRAMES:
                    dpy.click(button='right')
                    wink_count['right'] = 0

                # Double wink for double click
                now = time.time()
                if ear_l < EYE_AR_THRESH and ear_r < EYE_AR_THRESH:
                    if now - last_double < DOUBLE_WINK_INTERVAL:
                        dpy.doubleClick()
                        last_double = 0
                    else:
                        last_double = now

        # --- UI feedback ---
        cv2.putText(frame, f"Tracking: {tracking}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if tracking else (0, 0, 255), 2)
        cv2.imshow("Eye Cursor Control", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
