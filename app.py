from collections import deque
import numpy as np
import cv2
import mediapipe as mp
import joblib
import pygame
last_prediction = ""

# Load trained model
model = joblib.load("asl_model.pkl")

# Store past 20 frames of finger tips
index_history = deque(maxlen=10)  # For Z
pinky_history = deque(maxlen=10)  # For J

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Motion detection functions
def is_j_motion(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return ys[-1] > ys[0] and xs[-1] > xs[0] and abs(xs[-1] - xs[0]) > 0.05

def is_z_motion(points):
    xs = np.array([p[0] for p in points])
    diffs = np.diff(xs)
    if len(diffs) < 3:
        return False
    return diffs[0] > 0 and diffs[1] < 0 and diffs[2] > 0

#Audio Output
pygame.mixer.init()

def play_audio(letter):
    try:
        pygame.mixer.music.load(f"audio/{letter}.wav")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Audio error: {e}")

# Cooldowns and trigger flags
cooldown = 0
z_static_ready = False
z_motion_timeout = 0  # countdown after Z is triggered by static gesture

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten all landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Get pinky (20) and index (8) tips
            index_tip = hand_landmarks.landmark[8]
            pinky_tip = hand_landmarks.landmark[20]

            index_history.append((index_tip.x, index_tip.y))
            pinky_history.append((pinky_tip.x, pinky_tip.y))

            # Predict static gesture
            prediction = model.predict([landmarks])[0]
            final_output = prediction

            # Handle "J" motion with pinky
            if cooldown == 0 and len(pinky_history) == 20 and is_j_motion(pinky_history):
                final_output = "J"
                pinky_history.clear()
                cooldown = 30

            # Trigger Z motion if static Z is detected
            if prediction == "Z":
                z_static_ready = True
                z_motion_timeout = 40  # 2 seconds window

            # Allow motion Z detection only after Z pose
            if z_static_ready and z_motion_timeout > 0:
                if len(index_history) == 20 and is_z_motion(index_history):
                    final_output = "Z"
                    index_history.clear()
                    cooldown = 30
                    z_static_ready = False
                    z_motion_timeout = 0

            # Play audio if new letter is predicted
            if final_output != last_prediction and final_output in ["C", "X", "J", "Q", "R", "N", "H", "M", "E", "P"]:
                play_audio(final_output)
                last_prediction = final_output

            # Display prediction
            cv2.putText(frame, f"Symbol: {final_output}", (10, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 2)
            
            # Draw pinky motion path (magenta)
            for i in range(1, len(pinky_history)):
                pt1 = pinky_history[i - 1]
                pt2 = pinky_history[i]
                cv2.line(frame,
                         (int(pt1[0] * frame.shape[1]), int(pt1[1] * frame.shape[0])),
                         (int(pt2[0] * frame.shape[1]), int(pt2[1] * frame.shape[0])),
                         (255, 0, 255), 2)

    # Decrease cooldowns
    if cooldown > 0:
        cooldown -= 1
    if z_motion_timeout > 0:
        z_motion_timeout -= 1

    # Show window
    cv2.imshow("Group 2 Sign Language Recognition Project - ESC to exit", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
