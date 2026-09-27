import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# SETTINGS
# -----------------------------
LABEL = input("Enter sign name: ").strip().upper()

os.makedirs("dataset/data", exist_ok=True)

file_path = f"dataset/data/{LABEL}.csv"

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

print()
print("================================")
print(" KAIKURAL DATA COLLECTION")
print("================================")
print(f"Sign: {LABEL}")
print("Show your sign to the camera")
print("Press Q to stop")
print("================================")

with open(file_path, "a", newline="") as f:

    writer = csv.writer(f)

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = []

            for landmark in hand.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            writer.writerow(landmarks)

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.putText(
                frame,
                f"COLLECTING: {LABEL}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("KAIKURAL - Dataset Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"\nData saved to: {file_path}")