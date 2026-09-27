import cv2
import mediapipe as mp
import joblib
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/isl_model.pkl")

print("Model loaded successfully!")

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

print("KAIKURAL Live Translation Started")
print("Press Q to exit")

while True:

    success, frame = cap.read()

    if not success:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction = "No Hand"
    confidence = 0

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Extract landmarks
        landmarks = []

        for landmark in hand.landmark:
            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)

        # Convert to numpy
        data = np.array(landmarks).reshape(1, -1)

        # Prediction
        prediction = model.predict(data)[0]

        # Confidence
        probabilities = model.predict_proba(data)[0]
        confidence = np.max(probabilities) * 100

        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    # -----------------------------
    # DISPLAY RESULT
    # -----------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (640, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Sign: {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "KAIKURAL - ISL Translation",
        frame
    )

    # Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()

print("KAIKURAL stopped.")