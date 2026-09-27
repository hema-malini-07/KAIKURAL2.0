from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64


app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

app.config["SECRET_KEY"] = "kaikural-secret"

socketio = SocketIO(
    app,
    cors_allowed_origins="*"
)


# =========================
# LOAD MODEL
# =========================

model = joblib.load("model/isl_model.pkl")

print("KAIKURAL model loaded successfully!")


# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# TRANSLATIONS
# =========================

translations = {
    "HELLO": "வணக்கம்",
    "THANK YOU": "நன்றி"
}


# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/learn")
def learn():
    return render_template("learn.html")


# =========================
# TRANSLATOR
# =========================

@app.route("/translator")
def translator():

    return render_template("translator.html")


# =========================
# CAMERA FRAME
# =========================

@socketio.on("camera_frame")
def camera_frame(data):

    try:

        # Remove base64 header
        image_data = data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy
        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        # Convert to OpenCV image
        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe
        results = hands.process(rgb)

        prediction = "NO HAND"
        confidence = 0

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = []

            for landmark in hand.landmark:

                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            data_array = np.array(
                landmarks
            ).reshape(1, -1)

            # Prediction
            prediction = model.predict(
                data_array
            )[0]

            # Confidence
            probabilities = model.predict_proba(
                data_array
            )[0]

            confidence = float(
                np.max(probabilities) * 100
            )

        tamil = translations.get(
            prediction,
            "—"
        )

        emit(
            "prediction",
            {
                "sign": str(prediction),
                "translation": tamil,
                "confidence": round(
                    confidence,
                    1
                )
            }
        )

    except Exception as e:

        print("Prediction error:", e)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    print()
    print("==============================")
    print(" KAIKURAL AI SERVER")
    print("==============================")
    print("Server: http://127.0.0.1:5000")
    print("==============================")
    print()

    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
