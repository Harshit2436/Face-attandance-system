from flask import Flask, render_template, Response
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# ========== CONFIG ==========
WAIT_TIME = 5           # seconds between IN and OUT
CONFIDENCE_LIMIT = 70   # lower = stricter recognition
# ============================

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_model.yml")

label_map = np.load("models/labels.npy", allow_pickle=True).item()

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

# Memory to prevent duplicates
last_action = {}   # name -> "IN" or "OUT"
last_time = {}     # name -> datetime


def mark_attendance(name, action):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[name, action, time]],
                      columns=["Name", "Action", "Time"])
    df.to_csv("attendance.csv", mode="a", header=False, index=False)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        now = datetime.now()

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_LIMIT:
                name = label_map[label]

                # Initialize user state
                if name not in last_action:
                    last_action[name] = "OUT"
                    last_time[name] = now - timedelta(seconds=WAIT_TIME)

                time_diff = (now - last_time[name]).seconds

                # Punch IN
                if last_action[name] == "OUT":
                    last_action[name] = "IN"
                    last_time[name] = now
                    message = "PUNCHED IN"
                    mark_attendance(name, "IN")

                # Punch OUT after wait time
                elif last_action[name] == "IN":
                    if time_diff < WAIT_TIME:
                        message = f"WAIT {WAIT_TIME - time_diff}s"
                    else:
                        last_action[name] = "OUT"
                        last_time[name] = now
                        message = "PUNCHED OUT"
                        mark_attendance(name, "OUT")

                cv2.putText(
                    frame,
                    f"{name} - {message}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
