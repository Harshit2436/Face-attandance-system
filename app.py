import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import threading
import time

# -----------------------------
# WebRTC config for browser camera
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# Load face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load face recognition model
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("models/face_model.yml")
    label_map = np.load("models/labels.npy", allow_pickle=True).item()
except:
    st.error("Face recognition model not found. Please train first.")
    st.stop()

# Initialize attendance dataframe
try:
    df = pd.read_csv("attendance.csv")
except:
    df = pd.DataFrame(columns=["Name", "Action", "Time"])

# Keep track of last action per person
last_action = {}
lock = threading.Lock()

# -----------------------------
# Video transformer class
class FaceAttendance(VideoTransformerBase):
    def __init__(self):
        self.df = df
        self.last_action = last_action
        self.last_time = {}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < 70:
                name = label_map[label]

                now = datetime.now()
                # Avoid duplicate punches within 5 seconds
                if name not in self.last_time or (now - self.last_time[name]).seconds > 5:
                    # Determine action
                    if name not in self.last_action or self.last_action[name] == "OUT":
                        action = "IN"
                        self.last_action[name] = "IN"
                    else:
                        action = "OUT"
                        self.last_action[name] = "OUT"

                    self.last_time[name] = now

                    # Append to dataframe
                    with lock:
                        new_row = {"Name": name, "Action": action, "Time": now.strftime("%Y-%m-%d %H:%M:%S")}
                        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
                        self.df.to_csv("attendance.csv", index=False)

                    # Put text on frame
                    cv2.putText(img, f"{name} - Punch {action}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

# -----------------------------
# Streamlit UI
st.title("🟢 Face Recognition Attendance System")
st.write("Automatically punches IN / OUT when a registered face is detected.")

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="attendance",
    video_transformer_factory=FaceAttendance,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# Show live attendance table
st.subheader("Attendance Records")
st.dataframe(df)
