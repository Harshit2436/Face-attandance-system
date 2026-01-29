import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ---------------------------------------
# Config for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------------------------
# Load face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load face recognition model
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("models/face_model.yml")
    label_map = np.load("models/labels.npy", allow_pickle=True).item()
except Exception as e:
    st.error("Face recognition model not found. Please train first.")
    st.stop()

# Initialize attendance dataframe
try:
    df = pd.read_csv("attendance.csv")
except:
    df = pd.DataFrame(columns=["Name", "Action", "Time"])

# Keep track of last action
last_action = {}

# ---------------------------------------
# Video Transformer for Streamlit WebRTC
class FaceAttendance(VideoTransformerBase):
    def __init__(self):
        self.df = df
        self.last_action = last_action

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            if confidence < 70:
                name = label_map[label]
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Punch in / out logic
                if name not in self.last_action or self.last_action[name] == "OUT":
                    action = "IN"
                    self.last_action[name] = "IN"
                else:
                    action = "OUT"
                    self.last_action[name] = "OUT"

                # Append to dataframe
                new_row = {"Name": name, "Action": action, "Time": current_time}
                self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
                self.df.to_csv("attendance.csv", index=False)

                # Display text on frame
                text = f"{name} - Punch {action}"
                cv2.putText(img, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

# ---------------------------------------
# Streamlit UI
st.title("🟢 Face Recognition Attendance System")
st.write("Automatically punch IN / OUT when face is detected")

# Run WebRTC
webrtc_ctx = webrtc_streamer(
    key="attendance",
    video_transformer_factory=FaceAttendance,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# Show live attendance table
st.subheader("Attendance Records")
st.dataframe(df)


