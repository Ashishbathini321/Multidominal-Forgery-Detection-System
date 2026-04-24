import streamlit as st
import cv2
import numpy as np

def run_video():
    st.subheader("Video Fake Detection")

    v = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if not v:
        return

    path = "temp_video.mp4"
    with open(path, "wb") as f:
        f.write(v.read())

    cap = cv2.VideoCapture(path)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret or count > 30:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        count += 1

    cap.release()

    if len(frames) == 0:
        st.error("Could not process video")
        return

    score = np.mean(frames)

    # Simple logic
    if score < 80:
        st.error("AI Generated Video ❌")
    else:
        st.success("Real Video ✅")

    st.write(f"Sharpness Score: {score:.2f}")