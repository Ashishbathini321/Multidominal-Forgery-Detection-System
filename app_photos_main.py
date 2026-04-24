# app_photos_main.py

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from utils.ai_detector import is_ai_image


def _read(upload):
    data = upload.read()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# =========================
# 🔹 SINGLE PHOTO (AI vs Real)
# =========================
def run_photos():
    st.subheader("Single Photo — Real or AI")

    up = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if not up:
        return

    img = _read(up)
    if img is None:
        st.error("Error reading image")
        return

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 🔹 Get probability from detector
    is_ai, prob = is_ai_image(img)

    # 🔥 YOUR RULE (with slight stability adjustment)
    THRESH = 0.60

    if prob >= THRESH:
        st.error(f"AI Generated ❌ ({prob*100:.1f}%)")
    else:
        st.success(f"Real Image ✅ ({prob*100:.1f}%)")


# =========================
# 🔹 FACE MATCH (DeepFace)
# =========================
def run_compare_photos():
    st.subheader("Face Comparison (Improved AI Model)")

    col1, col2 = st.columns(2)

    with col1:
        f1 = st.file_uploader("Image 1", type=["jpg", "png"], key="a")

    with col2:
        f2 = st.file_uploader("Image 2", type=["jpg", "png"], key="b")

    if not f1 or not f2:
        return

    if st.button("Compare"):
        img1 = _read(f1)
        img2 = _read(f2)

        st.image([img1, img2], width=300)

        try:
            result = DeepFace.verify(
                img1,
                img2,
                model_name="Facenet",
                distance_metric="cosine",
                enforce_detection=False
            )

            distance = result["distance"]

            st.write(f"Distance: {distance:.3f}")

            # 🔥 CUSTOM DECISION (VERY IMPORTANT)
            # lower distance = more similar
            if distance < 0.40:
                st.success("Faces Match ✅")

            elif distance < 0.55:
                st.warning("Likely Same Person ⚠️")

            else:
                st.error("Faces Do Not Match ❌")

        except Exception as e:
            st.error(f"Error: {str(e)}")