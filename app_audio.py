import streamlit as st
import numpy as np
import librosa

def run_audio():
    st.subheader("Audio Fake Detection")

    a = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if not a:
        return

    y, sr = librosa.load(a, sr=None)

    # Feature
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    if spectral_centroid > 3000:
        st.error("AI Generated Audio ❌")
    else:
        st.success("Real Audio ✅")

    st.write(f"Spectral Centroid: {spectral_centroid:.2f}")