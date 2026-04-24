import streamlit as st

from app_photos_main import run_photos, run_compare_photos
from app_video import run_video
from app_audio import run_audio

st.set_page_config(page_title="Forgery Detection", page_icon="🛡️")

st.title("🛡️ Multimodal Forgery Detection System")

# Sidebar navigation
tool = st.sidebar.radio(
    "Choose Tool",
    [
        "Photo (AI or Real)",
        "Photo vs Photo (Compare Faces)",
        "Single Video (Fake Check)",
        "Single Audio (Fake Check)"
    ]
)

# Routing
if tool == "Photo (AI or Real)":
    run_photos()

elif tool == "Photo vs Photo (Compare Faces)":
    run_compare_photos()

elif tool == "Single Video (Fake Check)":
    run_video()

elif tool == "Single Audio (Fake Check)":
    run_audio()