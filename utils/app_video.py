import streamlit as st
from utils.video_pipeline import analyze_video
from utils.core_score import final_decision, confidence_score

def run():
    st.subheader("🎞 Single Video — AI Detection (Fixed)")

    v = st.file_uploader("Upload video", type=["mp4","mov","avi","mpeg4"])

    if st.button("Analyze") and v:
        path = "temp_video.mp4"

        with open(path, "wb") as f:
            f.write(v.read())

        with st.spinner("Analyzing..."):
            score, details = analyze_video(path)

        decision = final_decision(score)
        conf = confidence_score(score)

        st.metric("AI Probability", f"{conf}%")

        if decision == "AI Generated":
            st.error("AI Generated ❌")
        elif decision == "Real":
            st.success("Real Video ✅")
        else:
            st.warning("Uncertain ⚠️")

        with st.expander("Details"):
            st.json(details)