# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from video_utils4 import (
    save_uploaded_file,
    extract_frames,
    detect_faces_in_frames,
    load_xception_model,
    preprocess_frame
)
import numpy as np

# Updated detect_deepfake_in_frames function
def detect_deepfake_in_frames(frames, model, return_scores=False):
    results = []
    scores = []
    for frame in frames:
        preprocessed = preprocess_frame(frame)
        prediction = model.predict(preprocessed, verbose=0)
        label = "Fake" if prediction[0][0] > 0.5 else "Real"
        results.append(label)
        scores.append(prediction[0])
    return (results, scores) if return_scores else results

# Load the pre-trained Xception model
model = load_xception_model("xception_deepfake_model.keras")

st.set_page_config(page_title="Deepfake Detection", layout="centered")
st.title("🎭 Deepfake Detection")
st.markdown("Upload a video, and we’ll detect faces and deepfakes in the frames.")

uploaded_video = st.file_uploader("Upload your video (mp4 only)", type=["mp4"])

if uploaded_video is not None:
    video_path = save_uploaded_file(uploaded_video)
    st.success("Video uploaded successfully!")

    with st.spinner("Extracting frames..."):
        frames = extract_frames(video_path)
        st.success(f"{len(frames)} frames extracted.")

    with st.spinner("Analyzing frames for faces and deepfakes..."):
        detected_faces = detect_faces_in_frames(frames)
        deepfake_results, prediction_scores = detect_deepfake_in_frames(frames, model, return_scores=True)

    st.markdown("### 🧠 Deepfake Detection Results")
    for idx, res in enumerate(deepfake_results[:10]):
        st.image(frames[idx], caption=f"Frame {idx + 1} - {res} (Score: {prediction_scores[idx][0]:.2f})", use_container_width=True)

    st.markdown("### 👥 Faces Detected in Frames")
    for idx, faces in enumerate(detected_faces[:10]):
        st.image(faces, caption=f"Face(s) Detected in Frame {idx + 1}", use_container_width=True)

    # Accuracy Count
    total = len(deepfake_results)
    real_count = deepfake_results.count("Real")
    fake_count = deepfake_results.count("Fake")
    accuracy_percent = (real_count / total) * 100 if total > 0 else 0

    st.markdown(f"### 📊 Accuracy Summary")
    st.write(f"Total Frames: {total}")
    st.write(f"Real: {real_count}, Fake: {fake_count}")
    st.write(f"Accuracy (Real %): {accuracy_percent:.2f}%")

    # Bar Chart
    st.markdown("### 📈 Deepfake Distribution Chart")
    df = pd.DataFrame({"Label": ["Real", "Fake"], "Count": [real_count, fake_count]})
    fig, ax = plt.subplots()
    ax.bar(df["Label"], df["Count"], color=["green", "red"])
    ax.set_ylabel("Frame Count")
    ax.set_title("Real vs Fake Frame Distribution")
    st.pyplot(fig)

    # Download Results as CSV
    st.markdown("### 💾 Download Detection Results")
    result_df = pd.DataFrame({
        "Frame": list(range(1, total + 1)),
        "Prediction": deepfake_results,
        "Score": [f"{score[0]:.2f}" for score in prediction_scores]
    })
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "deepfake_results.csv", "text/csv")
