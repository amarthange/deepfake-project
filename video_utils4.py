# video_utils.py
import os
import cv2
import numpy as np
from keras.models import load_model
from deepface import DeepFace

# Save the uploaded video to a file
def save_uploaded_file(uploaded_file, save_path="uploaded_video.mp4"):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Extract frames from video
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Detect faces in frames using DeepFace
def detect_faces_in_frames(frames):
    detected_faces = []
    for frame in frames:
        try:
            result = DeepFace.detectFace(frame, detector_backend='opencv', enforce_detection=False)
            result = (result * 255).astype(np.uint8)  # Convert to displayable image
        except:
            result = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image if detection fails
        detected_faces.append(result)
    return detected_faces

# Load trained Xception model
def load_xception_model(model_path="xception_deepfake_model.keras"):
    return load_model(model_path)

# Preprocess a single frame for Xception model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (299, 299))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame
