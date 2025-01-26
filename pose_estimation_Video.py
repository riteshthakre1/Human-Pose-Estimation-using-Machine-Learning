import streamlit as st
import cv2
import numpy as np

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Load the TensorFlow model
model_path = "graph_opt.pb"  # Replace with the correct path to your file
net = cv2.dnn.readNetFromTensorflow(model_path)

# Streamlit setup
st.title("Human Pose Estimation")
st.text("Upload a video file to perform pose estimation.")

# File uploader
video_file = st.file_uploader("Upload a video file (e.g., .mp4, .avi)", type=["mp4", "avi"])

# Threshold slider
thres = st.slider("Threshold for detecting key points", min_value=0, value=20, max_value=100, step=5) / 100


def process_video(video_file, threshold):
    cap = cv2.VideoCapture(video_file)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))  # Resize for consistent display
        frameWidth, frameHeight = frame.shape[1], frame.shape[0]

        # Forward pass
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        points = []
        for i, heatMap in enumerate(out[0]):
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            if conf > threshold:
                x = int((frameWidth * point[0]) / out.shape[3])
                y = int((frameHeight * point[1]) / out.shape[2])
                points.append((x, y))
            else:
                points.append(None)

        for partFrom, partTo in POSE_PAIRS:
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[idFrom], 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[idTo], 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # Convert frame to RGB for Streamlit
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1

    cap.release()
    return frames


if video_file:
    # Save video temporarily for processing
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.text("Processing video...")
    frames = process_video("temp_video.mp4", thres)
    st.text("Processing complete. Displaying video...")

    # Display video frames in Streamlit
    for frame in frames:
        st.image(frame, use_column_width=True)
else:
    st.warning("Please upload a video file to start pose estimation.")
