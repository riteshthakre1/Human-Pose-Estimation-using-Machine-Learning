import streamlit as st
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'input_Image.jpg'

# Body parts and pose pairs
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

# Input dimensions for the network
inWidth, inHeight = 368, 368
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Streamlit app
st.title("Human Pose Estimation System")
st.text("Ensure the uploaded image has clear visibility of all body parts.")

# Image uploader
img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    image = np.array(Image.open(DEMO_IMAGE))

# Display original image
st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider
thres = st.slider("Threshold for detecting key points", min_value=0, value=20, max_value=100, step=5) / 100


@st.cache_data
def poseDetector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :19, :, :]

    points = []
    for i, heatMap in enumerate(out[0]):
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        if conf > thres:
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

    return frame


# Apply pose detection
output = poseDetector(image)

# Display processed image
st.subheader("Positions Estimated")
st.image(output, caption="Positions Estimated", use_column_width=True)
