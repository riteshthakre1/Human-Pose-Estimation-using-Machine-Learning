# Human Pose Estimation using Machine Learning


## Overview
This project demonstrates a machine learning-based system for detecting and estimating human poses in images and videos. Using advanced deep learning frameworks such as TensorFlow and PyTorch, the system identifies key body parts and their spatial relationships. The application features an interactive user interface built with Streamlit and supports image and video processing.

## Features
- **Real-time Pose Estimation**: Detect human poses from images and video streams.
- **Pre-trained Models**: Leverages TensorFlow and PyTorch models for accurate pose detection.
- **Interactive UI**: Upload images or videos for real-time pose visualization using Streamlit.
- **Efficient Processing**: Uses OpenCV, NumPy, and Pandas for optimized image and video handling.

## Technologies Used
- **Programming Language**: Python
- **Libraries/Frameworks**:
  - TensorFlow
  - PyTorch
  - OpenCV
  - Streamlit
  - NumPy
  - Pandas
  - Pillow

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ritesh-kumar-keshri-prasad-thakre/Human-Pose-Estimation.git
cd Human Pose Estimation
```

### Step 2: Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
Install the required libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Alternatively, install manually:
```bash
pip install opencv-python tensorflow torch numpy pandas streamlit pillow
```

### Step 4: Run the Application
```bash
streamlit run Run_est.py
```
This will launch the Streamlit app, where you can upload images or videos to detect and visualize poses.

## Usage
1. **Image Pose Estimation**:
   - Upload an image via the Streamlit interface.
   - The system processes the image and displays the detected pose with key points and connections.

2. **Video Pose Estimation**:
   - Upload a video file.
   - The system processes each frame and visualizes the pose estimation results.

## Results
- **Key Points**: Blue dots on detected body parts.
- **Pose Pairs**: Green lines connecting key points.
