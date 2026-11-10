# 🚀 AI Rope Jump Counter Using YOLO11 Pose Estimation

Build an AI-powered Rope Jump Counter using Python, OpenCV, and YOLO11 Pose Estimation! 🤖

This project automatically detects and counts jumping movements using human pose estimation. Instead of manually counting repetitions, the system tracks body keypoints, analyzes vertical body displacement, and identifies complete jump cycles in real time.

## 🎥 YouTube Tutorial

Watch the complete project implementation on YouTube:

👉 [Watch the Full Tutorial](https://youtu.be/meXFetXjIJI)

## 🔥 Key Features

- **YOLO11 Pose Estimation:** Detect human body keypoints.
- **Hip Movement Tracking:** Calculate vertical body displacement using left and right hip coordinates.
- **Automatic Jump Counting:** Detect and count completed jump cycles.
- **Movement Smoothing:** Use a Moving Average technique to reduce unwanted fluctuations.
- **State-Based Detection:** Identify GROUND, RISING, and FALLING movement states.
- **Hysteresis:** Reduce false detections and duplicate jump counting.
- **Adaptive Baseline:** Use Exponential Moving Average (EMA) to adjust to gradual posture changes.
- **Real-Time Visualization:** Display jump counts and current movement states using OpenCV.

## 🛠️ Technologies Used

- Python
- Ultralytics YOLO11 Pose
- OpenCV
- NumPy
- Collections (Deque)
- Computer Vision
- Human Pose Estimation

## 📌 Project Highlights

This project demonstrates how human pose estimation and motion analysis can be combined to develop an intelligent fitness application.

It also explores practical techniques such as smoothing, hysteresis, state transitions, and adaptive baseline tracking to make movement-based counting more reliable.

⭐ If you find this project useful, don't forget to star this repository and subscribe to the YouTube channel for more exciting Computer Vision and AI projects!
