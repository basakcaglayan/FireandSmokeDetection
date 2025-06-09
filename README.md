# 🔥  Fire and Smoke Detection with YOLO and Firebase

This project provides a complete solution for detecting fire in images and videos using deep learning and IoT-based techniques. It includes:

- A **fine-tuned YOLOv8 model** for real-time fire detection with bounding boxes.
- A **custom-built CNN classifier** trained from scratch for binary image classification (fire vs. no fire).
- Integration with **Firebase and ESP32-CAM** to monitor images uploaded in real time and trigger alerts via a **buzzer** if fire is detected.
- Webcam-based live detection for desktop setups using OpenCV.

The goal of this project is not only to compare the performance of state-of-the-art vs. from-scratch models, but also to implement a practical fire detection system using affordable IoT components and cloud storage.

---

## 📖 Table of Contents

- [Documentation](#-documentation)
- [Features](#-features)
- [Models Implemented](#-models-implemented)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Environment Variables](#-environment-variables)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
## 📖 Documentation

### Project Overview

This project explores fire detection with:

- A **YOLOv8 model**, fine-tuned on a custom dataset of 9,000 fire and non-fire images created via [Roboflow](https://roboflow.com/).
- A **custom CNN model** built using Keras for image classification.
- Real-time application using **ESP32-CAM**, which captures and uploads images to Firebase, and a Python script polls and processes the images to trigger a hardware alarm system.
### ✨ Features

- **Dual Modeling Approaches**: Object detection (YOLOv8) and binary image classification (CNN).
- **End-to-End Notebooks**: Easy-to-follow Jupyter Notebooks for each modeling method.
- **Fine-Tuned & From-Scratch Models**: Practical comparison of state-of-the-art and fundamental models.
- **Visualization Support**: Training history and model predictions visualized.
- **Lightweight Setup**: Simple environment setup with `requirements.txt`.

## ✨ Features

- 🔍 **YOLOv8 Fire Detection** (Object Detection)
- 🧠 **Custom CNN Classifier** (Binary Image Classification)
- 🔧 **Webcam Detection** via `yolo.py`
- ☁️ **Firebase ESP32-CAM Integration** via `yolo_firebase2.py`
- 🚨 **Buzzer Trigger System** when fire is detected
- 📊 Visual training history, evaluation, and prediction visualization
- 📁 Modular and well-organized code and notebooks

## 🧠 Models Implemented

### 🔥 YOLOv8 Fire Detection
- **Model File**: `best.pt` – Trained YOLOv8 model exported from Google Colab
- **Training Tool**: Ultralytics YOLOv8 via Roboflow export
- **Detection Scripts**:
  - `yolo.py`: Detects fire in real-time using a webcam.
  - `yolo_firebase2.py`: Downloads image frames from Firebase and checks for fire using `best.pt`.

### 🔥 Custom CNN Classifier
- **File**: `FireDetectionwith_CNN.ipynb`
- **Task**: Classify image as "fire" or "no fire"
- **Framework**: TensorFlow/Keras

## Tech Stack

**Client:** React, Redux, TailwindCSS

**Server:** Node, Express

## 💻 Technology Stack

- **Python 3.8+**
- **YOLOv8 (Ultralytics)**
- **TensorFlow & Keras**
- **OpenCV**
- **Firebase SDK**
- **ESP32-CAM** (Hardware)
- **Buzzer Alarm System**
- **Jupyter Notebooks**## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/basakcaglayan/FireDetection.git
cd FireDetection
```
### 2. Create and Activate a Virtual Environment
On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
## 🚀 Usage

### 1. Webcam Fire Detection
Run this script to detect fire using a standard webcam:

```bash
python yolo.py
```
### 2. Firebase + ESP32-CAM Detection (Real-Time)
Upload images from ESP32-CAM to Firebase. The Python script will:

- Download the most recent image
- Run detection using best.pt
- Trigger buzzer if fire is detected

```bash
python yolo_firebase2.py
```
### 3. Notebook-based Experiments
- FireDetection-Yolov8.ipynb: Train or fine-tune YOLOv8
- FireDetectionwith_CNN.ipynb: Train and evaluate the custom CNN
Launch notebooks:

```bash
jupyter notebook
```
## Deployment
You can deploy the detection system on a local Raspberry Pi or any PC with a webcam. Connect a buzzer through GPIO or USB-controlled relay, and modify yolo_firebase2.py to send signals accordingly.
## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Firebase](https://firebase.google.com/)
- [Roboflow](https://roboflow.com/) – for dataset annotation
<<<<<<< HEAD
- [ESP32-CAM Tutorials](https://randomnerdtutorials.com/esp32-cam/)
=======
- [ESP32-CAM Tutorials](https://randomnerdtutorials.com/esp32-cam/)
>>>>>>> c7323478ff296de60eafd9d5683dfbd1ba3cd557
