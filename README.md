# 🧠 Real-Time Ultrasound Nerve Segmentation Using U-Net Model

A deep learning-based project that uses the U-Net architecture to automatically segment nerve structures in ultrasound images — assisting medical professionals in real-time during procedures like regional anesthesia.

## 📌 Project Highlights

- 🚀 Built using **U-Net**, a state-of-the-art architecture for medical image segmentation
- 💡 Trained on the **Ultrasound Nerve Segmentation** dataset from [Kaggle](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation)
- 🎯 Achieved **98.98% test accuracy**, Dice coefficient of **0.4194**, and IoU of **0.2653**
- 🛠️ Technologies: **Python**, **TensorFlow/Keras**, **OpenCV**, **scikit-learn**, **Matplotlib**
- 📈 Real-time segmentation output for clinical and educational use

---

## 🧰 Tech Stack

- **Python 3.7+**
- **TensorFlow/Keras**
- **OpenCV**
- **scikit-learn**
- **Matplotlib & Seaborn**
- **Google Colab / Jupyter Notebook**

---

## 📁 Dataset

- Dataset: Ultrasound Nerve Segmentation - [Kaggle Link](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation)
- ~5600 ultrasound image-mask pairs
- Format: 128x128 grayscale `.png` images
- Labels: Binary masks with pixel-wise nerve annotations

---

## 🏗️ U-Net Architecture

The architecture is composed of:
- Contracting path (encoder) with convolutional + max-pooling layers
- Expanding path (decoder) with upsampling + skip connections
- Final 1x1 convolution for binary segmentation mask output

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*X8zU0iL3whjkME8jNQjX8w.png" width="600"/>
</p>

---

## 🧪 Evaluation Metrics

- **Accuracy**: 98.98%
- **Dice Coefficient**: 0.4194
- **Intersection over Union (IoU)**: 0.2653

---

A deep learning-based medical imaging project to automatically detect and segment nerve structures in ultrasound scans using the U-Net model. This tool is designed to assist medical professionals in real-time procedures like regional anesthesia and pain management.

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Overview](#overview)
- [Scope of the Project](#scope-of-the-project)

---

## 📌 Introduction

Ultrasound imaging is widely used in modern medicine due to its safety, cost-effectiveness, and real-time capabilities. One of its critical uses is in **nerve visualization** during procedures like regional anesthesia and pain management.

However, identifying nerves manually in ultrasound images is challenging due to:
- Poor contrast and noise in images
- Anatomical variations across patients
- Ambiguous edges of nerve structures

This project proposes an AI-powered solution using a **U-Net deep learning model** for **automatic nerve segmentation**. The model learns from labeled datasets and can accurately highlight nerve structures in new ultrasound images — improving decision-making, safety, and efficiency in clinical practice.

### Why U-Net?

- Designed for medical image segmentation
- Can learn from both global features and fine details
- Includes **skip connections** for retaining spatial information
- Works well even on small datasets (with augmentation)

The model is trained using a loss function and evaluated with metrics like:
- **Accuracy**
- **Precision**
- **Recall**
- **IoU (Intersection over Union)**

Once trained, the system performs **real-time segmentation**, suitable for integration into ultrasound machines or desktop/mobile applications. It also serves educational purposes and lays the foundation for other tasks like vessel, organ, or tumor segmentation.

---

## 📋 Overview

The project focuses on developing a **real-time intelligent system** for automatic nerve segmentation using the **U-Net model**.

### Key Goals:
- Assist doctors during **ultrasound-guided procedures**
- Reduce **manual effort** and **human error**
- Enable **real-time segmentation** of nerves
- Work effectively even on **low-quality images** through data augmentation

### How It Works:

1. **Dataset Preparation**:
   - Ultrasound images with labeled masks (ground truth)
   - Augmented using flips, rotations, brightness, zooming, etc.

2. **Model Architecture**:
   - U-Net with encoder (contracting path) and decoder (expanding path)
   - Skip connections preserve high-resolution features

3. **Model Evaluation**:
   - Performance measured using accuracy, precision, recall, and IoU

4. **Deployment**:
   - Fast, optimized system usable on standard hardware
   - Can run on desktops, tablets, or mobile devices

### Advantages:
- Saves doctor’s time and improves accuracy
- Reliable for training medical students and junior radiologists
- Adaptable for segmenting other medical structures (e.g., tumors, vessels)
- Can be deployed in rural/remote settings

---

## 🔍 Scope of the Project

### Core Features:
- Real-time **nerve segmentation** from ultrasound images
- High precision using the **U-Net architecture**
- **Fast performance** for use in operating rooms and clinical procedures

### Training Pipeline:
- Preprocessing and augmentation of medical images
- Evaluation using relevant metrics
- Optimized for standard CPU/GPU devices

### Practical Applications:
- Hospitals, clinics, and educational setups
- Potential use in **telemedicine** and **remote diagnosis**
- Supports expansion to other medical segmentation tasks (e.g., veins, tumors)

### Future Expansion:
- **Mobile deployment** for rural areas
- **Integration with hospital systems** like PACS or EHR
- Testing newer architectures like **U-Net++** and **Attention U-Net**
- Scalable framework for diverse medical use-cases

---

> 🔬 This project demonstrates how AI, particularly deep learning, can transform healthcare by assisting clinicians, improving accuracy, and reducing diagnostic delays.

## ⚙️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/RSathvikareddy/Real-Time-Ultrasound-Nerve-Segmentation-Using-U-Net-Model.git
cd Real-Time-Ultrasound-Nerve-Segmentation-Using-U-Net-Model

