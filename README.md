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

## ⚙️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/RSathvikareddy/Real-Time-Ultrasound-Nerve-Segmentation-Using-U-Net-Model.git
cd Real-Time-Ultrasound-Nerve-Segmentation-Using-U-Net-Model
