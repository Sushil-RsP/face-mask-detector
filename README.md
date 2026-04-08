# 😷 Face Mask Detection System (CNN + Streamlit)

A deep learning-based web application that detects whether a person is wearing a mask correctly, incorrectly, or not at all.

---

## 🚀 Live Demo
🔗 https://face-mask-detector-sushil.streamlit.app/

---

## 📌 Features

- 🧠 CNN-based image classification model
- 🔍 Face detection using OpenCV (Haar Cascade)
- ⚡ Real-time image prediction
- 🎨 Clean and modern UI using Streamlit
- 🌐 Deployed on Streamlit Cloud

---

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

---

## 📂 Project Structure


face-mask-detector/
│
├── new.py # Streamlit app
├── mask_detection.h5 # Trained CNN model
├── requirements.txt
├── runtime.txt
└── README.md


---

## 🧠 How it Works

1. User uploads an image 📷  
2. Face is detected using OpenCV  
3. Face is cropped and resized (128x128)  
4. CNN model predicts:
   - 😷 Mask Detected  
   - ❌ No Mask  
   - ⚠️ Incorrect Mask  

---

## ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/Sushil-RsP/face-mask-detector.git
cd face-mask-detector

pip install -r requirements.txt
streamlit run new.py
