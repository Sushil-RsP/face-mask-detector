import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Face Mask Detector", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    
    p.title {
        text-align: center;
        font-size: 46px;
        font-weight: bold;
        color: #00ffcc;
    }
    
    p.subtitle {
        text-align: center;
        color: #bbbbbb;
        font-weight: bold;
    }
    
    p.no_mask {
    text-align: center;
    font-size: 18px;
    color: black; 
    font-weight: bold;
    background-color: #f69697; 
    padding: 10px;
    border-radius: 5px;
    }
    
    p.with_mask {
    text-align: center;
    font-size: 18px;
    color: black; 
    font-weight: bold;
    background-color: #aaf0c9; 
    padding: 10px;
    border-radius: 5px;
    }
    
    p.incorrect_mask {
    text-align: center;
    font-size: 18px;
    color: black; 
    font-weight: bold;
    background-color: #f1ee8e; 
    padding: 10px;
    border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">😷 Face Mask Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and detect mask status</p>', unsafe_allow_html=True)

# Load model
model = load_model("mask_detection.h5")
classes = ['Incorrect Mask', 'No Mask', 'Mask Worn']

# Face detection
def detect_and_crop_face(img):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    return img

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = detect_and_crop_face(img)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    pred = model.predict(img.reshape(1, 128, 128, 3))
    label = np.argmax(pred)
    confidence = pred[0][label] * 100

    with col2:
        st.write("### 🧠 Prediction Result")

        if label == 2:
            st.markdown(f'<p class="with_mask">Mask Detected ({confidence:.2f}%)</p>', unsafe_allow_html=True)
        elif label == 1:
            st.markdown(f'<p class="no_mask">No Mask ({confidence:.2f}%)</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="incorrect_mask">Incorrect Mask ({confidence:.2f}%)</p>', unsafe_allow_html=True)

        st.progress(int(confidence))