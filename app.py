import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

# Load model
model = joblib.load('best_modelCNN.joblib')

# Gunakan path cascade bawaan OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

st.title("Drunk vs Sober Face Classifier (Haar Cascade)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if len(faces) == 0:
        st.warning("Tidak ada wajah terdeteksi pada gambar.")
    else:
        for i, (x, y, w, h) in enumerate(faces):
            face_img = img_np[y:y+h, x:x+w]
            st.image(face_img, caption=f"Wajah #{i+1}", width=150)
            
            # Preprocessing
            face_resized = Image.fromarray(face_img).resize((224, 224))
            face_array = np.array(face_resized) / 255.0
            face_array = face_array.reshape(1, 224, 224, 3)
            
            # Prediksi
            pred = model.predict(face_array)
            label = np.argmax(pred, axis=1)[0]
            label_str = "Drunk" if label == 0 else "Sober"
            st.write(f"Hasil prediksi: **{label_str}**")
