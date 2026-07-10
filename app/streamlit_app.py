import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

@st.cache_resource
def load_pneumonia_model():
    return load_model('../models/pneumonia_cnn_model.keras')

model = load_pneumonia_model()

IMG_SIZE = 150

st.title("Pneumonia Detector from X-Ray")
st.write("Upload a chest X-ray image,and this model will predict whether it shows sign of pneumonia.")

st.warning("This is an educational project, not a medical diagnostic tool. Do not use for real medical decisions.")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE,1)

    prediction = model.predict(img_array)[0] [0]

    if prediction > 0.5:
        st.error(f"Prediction: PNEUMONIA detected (confidence: {prediction:.2f})")
    else:
        st.success(f"Prediction: NORMAL (confidence: {(1 - prediction):.2%})")
