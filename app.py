import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Load the model
model = tf.keras.models.load_model("artifacts/data_ingestion/model/my_model.h5")

# Load class names
with open("artifacts/class_names.json", "r") as f:
    class_names = json.load(f)

# Image preprocessing
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to match your training size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("♻️ Waste Image Classifier")
st.write("Upload an image of **Organic** or **Recyclable** waste to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Prediction: **{predicted_class}**")
