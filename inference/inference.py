import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Paths
model_path = "artifacts/data_ingestion/model/my_model.h5"
class_names_path = "artifacts/class_names.json"
image_path = "inference/sample.jpg"  # 👈 Replace with your actual image path

# Load model and class names
model = load_model(model_path)

with open(class_names_path, 'r') as f:
    class_names = json.load(f)

# Image size (should match model input)
image_size = 64

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    predicted_class = np.argmax(preds, axis=1)[0]
    print(f"✅ Prediction: {class_names[predicted_class]}")

if __name__ == "__main__":
    predict(image_path)

