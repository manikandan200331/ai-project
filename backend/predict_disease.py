import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use NEW trained model (.keras)
model_path = os.path.join(BASE_DIR, "model", "plant_model.keras")

# Check model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load model
model = tf.keras.models.load_model(model_path, compile=False)

# Class labels (must match training)
classes = [
    "peper_bell_bacterial_spot",
    "peper_bell_healthy",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight"
]

def predict_plant(img_path):
    try:
        # Load image (PIL)
        img = image.load_img(img_path, target_size=(128, 128))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Prediction
        pred = model.predict(x, verbose=0)
        class_index = int(np.argmax(pred))
        plant_disease = classes[class_index]
        confidence = float(pred[0][class_index] * 100)

        # OpenCV read
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            return "error", 0, 0, "Image read failed", "-", "-"

        # Convert grayscale & threshold
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)

        total_pixels = img_cv.shape[0] * img_cv.shape[1]
        affected_pixels = np.sum(thresh == 255)
        affected_percent = float((affected_pixels / total_pixels) * 100)

        if "healthy" in plant_disease:
            affected_percent = 0.0

        # Reasons
        reasons = {
            "peper_bell_bacterial_spot": "Bacterial infection caused by Xanthomonas.",
            "peper_bell_healthy": "Plant is healthy.",
            "potato_early_blight": "Fungal infection due to humid conditions.",
            "potato_healthy": "Healthy leaf.",
            "potato_late_blight": "Caused by Phytophthora infestans.",
            "tomato_early_blight": "Fungal disease with brown spots.",
            "tomato_healthy": "No disease detected.",
            "tomato_late_blight": "Spreads in wet and cool weather."
        }

        # Remedies
        remedies = {
            "peper_bell_bacterial_spot": {"organic": "Remove infected leaves.", "chemical": "Copper spray."},
            "peper_bell_healthy": {"organic": "Maintain nutrients.", "chemical": "No need."},
            "potato_early_blight": {"organic": "Neem oil spray.", "chemical": "Mancozeb."},
            "potato_healthy": {"organic": "Proper watering.", "chemical": "No need."},
            "potato_late_blight": {"organic": "Remove infected plants.", "chemical": "Copper fungicide."},
            "tomato_early_blight": {"organic": "Neem oil.", "chemical": "Chlorothalonil."},
            "tomato_healthy": {"organic": "No treatment.", "chemical": "No need."},
            "tomato_late_blight": {"organic": "Improve airflow.", "chemical": "Copper fungicide."}
        }

        reason = reasons.get(plant_disease, "Not available")
        remedy_data = remedies.get(plant_disease, {"organic": "-", "chemical": "-"})

        return (
            plant_disease,
            round(confidence, 2),
            round(affected_percent, 2),
            reason,
            remedy_data["organic"],
            remedy_data["chemical"]
        )

    except Exception as e:
        return "error", 0, 0, str(e), "-", "-"