import joblib
import numpy as np
from PIL import Image, ImageOps

# --- Load trained Random Forest model ---
model = joblib.load("random_forest_silver_detector.pkl")
print("Model loaded")

# --- Preprocessing identical to training ---
def preprocess_image(path, size=(64, 64)):
    img = Image.open(path).convert('L')      # grayscale
    img = ImageOps.pad(img, size, color=0)   # pad while preserving aspect ratio
    return np.array(img).flatten().reshape(1, -1)  # flatten for RF

# --- Prediction ---
def predict_image(image_path):
    features = preprocess_image(image_path)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    if prediction == 0:
        print(f"🩶 {image_path} → Silver Thread Detected")
    else:
        print(f"⬛ {image_path} → No Silver Thread Detected")

    print(f"Predicted probabilities → Silver: {prob[0]:.3f}, Non-silver: {prob[1 ]:.3f}")

# --- Example usage ---
if __name__ == "__main__":
    test_image_path = r"F:\CopperCloud\LOOM\3_Datasets\Binary_images\non_silver\non_silver_024.png"
    predict_image(test_image_path)
