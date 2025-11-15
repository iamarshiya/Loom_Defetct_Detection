from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image, ImageOps
import os
from datetime import datetime

app = Flask(__name__)

# --- Load trained Random Forest model ---
model = joblib.load("C:/Users/Admin/Desktop/CI Project/random_forest_silver_detector.pkl")
print("Model loaded successfully!")

# --- Preprocessing identical to training ---
def preprocess_image(path, size=(64, 64)):
    img = Image.open(path).convert('L')      # grayscale
    img = ImageOps.pad(img, size, color=0)   # pad while preserving aspect ratio
    return np.array(img).flatten().reshape(1, -1)

# --- Prediction function ---
def predict_image(image_path):
    features = preprocess_image(image_path)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    if prediction == 0:
        label = "Silver Thread Detected"
        confidence = prob[0]
        status_color = "green"
    else:
        label = "No Silver Thread Detected"
        confidence = prob[1]
        status_color = "red"

    return label, confidence, status_color


# --- Define available spindles ---
SPINDLES = [
    {"id": 1, "name": "Spindle #1", "status": "No camera connected", "color": "gray"},
    {"id": 2, "name": "Spindle #2", "status": "No camera connected", "color": "gray"},
    {"id": 3, "name": "Spindle #3", "status": "No camera connected", "color": "gray"},
    {"id": 4, "name": "Spindle #4", "status": "No camera connected", "color": "gray"},
    {"id": 5, "name": "Spindle #5", "status": "No camera connected", "color": "gray"},
    {"id": 6, "name": "Spindle #6", "status": "No camera connected", "color": "gray"},
]


# --- Dashboard route ---
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    result = None
    confidence = None
    image_path = None
    status_color = "gray"

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('dashboard.html', result="No file selected", spindles=SPINDLES)

        # Save uploaded file to static folder
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        result, confidence, status_color = predict_image(filepath)

        # --- Log results safely ---
        with open("prediction_log.csv", "a", encoding="utf-8") as log:
            log.write(f"{datetime.now()},{file.filename},{result},{confidence:.3f}\n")

        return render_template(
            'dashboard.html',
            result=result,
            confidence=f"{confidence*100:.2f}%",
            image_path=filepath,
            status_color=status_color,
            spindles=SPINDLES
        )

    return render_template('dashboard.html', spindles=SPINDLES)


if __name__ == '__main__':
    app.run(debug=True)
