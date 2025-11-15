# 🧵 Silver Thread Detection & Spindle Monitoring Dashboard

A Flask-based machine vision dashboard that detects silver thread defects in textile images using a Random Forest ML model, and provides a live overview of spindle camera connectivity on the factory floor.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-0.104-green)]
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

This project automates silver thread detection in textile manufacturing.
A trained Random Forest Classifier analyzes uploaded images and predicts whether silver thread is present, along with confidence values.

The dashboard also displays the operational status of multiple factory spindles, showing whether camera feeds are connected or offline.


### Key Features

- Image Analysis

Upload any fabric image

Automated silver thread detection using a Random Forest model

Displays detection result + confidence

Processed image preview

- Spindle Monitoring

Shows live spindle grid

Displays status for each spindle (OK / Warning / No Camera)

Clean, dark-themed industrial UI

## 🌐 Web Dashboard (Flask)

Fully responsive HTML + CSS UI

Jinja templating for dynamic content

Lightweight backend suitable for deployment
## 🧠 Tech Stack

**Backend:**

Python
Flask
Joblib (ML model loading)
NumPy
Scikit-Learn (Random Forest)
OpenCV
Pillow
Pandas

**Frontend:**
HTML5
CSS3

📁 Project Structure

```
Loom_Defetct_Detection/
│
├── .venv/                 # Virtual environment
├── static/
│     └── style.css        # Styling
├── templates/
│     └── index.html       # Dashboard UI
├── app.py                 # Flask backend
├── dataset_collection.py  # Dataset
├── deploy_rf
├── prediction_log.csv
├── random_forest_silver_detector.pkl     # ML model
├──README.md              # Project documentation
├──requirements.txt
└──Research_Paper.md      # Research Paper on Topic
```

📊 Machine Learning Model

The silver thread detection feature uses a Random Forest Classifier trained on fabric images.
The model predicts:

Silver Thread Present

Silver Thread Not Detected

Confidence scores are generated from model prediction probabilities.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository or navigate to the project directory:
```bash
cd Loom_Defect_Detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Flask server
```bash
python app.py
```
4. Open in browser
```bash
Navigate to:

http://127.0.0.1:5000
```


## 📊 Models Implemented

1. **Random Forest Classifier**: primary ML model used for defect detection
2. **Baseline Image Processing Checks**: preprocessing + feature extraction

## 🔧 Key Features Generated

### Image-Based Features

## Extracted from images during dataset creation:
- Color histograms
- Texture features
- Edge density
- Pixel intensity patterns
- Shape/defect contours

## Statistical Features
- Mean pixel intensity
- Standard deviation
- Contrast
- Brightness index

## Label Classes
- Defective (Silver Defect)
- Non-defective (Normal Loom Output)


## 📈 Performance Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Prediction latency (ms) — for deployment
- Real-time prediction log (stored in prediction_log.csv)
- Random Forest typically gives:
- Accuracy: ~90–95% (depending on dataset quality)
- Low overfitting

## 💾 Using the Utility Functions

```python
from deploy_rf.predict import predict_defect
from PIL import Image
import numpy as np

# Load trained model
model = joblib.load("random_forest_silver_detector.pkl")

# Load an image
img = Image.open("test_image.jpg")
img_arr = np.array(img)

# Predict
result = predict_defect(model, img_arr)
print("Prediction:", result)

```

## 📝 Model Usage Example

```python
import joblib
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

model = joblib.load("random_forest_silver_detector.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files['file']
    img = Image.open(image).convert("RGB")
    img_arr = np.array(img)

    # extract features + predict
    result = model.predict([img_arr.flatten()])[0]
    
    return jsonify({"prediction": str(result)})

```

## 🎯 Results Summary

Typical performance metrics:
- **Accuracy**: 90–95%
- **High precision for defect detection**
- **Fast inference**


## 📦 Dependencies

Main libraries used:
- **Data Processing**: Opencv numpy
- **Machine Learning**: scikit-learn, joblib, numpy, pillow
- **Visualization**: HTML, CSS, Flask
- **Model Persistence**: joblib

See `requirements.txt` for complete list with versions.

## 🔄 Workflow

1. **Data Collection**: 
- Script: dataset_collection.py
- Captures loom images & labels defects manually
2. **Preprocessing**: 
- Basic image features (color + texture)
3. **Model Training**: 
- Random Forest model
- Saved as random_forest_silver_detector.pkl
4. **Deployment**: 
- Flask API (app.py)
- Dashboard UI (index.html, style.css)
6. **Real-time Monitoring**: 
- All predictions logged into prediction_log.csv
7. **Dashboard Output**: 
- Shows defect status
- Displays uploaded image
- Colored status indicator (green/red)

## 📌 Best Practices 

- Use consistent lighting when collecting images
- Preprocess images (resize, normalize) before predictions
- Retrain model when new defect types appear
- Add more defective samples for better accuracy
- Use train–test split based on batch/time to avoid data leakage


## 📞 Support

For questions or issues, please check:
1. Notebook documentation and comments
2. Utility function docstrings
3. Error messages and logs

## 📄 License

This project is licensed under the MIT License.

## 👤 Author
Arshiya Attar

GitHub: [Arshiya Attar](https://github.com/iamarshiya)

LinkedIn:[Arshiya Attar](https://www.linkedin.com/in/arshiya-attar-91b4ab2b5/)

## 🙏 Acknowledgments

- Built using industry-standard ML libraries
- Follows best practices for time series forecasting
- Designed for production deployment

---

