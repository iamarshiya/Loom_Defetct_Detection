**🧵 Silver Thread Detection & Spindle Monitoring Dashboard**

A Flask-based machine vision dashboard that detects silver thread defects in textile images using a Random Forest ML model, and provides a live overview of spindle camera connectivity on the factory floor.

**Project Overview**

This project automates silver thread detection in textile manufacturing.
A trained Random Forest Classifier analyzes uploaded images and predicts whether silver thread is present, along with confidence values.

The dashboard also displays the operational status of multiple factory spindles, showing whether camera feeds are connected or offline.

**✨ Key Features**

- Image Analysis

Upload any fabric image

Automated silver thread detection using a Random Forest model

Displays detection result + confidence

Processed image preview

- Spindle Monitoring

Shows live spindle grid

Displays status for each spindle (OK / Warning / No Camera)

Clean, dark-themed industrial UI

**🌐 Web Dashboard (Flask)**

Fully responsive HTML + CSS UI

Jinja templating for dynamic content

Lightweight backend suitable for deployment

**🧠 Tech Stack**

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

**📁 Project Structure**

Loom_Defetct_Detection/
│
├── .venv/ # Virtual environment
├── static/
│ └── style.css # Styling
│
├── templates/
│ └── index.html # Dashboard UI
│
├── app.py # Flask backend
├── dataset_collection.py # Dataset collection script
├── deploy_rf/ # Deployment files
├── prediction_log.csv # Inference log
├── random_forest_silver_detector.pkl # Trained ML model
│
├── README.md # Project documentation
├── requirements.txt # Dependencies
└── Research_Paper.md # Research paper on Loom Defect Detection

**📊 Machine Learning Model**

The silver thread detection feature uses a Random Forest Classifier trained on fabric images.
The model predicts:

Silver Thread Present

Silver Thread Not Detected

Confidence scores are generated from model prediction probabilities.

**🛠️ How It Works**

User uploads a fabric image

Backend preprocesses image using OpenCV/PIL

ML model predicts thread presence

UI updates dynamically with:

Prediction label

Confidence percentage

Image preview

Spindle status grid displays simulated live factory feed status


**🧑‍💻 Author**


Arshiya Attar.

Janhavi Pohnerkar.
