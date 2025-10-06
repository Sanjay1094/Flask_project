import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import base64
import cv2
import requests
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

app = Flask(__name__)

# ==========================
# Google Drive DenseNet Model
# ==========================
MODEL_PATH = "DenseNet_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1k4h7nCN8TniafdXN64YoMO5WFKnix2iZ"

if not os.path.exists(MODEL_PATH):
    print("Downloading DenseNet_model.h5...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Download completed ✅")
else:
    print("DenseNet_model.h5 already exists.")

# ==========================
# Load Models
# ==========================
densenet_model = tf.keras.models.load_model(MODEL_PATH)
print("✅ DenseNet model loaded.")

# Optional models
try:
    lstm_model = tf.keras.models.load_model("LSTM_model.h5")
    print("✅ LSTM model loaded.")
except:
    lstm_model = None
    print("⚠️ LSTM_model.h5 not found. Skipping...")

try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgboost_model.json")
    print("✅ XGBoost model loaded.")
except:
    xgb_model = None
    print("⚠️ xgboost_model.json not found. Skipping...")

try:
    stacking_model = joblib.load("stacking_model.joblib")
    print("✅ Stacking model loaded.")
except:
    stacking_model = None
    print("⚠️ stacking_model.joblib not found. Skipping...")

# ==========================
# Image Requirements
# ==========================
REQUIRED_SIZE = (512, 512)

def preprocess_image(image_path, image_size=(256, 256)):
    try:
        img = load_img(image_path, target_size=image_size, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape(1, *image_size, 1)
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")

def validate_ct_scan(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "Invalid image format"
    if img.shape[:2] != REQUIRED_SIZE:
        return False, f"Invalid image size. Expected {REQUIRED_SIZE}, but got {img.shape[:2]}"
    return True, "Valid CT scan image"

# ==========================
# Routes for Research Papers
# ==========================
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/papers19')
def papers19():
    return render_template('papers19.html')

@app.route('/papers20')
def papers20():
    return render_template('papers20.html')

@app.route('/papers21')
def papers21():
    return render_template('papers21.html')

@app.route('/papers22')
def papers22():
    return render_template('papers22.html')

@app.route('/papers23')
def papers23():
    return render_template('papers23.html')

@app.route('/papers24')
def papers24():
    return render_template('papers24.html')

# ==========================
# Prediction Logic
# ==========================
def predict_cancer(image_path):
    img = preprocess_image(image_path)
    feature_map = densenet_model.predict(img)

    if lstm_model is not None and stacking_model is not None:
        feature_map_reshaped = feature_map.reshape(1, -1, 1)
        lstm_feature = lstm_model.predict(feature_map_reshaped)
        combined_features = np.hstack((feature_map.reshape(1, -1), lstm_feature))
        prediction = stacking_model.predict(combined_features)[0]
        prediction_proba = stacking_model.predict_proba(combined_features)[0][1] * 100
    else:
        prediction = np.argmax(feature_map)
        prediction_proba = np.max(feature_map) * 100

    return ("Cancer detected", prediction_proba) if prediction == 1 else ("No cancer detected", prediction_proba)

def highlight_cancer_region(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    red_mask = np.zeros_like(img)
    red_mask[:, :, 2] = threshold
    highlighted_img = cv2.addWeighted(img, 0.7, red_mask, 0.3, 0)
    segmented_path = "segmented_image.jpg"
    cv2.imwrite(segmented_path, highlighted_img)
    return segmented_path

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'result': 'error', 'message': 'No image uploaded'})

    file = request.files['image']
    temp_file = "temp_image.jpg"
    file.save(temp_file)

    is_valid, validation_message = validate_ct_scan(temp_file)
    if not is_valid:
        os.remove(temp_file)
        return jsonify({'result': 'error', 'message': validation_message})

    try:
        with open(temp_file, "rb") as img_file:
            input_image = base64.b64encode(img_file.read()).decode('utf-8')

        preprocessed_image = preprocess_image(temp_file)
        preprocessed_path = "preprocessed_image.jpg"
        save_img(preprocessed_path, preprocessed_image[0])
        with open(preprocessed_path, "rb") as img_file:
            preprocessed_image_encoded = base64.b64encode(img_file.read()).decode('utf-8')

        prediction_result, accuracy = predict_cancer(temp_file)

        if "Cancer detected" in prediction_result:
            segmented_path = highlight_cancer_region(temp_file)
        else:
            segmented_path = temp_file

        with open(segmented_path, "rb") as img_file:
            segmented_image_encoded = base64.b64encode(img_file.read()).decode('utf-8')

        os.remove(temp_file)
        os.remove(preprocessed_path)
        if "Cancer detected" in prediction_result:
            os.remove(segmented_path)

        return jsonify({
            'input_image': input_image,
            'preprocessed_image': preprocessed_image_encoded,
            'segmented_image': segmented_image_encoded,
            'result': prediction_result,
            'accuracy': f"{accuracy:.2f}%"
        })
    except Exception as e:
        return jsonify({'result': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
