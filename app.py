import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import base64
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# ==========================
# Image Requirements
# ==========================
REQUIRED_SIZE = (512, 512)

# ==========================
# Lazy-loaded Models
# ==========================
densenet_model = None
lstm_model = None
stacking_model = None
xgb_model = None

def get_densenet_model():
    global densenet_model
    if densenet_model is None:
        densenet_model = tf.keras.models.load_model("DenseNet_model.h5")
    return densenet_model

def get_lstm_model():
    global lstm_model
    if lstm_model is None:
        try:
            lstm_model = tf.keras.models.load_model("LSTM_model.h5")
        except:
            lstm_model = None
    return lstm_model

def get_stacking_model():
    global stacking_model
    if stacking_model is None:
        try:
            stacking_model = joblib.load("stacking_model.joblib")
        except:
            stacking_model = None
    return stacking_model

def get_xgb_model():
    global xgb_model
    if xgb_model is None:
        try:
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model("xgboost_model.json")
        except:
            xgb_model = None
    return xgb_model

# ==========================
# Image preprocessing
# ==========================
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
# Routes for pages
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
# Prediction logic
# ==========================
def predict_cancer(image_path):
    img = preprocess_image(image_path)
    densenet = get_densenet_model()
    feature_map = densenet.predict(img, batch_size=1)

    lstm = get_lstm_model()
    stacking = get_stacking_model()

    if lstm and stacking:
        feature_map_reshaped = feature_map.reshape(1, -1, 1)
        lstm_feature = lstm.predict(feature_map_reshaped)
        combined_features = np.hstack((feature_map.reshape(1, -1), lstm_feature))
        prediction = stacking.predict(combined_features)[0]
        prediction_proba = stacking.predict_proba(combined_features)[0][1] * 100
    else:
        # fallback if LSTM/stacking not available
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

# ==========================
# Flask route for prediction
# ==========================
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
        # Encode input image
        with open(temp_file, "rb") as f:
            input_image = base64.b64encode(f.read()).decode('utf-8')

        preprocessed_image = preprocess_image(temp_file)
        preprocessed_path = "preprocessed_image.jpg"
        save_img(preprocessed_path, preprocessed_image[0])
        with open(preprocessed_path, "rb") as f:
            preprocessed_image_encoded = base64.b64encode(f.read()).decode('utf-8')

        # Get prediction
        prediction_result, accuracy = predict_cancer(temp_file)

        if "Cancer detected" in prediction_result:
            segmented_path = highlight_cancer_region(temp_file)
        else:
            segmented_path = temp_file

        with open(segmented_path, "rb") as f:
            segmented_image_encoded = base64.b64encode(f.read()).decode('utf-8')

        # Cleanup
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
