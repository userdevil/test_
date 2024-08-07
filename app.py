from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import sqlite3
import random
import os
import requests

app = Flask(__name__)

# Define URL and local path for the Keras model
MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/test-75d65.appspot.com/o/cow_muzzle_feature_extractor.h5?alt=media&token=58fe8537-7fe1-45ac-a4e6-b92a3657c7ff"
MODEL_PATH = "cow_muzzle_feature_extractor.h5"

# Function to download the model if it doesn't exist locally
def download_model(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded and saved to {save_path}")
    except requests.RequestException as e:
        print(f"Error downloading model: {e}")
        raise

# Check if model file exists, if not, download it
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Load Keras model
model = load_model(MODEL_PATH)

# Print the expected input shape
input_shape = model.input_shape
print('Expected input shape:', input_shape)

# Database setup
def init_db():
    conn = sqlite3.connect('features.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS features
                      (id INTEGER PRIMARY KEY, feature BLOB, image_id TEXT)''')
    conn.commit()
    conn.close()

# Function to preprocess and load an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image
    return img_array

# Extract features from an image using the Keras model
def extract_features(img_array):
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)  # The layer before the final softmax layer
    features = feature_extractor.predict(img_array)
    return features

# Save features to the database
def save_features(features, image_id):
    conn = sqlite3.connect('features.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO features (feature, image_id) VALUES (?, ?)", (features.tobytes(), image_id))
    conn.commit()
    conn.close()

# Get all features from the database
def get_all_features():
    conn = sqlite3.connect('features.db')
    cursor = conn.cursor()
    cursor.execute("SELECT feature, image_id FROM features")
    rows = cursor.fetchall()
    conn.close()
    return [(np.frombuffer(row[0], dtype=np.float32), row[1]) for row in rows]

# Calculate cosine similarity
def cosine_similarity(features1, features2):
    dot_product = np.dot(features1, features2.T)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    return dot_product / (norm1 * norm2)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    # Define temporary paths for saving files
    temp_dir = 'temp_dir'
    temp_path = os.path.join(temp_dir, file.filename)
    
    # Ensure the temp directory exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        # Save the files temporarily
        file.save(temp_path)

        # Preprocess the images from the saved file paths
        img_array = preprocess_image(temp_path)
        features = extract_features(img_array).flatten()

        all_features = get_all_features()
        for saved_features, image_id in all_features:
            similarity = cosine_similarity(features, saved_features)
            if similarity >= 0.841:
                return jsonify({'message': 'ID already exists', 'image_id': image_id})

        new_image_id = str(random.randint(10, 99))
        save_features(features, new_image_id)

        return jsonify({'message': 'New ID Added', 'image_id': new_image_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
