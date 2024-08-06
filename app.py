from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import sqlite3
import random
from PIL import Image

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="cow_muzzle_feature_extractor.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print the expected input shape
print('Expected input shape:', input_details[0]['shape'])

# Database setup
def init_db():
    conn = sqlite3.connect('features.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS features
                      (id INTEGER PRIMARY KEY, feature BLOB, image_id TEXT)''')
    conn.commit()
    conn.close()

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure RGB
    target_size = tuple(input_details[0]['shape'][1:3])  # Get target size from model
    img = img.resize(target_size)  # Adjust to model's expected size
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    print("Image shape after preprocessing:", img_array.shape)  # Check shape

    return img_array

# Extract features from an image using the TFLite model
def extract_features(img_array):
    # Ensure that img_array is in the correct shape
    img_array = np.array(img_array, dtype=np.float32)  # Ensure the type is float32
    if not np.array_equal(img_array.shape, input_details[0]['shape']):
        raise ValueError(f"Input shape {img_array.shape} does not match model's expected shape {input_details[0]['shape']}")
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

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

    try:
        img = Image.open(io.BytesIO(file.read()))

        # Process the image from the saved file path
        img_array = preprocess_image(img)
        features = extract_features(img_array).flatten()

        all_features = get_all_features()
        for saved_features, image_id in all_features:
            similarity = cosine_similarity(features, saved_features)
            if similarity >= 0.952:
                return jsonify({'image_id': image_id})

        new_image_id = str(random.randint(10, 99))
        save_features(features, new_image_id)

        return jsonify({'image_id': new_image_id,'similarity': similarity})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
