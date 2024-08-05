from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="cow_muzzle_feature_extractor.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image
    return img_array

# Extract features from an image using the TFLite model
def extract_features(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Calculate cosine similarity
def cosine_similarity(features1, features2):
    dot_product = np.dot(features1, features2.T)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    return dot_product / (norm1 * norm2)

# Define similarity endpoint
@app.route('/similarity', methods=['POST'])
def similarity():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Please provide two image files'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Please provide two valid image files'}), 400

    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))

        img1_array = preprocess_image(img1)
        img2_array = preprocess_image(img2)

        features1 = extract_features(img1_array)
        features2 = extract_features(img2_array)

        similarity_score = cosine_similarity(features1, features2)

        return jsonify({'similarity_score': float(similarity_score)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
