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
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Ensure RGB
    img = img.resize((1, 1))  # Adjust to model's expected size
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    print("Image shape after preprocessing:", img_array.shape)  # Check shape

    return img_array

# Extract features from an image using the TFLite model
# Extract features from an image using the TFLite model
def extract_features(img_array):
    # Ensure that img_array is in the correct shape
    img_array = np.array(img_array, dtype=np.float32)  # Ensure the type is float32
    if not np.array_equal(img_array.shape, input_details[0]['shape']):
        raise ValueError(f"Input shape {img_array.shape} does not match model's expected shape {input_details[0]['shape']}")
    
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
