from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
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
    img = image.load_img(img_path, target_size=(input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
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

    # Define temporary paths for saving files
    temp_path1 = os.path.join('temp_dir', file1.filename)
    temp_path2 = os.path.join('temp_dir', file2.filename)
    
    # Ensure the temp directory exists
    if not os.path.exists('temp_dir'):
        os.makedirs('temp_dir')

    try:
        # Save the files temporarily
        file1.save(temp_path1)
        file2.save(temp_path2)

        # Preprocess the images from the saved file paths
        img1_array = preprocess_image(temp_path1)
        img2_array = preprocess_image(temp_path2)

        # Extract features
        features1 = extract_features(img1_array)
        features2 = extract_features(img2_array)

        # Calculate similarity score
        similarity_score = cosine_similarity(features1, features2)

        return jsonify({'similarity_score': float(similarity_score)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the temporary files
        try:
            os.remove(temp_path1)
            os.remove(temp_path2)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

if __name__ == '__main__':
    app.run(debug=True)
