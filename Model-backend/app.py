import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
import h5py
import shutil

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Initialize Flask app
app = Flask(__name__, static_folder="../Frontend", static_url_path="")
CORS(app)  # Enable CORS for all routes

# Get the directory of the current script (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script's directory
MODEL_ARCHITECTURE_PATH = os.path.join(BASE_DIR, "model_architecture.json")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.weights.h5")
MODEL_WEIGHTS_RENAMED_PATH = os.path.join(BASE_DIR, "model_weights_renamed.weights.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Check if required files exist
required_files = {
    MODEL_ARCHITECTURE_PATH: "Model architecture",
    MODEL_WEIGHTS_PATH: "Model weights",
    TOKENIZER_PATH: "Tokenizer",
    LABEL_ENCODER_PATH: "Label encoder"
}

for file, desc in required_files.items():
    if not os.path.exists(file):
        print(f"Error: {desc} file '{file}' not found.")
        exit(1)

# Load the TensorFlow model architecture
try:
    with open(MODEL_ARCHITECTURE_PATH, "r") as json_file:
        model = model_from_json(json_file.read())
    print("Model architecture loaded successfully.")
    model.summary()  # Print model summary
    embedding_layer = model.layers[0]
    print(f"Expected embedding weights shape: {(embedding_layer.input_dim, embedding_layer.output_dim)}")
except Exception as e:
    print(f"Error loading model architecture: {str(e)}")
    exit(1)

# Inspect the weights file
print("Inspecting original weights file...")
try:
    with h5py.File(MODEL_WEIGHTS_PATH, "r") as f:
        def inspect_hdf5_group(group, prefix=""):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"{prefix}{key}: Shape: {item.shape}")
                elif isinstance(item, h5py.Group):
                    inspect_hdf5_group(item, prefix=f"{prefix}{key}/")
        print("Weights file top-level keys:", list(f.keys()))
        inspect_hdf5_group(f)
except Exception as e:
    print(f"Error inspecting original weights file: {str(e)}")
    exit(1)

# Rename the weights to match Keras naming conventions
print("Renaming weights in the HDF5 file...")
try:
    # Copy the original weights file to a new file
    shutil.copy(MODEL_WEIGHTS_PATH, MODEL_WEIGHTS_RENAMED_PATH)
    # Open the new file in read-write mode
    with h5py.File(MODEL_WEIGHTS_RENAMED_PATH, "r+") as f_new:
        # Inspect the file before renaming
        print("Inspecting weights file before renaming...")
        inspect_hdf5_group(f_new)
        # Create new groups with the expected Keras naming
        # Embedding layer
        if 'embedding' not in f_new:
            f_new.create_group('embedding')
        if 'embeddings' not in f_new['embedding']:
            f_new['embedding'].create_group('embeddings')
        embedding_weights = np.array(f_new['layers']['embedding']['vars']['0'])
        if 'embeddings:0' in f_new['embedding']['embeddings']:
            del f_new['embedding']['embeddings']['embeddings:0']
        f_new['embedding']['embeddings']['embeddings:0'] = embedding_weights
        print("Renamed embedding weights to 'embedding/embeddings:0'.")
        # Dense layer
        if 'dense' not in f_new:
            f_new.create_group('dense')
        dense_weights = np.array(f_new['layers']['dense']['vars']['0'])
        dense_bias = np.array(f_new['layers']['dense']['vars']['1'])
        if 'kernel:0' in f_new['dense']:
            del f_new['dense']['kernel:0']
        if 'bias:0' in f_new['dense']:
            del f_new['dense']['bias:0']
        f_new['dense']['kernel:0'] = dense_weights
        f_new['dense']['bias:0'] = dense_bias
        print("Renamed dense weights to 'dense/kernel:0' and 'dense/bias:0'.")
        # Dense_1 layer
        if 'dense_1' not in f_new:
            f_new.create_group('dense_1')
        dense_1_weights = np.array(f_new['layers']['dense_1']['vars']['0'])
        dense_1_bias = np.array(f_new['layers']['dense_1']['vars']['1'])
        if 'kernel:0' in f_new['dense_1']:
            del f_new['dense_1']['kernel:0']
        if 'bias:0' in f_new['dense_1']:
            del f_new['dense_1']['bias:0']
        f_new['dense_1']['kernel:0'] = dense_1_weights
        f_new['dense_1']['bias:0'] = dense_1_bias
        print("Renamed dense_1 weights to 'dense_1/kernel:0' and 'dense_1/bias:0'.")
        # Inspect the file after renaming
        print("Inspecting weights file after renaming...")
        inspect_hdf5_group(f_new)
except Exception as e:
    print(f"Error renaming weights: {str(e)}")
    exit(1)

# Load the model weights from the renamed file
try:
    model.load_weights(MODEL_WEIGHTS_RENAMED_PATH)
    print("Model weights loaded successfully from renamed file.")
except Exception as e:
    print(f"Error loading renamed model weights: {str(e)}")
    # Attempt to load weights by layer name as a fallback
    try:
        print("Attempting to load weights by layer name as a final fallback...")
        with h5py.File(MODEL_WEIGHTS_PATH, "r") as f:
            # Load embedding weights
            embedding_weights = np.array(f['layers']['embedding']['vars']['0'])
            print(f"Loaded embedding weights shape: {embedding_weights.shape}")
            embedding_layer.set_weights([embedding_weights])
            print("Manually set embedding weights.")
            # Load dense layer weights
            dense_weights = np.array(f['layers']['dense']['vars']['0'])
            dense_bias = np.array(f['layers']['dense']['vars']['1'])
            model.layers[2].set_weights([dense_weights, dense_bias])
            print("Manually set dense layer weights.")
            # Load dense_1 layer weights
            dense_1_weights = np.array(f['layers']['dense_1']['vars']['0'])
            dense_1_bias = np.array(f['layers']['dense_1']['vars']['1'])
            model.layers[3].set_weights([dense_1_weights, dense_1_bias])
            print("Manually set dense_1 layer weights.")
    except Exception as e2:
        print(f"Error loading weights by layer name as final fallback: {str(e2)}")
        exit(1)

# Load the tokenizer and label encoder
try:
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
    print("Tokenizer loaded successfully.")
    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)
    print("Label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer or label encoder: {str(e)}")
    exit(1)

# Set MAX_LENGTH to match the trained model's expectation
MAX_LENGTH = 66  # Matches the calculated max_length from model_train.py
print(f"Using MAX_LENGTH: {MAX_LENGTH}")

# Function to predict emotion
def predict_emotion(text):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH)
        prediction = model.predict(padded_sequence)
        emotion_idx = np.argmax(prediction, axis=1)[0]
        emotion = label_encoder.inverse_transform([emotion_idx])[0]
        return emotion
    except Exception as e:
        raise Exception(f"Emotion prediction failed: {str(e)}")

# Function to generate advice based on detected emotion
def get_advice(emotion):
    advice_dict = {
        "sadness": "Take a deep breath and try to talk to someone you trust. Small steps like a walk or listening to music might help lift your mood.",
        "anger": "Take a moment to cool off. Try deep breathing or writing down your thoughts to process your feelings.",
        "joy": "Enjoy this positivity! Share it with others or do something creative to keep the good vibes going.",
        "love": "Cherish this feeling! Express it to those who matter or spend time nurturing your relationships.",
        "surprise": "Embrace the unexpected! Reflect on what surprised you and see if it opens new opportunities.",
        "fear": "It’s okay to feel this way. Ground yourself with a calm activity like meditation or talking it out."
    }
    return advice_dict.get(emotion, "Take some time to relax and reflect on your feelings.")

# Serve the frontend (index.html)
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "userInput" not in data:
            return jsonify({"success": False, "error": "No input provided"}), 400

        input_text = data["userInput"].strip()
        if not input_text:
            return jsonify({"success": False, "error": "Input is empty"}), 400

        # Detect emotion
        detected_emotion = predict_emotion(input_text)
        # Get advice
        advice_response = get_advice(detected_emotion)

        return jsonify({
            "success": True,
            "detectedEmotion": detected_emotion,
            "adviceResponse": advice_response
        })

    except Exception as e:
        print("Error in /predict endpoint:")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 3000))
        print(f"Starting Flask server on port {port}...")
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Error starting Flask server: {str(e)}")
        exit(1)