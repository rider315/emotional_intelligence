import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Initialize Flask app
app = Flask(__name__, static_folder="../Frontend", static_url_path="")
CORS(app)  # Enable CORS for all routes

# Get the directory of the current script (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script's directory
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Check if required files exist
required_files = {
    MODEL_PATH: "Model directory",
    TOKENIZER_PATH: "Tokenizer",
    LABEL_ENCODER_PATH: "Label encoder"
}

for file, desc in required_files.items():
    if not os.path.exists(file):
        print(f"Error: {desc} file '{file}' not found.")
        exit(1)

# Load the TensorFlow SavedModel
try:
    model = tf.saved_model.load(MODEL_PATH)
    print("Model loaded successfully from SavedModel format.")
    # The SavedModel contains the inference function, typically under the 'serving_default' signature
    infer = model.signatures["serving_default"]
    print("Inference function loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
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
        # Tokenize and pad the input text
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH)
        # Convert to tensor and ensure the correct dtype
        input_tensor = tf.convert_to_tensor(padded_sequence, dtype=tf.int32)
        # Run inference using the SavedModel's serving_default signature
        prediction = infer(input_tensor)['dense_1']  # Adjust the output key based on your model's output layer name
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
            return jsonify