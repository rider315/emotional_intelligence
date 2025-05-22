# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import traceback

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Check if required files exist
# required_files = {
#     "model_architecture.json": "Model architecture",
#     "model_weights.weights.h5": "Model weights",
#     "tokenizer.pkl": "Tokenizer",
#     "label_encoder.pkl": "Label encoder"
# }

# for file, desc in required_files.items():
#     if not os.path.exists(file):
#         print(f"Error: {desc} file '{file}' not found.")
#         exit(1)

# # Load the TensorFlow model architecture
# try:
#     with open("model_architecture.json", "r") as json_file:
#         model = model_from_json(json_file.read())
#     print("Model architecture loaded successfully.")
# except Exception as e:
#     print(f"Error loading model architecture: {str(e)}")
#     exit(1)

# # Load the model weights
# try:
#     model.load_weights("model_weights.weights.h5")
#     print("Model weights loaded successfully.")
# except Exception as e:
#     print(f"Error loading model weights: {str(e)}")
#     exit(1)

# # Load the tokenizer and label encoder
# try:
#     with open("tokenizer.pkl", "rb") as file:
#         tokenizer = pickle.load(file)
#     print("Tokenizer loaded successfully.")
#     with open("label_encoder.pkl", "rb") as file:
#         label_encoder = pickle.load(file)
#     print("Label encoder loaded successfully.")
# except Exception as e:
#     print(f"Error loading tokenizer or label encoder: {str(e)}")
#     exit(1)

# # Set MAX_LENGTH to match the trained model's expectation
# MAX_LENGTH = 66  # Matches the calculated max_length from model_train.py
# print(f"Using MAX_LENGTH: {MAX_LENGTH}")

# # Function to predict emotion
# def predict_emotion(text):
#     try:
#         sequence = tokenizer.texts_to_sequences([text])
#         padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH)
#         prediction = model.predict(padded_sequence)
#         emotion_idx = np.argmax(prediction, axis=1)[0]
#         emotion = label_encoder.inverse_transform([emotion_idx])[0]
#         return emotion
#     except Exception as e:
#         raise Exception(f"Emotion prediction failed: {str(e)}")

# # Function to generate advice based on detected emotion
# def get_advice(emotion):
#     advice_dict = {
#         "sadness": "Take a deep breath and try to talk to someone you trust. Small steps like a walk or listening to music might help lift your mood.",
#         "anger": "Take a moment to cool off. Try deep breathing or writing down your thoughts to process your feelings.",
#         "joy": "Enjoy this positivity! Share it with others or do something creative to keep the good vibes going.",
#         "love": "Cherish this feeling! Express it to those who matter or spend time nurturing your relationships.",
#         "surprise": "Embrace the unexpected! Reflect on what surprised you and see if it opens new opportunities.",
#         "fear": "It’s okay to feel this way. Ground yourself with a calm activity like meditation or talking it out."
#     }
#     return advice_dict.get(emotion, "Take some time to relax and reflect on your feelings.")

# # API endpoint for prediction
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         if not data or "userInput" not in data:
#             return jsonify({"success": False, "error": "No input provided"}), 400

#         input_text = data["userInput"].strip()
#         if not input_text:
#             return jsonify({"success": False, "error": "Input is empty"}), 400

#         # Detect emotion
#         detected_emotion = predict_emotion(input_text)
#         # Get advice
#         advice_response = get_advice(detected_emotion)

#         return jsonify({
#             "success": True,
#             "detectedEmotion": detected_emotion,
#             "adviceResponse": advice_response
#         })

#     except Exception as e:
#         print("Error in /predict endpoint:")
#         print(traceback.format_exc())
#         return jsonify({"success": False, "error": str(e)}), 500

# if __name__ == "__main__":
#     try:
#         app.run(debug=True, port=3000)
#     except Exception as e:
#         print(f"Error starting Flask server: {str(e)}")
#         exit(1)

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback

# Initialize Flask app
app = Flask(__name__, static_folder="../Frontend", static_url_path="")
CORS(app)  # Enable CORS for all routes

# Check if required files exist
required_files = {
    "Model-backend/model_architecture.json": "Model architecture",
    "Model-backend/model_weights.weights.h5": "Model weights",
    "Model-backend/tokenizer.pkl": "Tokenizer",
    "Model-backend/label_encoder.pkl": "Label encoder"
}

for file, desc in required_files.items():
    if not os.path.exists(file):
        print(f"Error: {desc} file '{file}' not found.")
        exit(1)

# Load the TensorFlow model architecture
try:
    with open("Model-backend/model_architecture.json", "r") as json_file:
        model = model_from_json(json_file.read())
    print("Model architecture loaded successfully.")
except Exception as e:
    print(f"Error loading model architecture: {str(e)}")
    exit(1)

# Load the model weights
try:
    model.load_weights("Model-backend/model_weights.weights.h5")
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {str(e)}")
    exit(1)

# Load the tokenizer and label encoder
try:
    with open("Model-backend/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    print("Tokenizer loaded successfully.")
    with open("Model-backend/label_encoder.pkl", "rb") as file:
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
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Error starting Flask server: {str(e)}")
        exit(1)