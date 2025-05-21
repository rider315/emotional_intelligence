import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
import pickle

# Load data
data = pd.read_csv("../train.txt", sep=';', header=None, names=['text', 'label'])

# Prepare data
texts = data['text'].values
labels = data['label'].values

# Tokenize the text
MAX_WORDS = 50000  # Adjust based on your vocabulary size
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Calculate max_length and pad sequences
max_length = max([len(seq) for seq in sequences])
print(f"Calculated max_length: {max_length}")  # Print max_length for app.py
sequences = pad_sequences(sequences, maxlen=max_length)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(sequences, categorical_labels, test_size=0.2, random_state=42)

# Build the model
embedding_dim = 128
model = Sequential([
    Embedding(MAX_WORDS + 1, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model architecture and weights
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_weights.weights.h5")

# Save the tokenizer and label encoder
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)