from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import os
import tempfile

app = Flask(__name__)

# Load the model
model = load_model('gender_emotion_classifier_model.h5')

def extract_mfcc(file_path, n_mfcc=13, sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Ensure correct shape
        expected_shape = model.input_shape[1]  # Get expected input size
        if len(mfccs_mean) != expected_shape:
            print(f"Error: Expected {expected_shape} features, but got {len(mfccs_mean)}")
            return None
        
        return mfccs_mean
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Define emotion labels based on your dataset
emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised", "calm"]

def predict_gender_emotion(audio_file):
    mfccs = extract_mfcc(audio_file)
    if mfccs is not None:
        input_data = np.expand_dims(mfccs, axis=0)
        prediction = model.predict(input_data)

        if len(prediction) != 2:
            print(f"Unexpected model output shape: {len(prediction)}")
            return None, None

        gender_pred = int(np.argmax(prediction[0]))  # 0: Male, 1: Female
        emotion_pred = int(np.argmax(prediction[1]))

        # Ensure the predicted emotion index is within range
        if 0 <= emotion_pred < len(emotion_labels):
            emotion_label = emotion_labels[emotion_pred]
        else:
            emotion_label = "unknown"

        return gender_pred, emotion_label
    return None, None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        file.save(temp_audio.name)
        temp_path = temp_audio.name

    gender, emotion = predict_gender_emotion(temp_path)

    # Remove temp file
    os.remove(temp_path)

    if gender is not None and emotion is not None:
        return jsonify({'gender': 'male' if gender == 0 else 'female', 'emotion': emotion})
    return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)


