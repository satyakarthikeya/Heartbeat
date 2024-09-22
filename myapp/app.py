from flask import Flask, request, render_template
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your trained model and scaler
model = load_model(r'C:\Users\pasar\Desktop\Sem_3\msa\Heartbeat\best_heart_sound_model.h5')
  # if you saved your scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    y, sr = librosa.load(file)
    features = extract_features(y, sr)  # Use your feature extraction function
    features_scaled = scaler.transform(features.reshape(1, -1))  # Scale the features
    prediction = model.predict(features_scaled)
    return f'Predicted class: {np.argmax(prediction)}'

if __name__ == '__main__':
    app.run(debug=True)
