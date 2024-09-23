# Heart Sound Classification with CNN + LSTM Hybrid Model

## Project Overview

This project focuses on classifying heart sounds using a hybrid CNN + LSTM model. The aim is to identify various heart conditions (such as Mitral Stenosis, Mitral Regurgitation, Mitral Valve Prolapse, Aortic Stenosis, and Normal) from heart sound recordings. 

### Key Features:
- **Data Preprocessing:** Audio features such as MFCC, Chroma Feature, Zero-Crossing Rate, Spectral Centroid, Spectral Contrast, Delta and Delta-Delta Coefficients, and Mel-Spectrogram with different window sizes are extracted.
- **Model Architecture:** A hybrid of CNN and LSTM is employed to capture both spatial and temporal patterns in the heart sound data.
- **Training Strategy:** Keras Tuner is used for hyperparameter tuning to optimize the model's performance.
- **Implementation:** The model is built from scratch using TensorFlow/Keras.
