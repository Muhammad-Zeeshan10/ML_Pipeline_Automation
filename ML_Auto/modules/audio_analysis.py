import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import io
import zipfile
import tempfile

# Emotion mapping for RAVDESS dataset
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_emotion_from_filename(filename):
    """
    Extract the emotion label from RAVDESS filename format
    Format: xx-xx-xx-xx-xx-xx-xx.wav
    Emotion is the third element (index 2)
    """
    parts = os.path.basename(filename).split('-')
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, 'unknown')

def extract_features(data):
    """Extract audio features from the audio signal"""
    result = np.array([])
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    
    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=22050).T, axis=0)
    result = np.hstack((result, chroma_stft))
    
    # MFCC (Mel-Frequency Cepstral Coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=22050).T, axis=0)
    result = np.hstack((result, mfcc))
    
    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=22050).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

def get_features_with_augmentation(path):
    """Extract features with data augmentation"""
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Original data
    res1 = extract_features(data)
    result = np.array(res1)
    
    # Data with noise
    noise_data = noise_injection(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))
    
    # Data with time stretching and pitch shifting
    new_data = time_stretch(data)
    data_stretch_pitch = pitch_shift(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))
    
    return result

def noise_injection(data):
    """Add random noise to audio signal"""
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def time_stretch(data, rate=0.8):
    """Change the speed of audio"""
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sample_rate, pitch_factor=0.7):
    """Shift the pitch of audio"""
    return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_factor)

def plot_waveform(data, sample_rate, title="Waveform"):
    """Plot audio waveform"""
    try:
        plt.figure(figsize=(10, 4))
        plt.title(title)
        
        # Some versions of librosa use waveshow, others use waveplot
        try:
            librosa.display.waveshow(data, sr=sample_rate)
        except AttributeError:
            librosa.display.waveplot(data, sr=sample_rate)
            
        return plt.gcf()
    except Exception as e:
        # Create a figure with an error message if plotting fails
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f'Unable to display waveform: {str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        plt.title(f'{title} (Error)')
        return fig

def plot_spectrogram(data, sample_rate, title="Spectrogram"):
    """Plot audio spectrogram"""
    plt.figure(figsize=(10, 4))
    plt.title(title)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # Create a mappable object and store it before adding colorbar
    img = librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
    
    # Only add colorbar if the specshow was successful
    if img is not None:
        plt.colorbar(img, format='%+2.0f dB')
    else:
        # If specshow failed, add a text note to the plot
        plt.text(0.5, 0.5, 'Unable to display spectrogram', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    return plt.gcf()

def prepare_data_from_directory(data_dir):
    """Prepare features and labels from a directory of audio files"""
    features = []
    labels = []
    file_paths = []
    error_messages = []
    
    try:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    emotion = extract_emotion_from_filename(file_path)
                    
                    # Skip unknown emotions
                    if emotion == 'unknown':
                        continue
                        
                    try:
                        # Extract features with augmentation
                        feature_set = get_features_with_augmentation(file_path)
                        for feature in feature_set:
                            features.append(feature)
                            labels.append(emotion)
                            file_paths.append(file_path)
                    except Exception as e:
                        error_msg = f"Error processing {file_path}: {str(e)}"
                        print(error_msg)
                        error_messages.append(error_msg)
                        if "No module named '_lzma'" in str(e):
                            error_messages.append(
                                "The '_lzma' module is missing. Try installing it with 'pip install -U lzma' "
                                "or 'brew install xz' on macOS, then install Python with lzma support."
                            )
    except Exception as e:
        print(f"Error scanning audio directory: {str(e)}")
        error_messages.append(f"Error scanning audio directory: {str(e)}")
    
    # If no features were extracted, return None with error messages
    if not features:
        if error_messages:
            raise ValueError(f"No features could be extracted. Errors: {'; '.join(error_messages)}")
        return None
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    feature_df['emotion'] = labels
    feature_df['file_path'] = file_paths
    
    return feature_df

def build_model(input_shape, num_classes):
    """Build a CNN model for emotion classification"""
    model = Sequential([
        Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        
        Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        
        Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Dropout(0.2),
        
        Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_audio_model(features_df, test_size=0.2, epochs=50):
    """Train the audio emotion recognition model"""
    # Check if dataframe is empty
    if features_df is None or len(features_df) == 0:
        raise ValueError("No features available for training. The feature dataframe is empty.")
    
    # Check if we have enough data for a meaningful split
    if len(features_df) < 10:
        raise ValueError(f"Not enough data for training. Found only {len(features_df)} samples, need at least 10.")
        
    try:
        # Separate features and labels
        X = features_df.iloc[:, :-2].values  # All columns except emotion and file_path
        y = features_df['emotion'].values
        
        # Encode labels
        encoder = OneHotEncoder()
        y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Reshape for CNN
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        
        # Build model
        model = build_model(input_shape=(X_train.shape[1], 1), num_classes=y_encoded.shape[1])
        
        # Create custom callback to track metrics per epoch
        class MetricsTracker(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.epoch_metrics = []
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                metrics = {
                    'epoch': epoch + 1,
                    'accuracy': logs.get('accuracy', 0),
                    'loss': logs.get('loss', 0),
                    'val_accuracy': logs.get('val_accuracy', 0),
                    'val_loss': logs.get('val_loss', 0),
                    'learning_rate': float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                }
                self.epoch_metrics.append(metrics)
                
                # Print metrics for each epoch (will be visible in terminal)
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  accuracy: {metrics['accuracy']:.4f}, loss: {metrics['loss']:.4f}", end='')
                print(f", val_accuracy: {metrics['val_accuracy']:.4f}, val_loss: {metrics['val_loss']:.4f}")
        
        # Initialize the metrics tracker
        metrics_tracker = MetricsTracker()
        
        # Train model with learning rate reduction and metrics tracking
        rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
        
        history = model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[rlrp, metrics_tracker]
        )
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Convert indices back to emotion labels
        emotion_labels = encoder.categories_[0]
        
        return {
            'model': model,
            'history': history,
            'accuracy': test_accuracy,
            'loss': test_loss,
            'y_pred': y_pred,
            'y_test': y_test,
            'scaler': scaler,
            'encoder': encoder,
            'emotion_labels': emotion_labels,
            'y_pred_classes': y_pred_classes,
            'y_true_classes': y_true_classes,
            'epoch_metrics': metrics_tracker.epoch_metrics  # Add epoch metrics to the result
        }
    except Exception as e:
        raise RuntimeError(f"Error during model training: {str(e)}")

def save_trained_model(model_results):
    """Save the trained model and preprocessing objects"""
    model = model_results['model']
    scaler = model_results['scaler']
    encoder = model_results['encoder']
    
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # Save the model to a temporary file with .keras extension
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
            temp_model_path = temp_model_file.name
        
        # Save the model to the temporary file
        model.save(temp_model_path)
        
        # Read the saved model file and add it to the zip
        with open(temp_model_path, 'rb') as f:
            zip_file.writestr('emotion_model.keras', f.read())
        
        # Clean up the temporary file
        os.unlink(temp_model_path)
        
        # Save and add the scaler
        scaler_buffer = io.BytesIO()
        np.save(scaler_buffer, scaler, allow_pickle=True)
        scaler_buffer.seek(0)
        zip_file.writestr('scaler.npy', scaler_buffer.read())
        
        # Add the emotion labels mapping
        emotion_labels = encoder.categories_[0].tolist()
        zip_file.writestr('emotion_labels.txt', '\n'.join(emotion_labels))
    
    zip_buffer.seek(0)
    return zip_buffer

def predict_emotion_from_audio(audio_file, model, scaler, encoder):
    """Predict emotion from a new audio file"""
    try:
        data, sample_rate = librosa.load(audio_file, duration=2.5, offset=0.6)
        features = extract_features(data)
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Reshape for CNN
        features_reshaped = np.expand_dims(features_scaled, axis=2)
        
        # Predict
        prediction = model.predict(features_reshaped)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Get emotion label
        emotion_labels = encoder.categories_[0]
        predicted_emotion = emotion_labels[predicted_class]
        
        # Get confidence
        confidence = prediction[0][predicted_class]
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'all_probabilities': dict(zip(emotion_labels, prediction[0]))
        }
    except Exception as e:
        return {'error': str(e)} 