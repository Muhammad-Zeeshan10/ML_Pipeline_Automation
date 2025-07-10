import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import librosa
import librosa.display
from modules.data_loader import load_data, get_data_preview
from modules.preprocessing import (
    detect_missing_values,
    handle_missing_values,
    encode_categorical_features,
    scale_features,
    detect_outliers,
    remove_outliers,
)
from modules.model_utils import (
    detect_task_type,
    get_model_suggestions,
    train_model,
    save_model,
    get_feature_importance
)
from modules.visualization import (
    plot_correlation_matrix,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_actual_vs_predicted,
    plot_clustering
)
# Audio analysis imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
import io
import zipfile

# Import audio analysis functions
from modules.audio_analysis import (
    extract_features,
    plot_waveform,
    plot_spectrogram,
    prepare_data_from_directory,
    train_audio_model,
    save_trained_model,
    predict_emotion_from_audio
)

# Set page configuration
st.set_page_config(
    page_title="ML Workflow Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'data_types' not in st.session_state:
    st.session_state.data_types = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Initialize app mode
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = "1. Upload Data"
if 'audio_step' not in st.session_state:
    st.session_state.audio_step = None
if 'audio_features' not in st.session_state:
    st.session_state.audio_features = None
if 'audio_model_results' not in st.session_state:
    st.session_state.audio_model_results = None

# App title
st.title("Machine Learning Assistant")
st.markdown("---")

# Mode selection (ML Workflow or Audio Analysis)
if st.session_state.app_mode is None:
    st.header("Select Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### ML Workflow")
        st.markdown("""
        - Upload CSV/Excel data
        - Explore and preprocess your data
        - Train and evaluate ML models
        - Download trained models
        """)
        if st.button("📊 ML Workflow", use_container_width=True):
            st.session_state.app_mode = "ml_workflow"
            st.session_state.current_step = "1. Upload Data"
            st.rerun()
    
    with col2:
        st.info("### Audio Analysis")
        st.markdown("""
        - Analyze emotional speech audio
        - Visualize audio features
        - Train emotion recognition models
        - Test with your own audio files
        """)
        if st.button("🎵 Audio Analysis", use_container_width=True):
            st.session_state.app_mode = "audio_analysis"
            st.session_state.audio_step = "1. Select Data Source"
            st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application provides tools for both general machine learning workflows and specialized audio analysis.
    
    Choose the mode that fits your needs:
    - **ML Workflow**: For tabular data analysis and model training
    - **Audio Analysis**: For speech emotion recognition using audio files
    """)

# Handle audio analysis mode
elif st.session_state.app_mode == "audio_analysis":
    # Create sidebar to show current workflow step for audio analysis
    st.sidebar.title("Audio Analysis Steps")
    # Initialize current step in session state if not present
    if st.session_state.audio_step is None:
        st.session_state.audio_step = "1. Select Data Source"

    # Display the current step using markdown with visual cues
    st.sidebar.markdown("### Current Step:")
    for step in ["1. Select Data Source", "2. Explore & Visualize", "3. Feature Extraction", "4. Model Training", "5. Evaluation & Testing"]:
        if step == st.session_state.audio_step:
            st.sidebar.markdown(f"**→ {step}** 🔵")
        else:
            st.sidebar.markdown(f"  {step}")
    
    # Add a button to switch back to mode selection
    if st.sidebar.button("← Back to Mode Selection"):
        st.session_state.app_mode = None
        st.rerun()
    
    st.title("Audio Emotion Analysis")
    st.markdown("---")
    
    # 1. Select Data Source
    if st.session_state.audio_step == "1. Select Data Source":
        st.header("🎵 Select Audio Data Source")
        
        with st.expander("About Speech Emotion Recognition", expanded=True):
            st.markdown("""
            Speech Emotion Recognition (SER) is the process of identifying human emotions from voice. This system analyzes:
            - Tone variations
            - Pitch changes
            - Energy levels
            - Spectral features
            
            The RAVDESS dataset contains emotional speech audio files with 8 emotions:
            - Neutral
            - Calm
            - Happy
            - Sad
            - Angry
            - Fearful
            - Disgust
            - Surprised
            """)
        
        data_source = st.radio(
            "Choose your data source:",
            ["Use RAVDESS Dataset", "Upload your own audio files"],
            index=0
        )
        
        if data_source == "Use RAVDESS Dataset":
            st.info("Using the RAVDESS emotional speech dataset")
            
            if os.path.exists("./archive"):
                st.success("✅ RAVDESS dataset found in the archive directory")
                
                # Count available audio files 
                num_files = 0
                for root, _, files in os.walk("./archive"):
                    for file in files:
                        if file.endswith(".wav"):
                            num_files += 1
                
                if num_files > 0:
                    st.write(f"Found {num_files} audio files in the archive directory")
                    
                    # Display a few sample emotions from files
                    st.markdown("### Sample emotions in dataset:")
                    emotions_found = set()
                    for root, _, files in os.walk("./archive"):
                        for file in files:
                            if file.endswith(".wav") and len(emotions_found) < 8:
                                try:
                                    parts = file.split('-')
                                    if len(parts) >= 3:
                                        emotion_code = parts[2]
                                        emotion_map = {
                                            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                                            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
                                        }
                                        if emotion_code in emotion_map:
                                            emotions_found.add(emotion_map[emotion_code])
                                except:
                                    pass
                    
                    if emotions_found:
                        st.write(", ".join(emotions_found))
                    
                    if st.button("✅ Continue with RAVDESS Dataset", use_container_width=True):
                        st.session_state.ravdess_path = "./archive"
                        st.session_state.audio_step = "2. Explore & Visualize"
                        st.rerun()
                else:
                    st.warning("No .wav files found in the archive directory")
            else:
                st.warning("RAVDESS dataset not found. Please upload your audio files instead.")
        
        else:
            st.info("Please upload your audio files (WAV format)")
            uploaded_files = st.file_uploader("Upload audio files", type=["wav"], accept_multiple_files=True)
            
            if uploaded_files:
                # Save uploaded files to a temporary directory
                temp_dir = "./uploaded_audio"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ Uploaded {len(uploaded_files)} audio files")
                
                if st.button("✅ Continue with Uploaded Files", use_container_width=True):
                    st.session_state.ravdess_path = temp_dir
                    st.session_state.audio_step = "2. Explore & Visualize"
                    st.rerun()
    
    # 2. Explore & Visualize
    elif st.session_state.audio_step == "2. Explore & Visualize":
        st.header("🔍 Explore & Visualize Audio")
        
        if not hasattr(st.session_state, 'ravdess_path'):
            st.warning("Please select a data source first")
            st.session_state.audio_step = "1. Select Data Source"
            st.rerun()
        
        with st.expander("About Audio Visualization", expanded=True):
            st.markdown("""
            Audio visualization helps understand the characteristics of sound:
            
            - **Waveform**: Shows amplitude (loudness) changes over time
            - **Spectrogram**: Displays frequency content over time
            - **MFCC**: Mel-Frequency Cepstral Coefficients - important features for speech analysis
            
            Different emotions have distinct patterns in these visualizations.
            """)
        
        # Find audio files
        audio_files = []
        for root, _, files in os.walk(st.session_state.ravdess_path):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))
        
        if audio_files:
            # Select an audio file to visualize
            selected_file = st.selectbox("Select an audio file to visualize:", audio_files)
            
            if selected_file:
                try:
                    # Load audio file
                    data, sample_rate = librosa.load(selected_file)
                    
                    # Extract emotion if possible
                    try:
                        parts = os.path.basename(selected_file).split('-')
                        emotion_code = parts[2]
                        emotion_map = {
                            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
                        }
                        emotion = emotion_map.get(emotion_code, "Unknown")
                        st.markdown(f"### Emotion: {emotion.capitalize()}")
                    except:
                        st.markdown("### Audio File")
                    
                    # Allow playback
                    st.audio(selected_file)
                    
                    # Show waveform
                    st.subheader("Waveform")
                    try:
                        fig_waveform = plot_waveform(data, sample_rate, "Audio Waveform")
                        st.pyplot(fig_waveform)
                    except Exception as e:
                        st.error(f"Error displaying waveform: {e}")
                    
                    # Show spectrogram
                    st.subheader("Spectrogram")
                    try:
                        fig_spectrogram = plot_spectrogram(data, sample_rate, "Audio Spectrogram")
                        st.pyplot(fig_spectrogram)
                    except Exception as e:
                        st.error(f"Error displaying spectrogram: {e}")
                    
                except Exception as e:
                    st.error(f"Error processing audio file: {e}")
        
            # Continue button
            if st.button("✅ Continue to Feature Extraction", use_container_width=True):
                st.session_state.audio_step = "3. Feature Extraction"
                st.rerun()
        else:
            st.error("No audio files found in the selected directory")
            if st.button("← Back to Data Source Selection", use_container_width=True):
                st.session_state.audio_step = "1. Select Data Source"
                st.rerun()
    
    # 3. Feature Extraction
    elif st.session_state.audio_step == "3. Feature Extraction":
        st.header("⚙️ Audio Feature Extraction")
        
        if not hasattr(st.session_state, 'ravdess_path'):
            st.warning("Please select a data source first")
            st.session_state.audio_step = "1. Select Data Source"
            st.rerun()
        
        with st.expander("About Feature Extraction", expanded=True):
            st.markdown("""
            For speech emotion recognition, we extract several key features:
            
            1. **Zero Crossing Rate**: Rate at which the signal changes from positive to negative
            2. **Chroma Features**: Representation of spectral energy distribution
            3. **RMS Energy**: Root Mean Square energy of the signal
            4. **Mel Spectrogram**: Spectrogram where frequencies are converted to the mel scale
            
            We'll also apply data augmentation to improve model performance:
            - Adding noise
            - Changing pitch
            - Time stretching
            """)
        
        if st.session_state.audio_features is None:
            if st.button("Start Feature Extraction Process", use_container_width=True):
                with st.spinner("Extracting features from audio files... This may take several minutes."):
                    try:
                        # Extract features from all audio files
                        features_df = prepare_data_from_directory(st.session_state.ravdess_path)
                        
                        if features_df is None or len(features_df) == 0:
                            st.error("⚠️ No features were extracted. Please check if your audio files are in the correct format and that you have the required dependencies installed.")
                            st.info("If you're seeing an error about '_lzma' module, you need to install it with:\n\n"
                                  "```\npip install -U lzma\n```\n\n"
                                  "Or on macOS:\n\n"
                                  "```\nbrew install xz\n```\n\n"
                                  "Then reinstall Python with lzma support.")
                        else:
                            st.session_state.audio_features = features_df
                            st.success(f"✅ Successfully extracted features from {len(features_df['file_path'].unique())} audio files")
                            
                            # Count emotions
                            emotion_counts = features_df['emotion'].value_counts()
                            
                            if not emotion_counts.empty:
                                # Display emotion distribution
                                st.subheader("Emotion Distribution in Dataset")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                emotion_counts.plot(kind='bar', ax=ax)
                                plt.title('Count of Emotions')
                                plt.xlabel('Emotions')
                                plt.ylabel('Count')
                                st.pyplot(fig)
                            else:
                                st.warning("No emotion data to display.")
                            
                            # Continue button appears after processing
                            if st.button("✅ Continue to Model Training", key="after_extraction", use_container_width=True):
                                st.session_state.audio_step = "4. Model Training"
                                st.rerun()
                            
                    except Exception as e:
                        error_message = str(e)
                        st.error(f"Error during feature extraction: {error_message}")
                        
                        if "_lzma" in error_message:
                            st.info("This error is related to the missing _lzma module. To fix it:\n\n"
                                  "1. Install the lzma module:\n"
                                  "```\npip install -U lzma\n```\n\n"
                                  "2. Or on macOS, install xz:\n"
                                  "```\nbrew install xz\n```\n\n"
                                  "Then reinstall Python with lzma support.")
                        elif "No features could be extracted" in error_message:
                            st.info("No audio features could be extracted. This might be because:\n"
                                  "1. The audio files are not in the correct format\n"
                                  "2. There are no valid audio files in the selected directory\n"
                                  "3. The audio files don't follow the RAVDESS naming convention\n\n"
                                  "Please check your audio files and try again.")
        else:
            # We already have features extracted
            features_df = st.session_state.audio_features
            
            if features_df is None or len(features_df) == 0:
                st.error("⚠️ Stored features are empty. Please re-run feature extraction.")
                if st.button("Re-run Feature Extraction", use_container_width=True):
                    st.session_state.audio_features = None
                    st.rerun()
            else:
                st.success(f"✅ Features already extracted from {len(features_df['file_path'].unique())} audio files")
                
                # Count emotions
                emotion_counts = features_df['emotion'].value_counts()
                
                if not emotion_counts.empty:
                    # Display emotion distribution
                    st.subheader("Emotion Distribution in Dataset")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    emotion_counts.plot(kind='bar', ax=ax)
                    plt.title('Count of Emotions')
                    plt.xlabel('Emotions')
                    plt.ylabel('Count')
                    st.pyplot(fig)
                else:
                    st.warning("No emotion data to display.")
                
                # Continue to model training
                if st.button("✅ Continue to Model Training", use_container_width=True):
                    st.session_state.audio_step = "4. Model Training"
                    st.rerun()
    
    # 4. Model Training
    elif st.session_state.audio_step == "4. Model Training":
        st.header("🧠 Model Training")
        
        if st.session_state.audio_features is None:
            st.warning("Please extract features first")
            st.session_state.audio_step = "3. Feature Extraction"
            st.rerun()
        
        # Check if features are empty
        features_df = st.session_state.audio_features
        if features_df is None or len(features_df) == 0:
            st.error("⚠️ No features available for training. Please go back and extract features.")
            if st.button("← Back to Feature Extraction", use_container_width=True):
                st.session_state.audio_features = None
                st.session_state.audio_step = "3. Feature Extraction"
                st.rerun()
            st.stop()
        
        with st.expander("About Model Architecture", expanded=True):
            st.markdown("""
            We're using a 1D Convolutional Neural Network (CNN) for emotion classification:
            
            - **Input**: Audio features (ZCR, Chroma, RMS, Mel Spectrogram)
            - **Architecture**:
                - Multiple Conv1D layers with increasing depth (256 → 256 → 128 → 64)
                - MaxPooling layers to reduce dimensionality
                - Dropout layers (0.2, 0.3) to prevent overfitting
                - Dense layers for final classification
            - **Output**: Emotion probability distribution
            
            This architecture has proven effective for audio classification tasks.
            """)
        
        # Model training options
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5) / 100
        with col2:
            epochs = st.slider("Number of epochs", min_value=10, max_value=100, value=20, step=5)
        
        # Start training
        if st.session_state.audio_model_results is None:
            if st.button("🚀 Start Model Training", use_container_width=True):
                with st.spinner("Training emotion recognition model... This may take several minutes."):
                    try:
                        # Progress bar for training
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        # Simulate initial progress
                        progress_text.text("Preparing data...")
                        progress_bar.progress(10)
                        time.sleep(0.5)
                        
                        # Train the model
                        model_results = train_audio_model(
                            st.session_state.audio_features,
                            test_size=test_size,
                            epochs=epochs
                        )
                        
                        st.session_state.audio_model_results = model_results
                        
                        # Complete progress bar
                        progress_bar.progress(100)
                        progress_text.text("Training completed!")
                        
                        # Show success message
                        st.success(f"✅ Model training completed with {model_results['accuracy']*100:.2f}% accuracy!")
                        
                        # Display training metrics per epoch
                        st.subheader("Training Progress (Epoch by Epoch)")
                        if 'epoch_metrics' in model_results:
                            metrics_df = pd.DataFrame(model_results['epoch_metrics'])
                            st.dataframe(metrics_df)
                            
                            # Additional metrics visualization
                            st.write("Epoch Metrics Visualization:")
                            
                            # Create a plot with both training and validation metrics
                            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                            
                            # Plot accuracy
                            ax[0].plot(metrics_df['epoch'], metrics_df['accuracy'], 'b-', label='Training Accuracy')
                            ax[0].plot(metrics_df['epoch'], metrics_df['val_accuracy'], 'r-', label='Validation Accuracy')
                            ax[0].set_title('Accuracy per Epoch')
                            ax[0].set_ylabel('Accuracy')
                            ax[0].legend()
                            ax[0].grid(True)
                            
                            # Plot loss
                            ax[1].plot(metrics_df['epoch'], metrics_df['loss'], 'b-', label='Training Loss')
                            ax[1].plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-', label='Validation Loss')
                            ax[1].set_title('Loss per Epoch')
                            ax[1].set_ylabel('Loss')
                            ax[1].set_xlabel('Epoch')
                            ax[1].legend()
                            ax[1].grid(True)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Display training history
                        st.subheader("Training History")
                        
                        # Plot training history
                        history = model_results['history']
                        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                        
                        # Plot accuracy
                        ax[0].plot(history.history['accuracy'], label='Training Accuracy')
                        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
                        ax[0].set_title('Model Accuracy')
                        ax[0].set_ylabel('Accuracy')
                        ax[0].set_xlabel('Epoch')
                        ax[0].legend()
                        
                        # Plot loss
                        ax[1].plot(history.history['loss'], label='Training Loss')
                        ax[1].plot(history.history['val_loss'], label='Validation Loss')
                        ax[1].set_title('Model Loss')
                        ax[1].set_ylabel('Loss')
                        ax[1].set_xlabel('Epoch')
                        ax[1].legend()
                        
                        st.pyplot(fig)
                        
                        # Continue button appears after training
                        if st.button("✅ Continue to Evaluation & Testing", key="after_training", use_container_width=True):
                            st.session_state.audio_step = "5. Evaluation & Testing"
                            st.rerun()
                            
                    except Exception as e:
                        error_message = str(e)
                        st.error(f"Error during model training: {error_message}")
                        
                        if "No features available for training" in error_message:
                            st.info("The feature extraction process didn't produce valid features. Please go back to Feature Extraction "
                                   "and try again, ensuring your audio files are in the correct format.")
                        elif "Not enough data for training" in error_message:
                            st.info("You need at least 10 audio samples for training. Please add more audio files or try "
                                   "a different dataset.")
                        elif "_lzma" in error_message:
                            st.info("This error is related to the missing _lzma module. To fix it:\n\n"
                                   "1. Install the lzma module:\n"
                                   "```\npip install -U lzma\n```\n\n"
                                   "2. Or on macOS, install xz:\n"
                                   "```\nbrew install xz\n```\n\n"
                                   "Then restart the application.")
                        else:
                            st.info("Try the following:\n"
                                   "1. Make sure your audio files are valid WAV files\n"
                                   "2. Check that the file names follow the RAVDESS format\n"
                                   "3. Try with a smaller number of epochs\n"
                                   "4. Restart the application")
        else:
            # We already have a trained model
            model_results = st.session_state.audio_model_results
            
            st.success(f"✅ Model already trained with {model_results['accuracy']*100:.2f}% accuracy!")
            
            # Display training metrics per epoch
            st.subheader("Training Progress (Epoch by Epoch)")
            if 'epoch_metrics' in model_results:
                metrics_df = pd.DataFrame(model_results['epoch_metrics'])
                st.dataframe(metrics_df)
                
                # Additional metrics visualization
                st.write("Epoch Metrics Visualization:")
                
                # Create a plot with both training and validation metrics
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                
                # Plot accuracy
                ax[0].plot(metrics_df['epoch'], metrics_df['accuracy'], 'b-', label='Training Accuracy')
                ax[0].plot(metrics_df['epoch'], metrics_df['val_accuracy'], 'r-', label='Validation Accuracy')
                ax[0].set_title('Accuracy per Epoch')
                ax[0].set_ylabel('Accuracy')
                ax[0].legend()
                ax[0].grid(True)
                
                # Plot loss
                ax[1].plot(metrics_df['epoch'], metrics_df['loss'], 'b-', label='Training Loss')
                ax[1].plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-', label='Validation Loss')
                ax[1].set_title('Loss per Epoch')
                ax[1].set_ylabel('Loss')
                ax[1].set_xlabel('Epoch')
                ax[1].legend()
                ax[1].grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Display training history
            st.subheader("Training History")
            
            # Plot training history
            history = model_results['history']
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax[0].plot(history.history['accuracy'], label='Training Accuracy')
            ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax[0].set_title('Model Accuracy')
            ax[0].set_ylabel('Accuracy')
            ax[0].set_xlabel('Epoch')
            ax[0].legend()
            
            # Plot loss
            ax[1].plot(history.history['loss'], label='Training Loss')
            ax[1].plot(history.history['val_loss'], label='Validation Loss')
            ax[1].set_title('Model Loss')
            ax[1].set_ylabel('Loss')
            ax[1].set_xlabel('Epoch')
            ax[1].legend()
            
            st.pyplot(fig)
            
            # Continue to evaluation
            if st.button("✅ Continue to Evaluation & Testing", use_container_width=True):
                st.session_state.audio_step = "5. Evaluation & Testing"
                st.rerun()
    
    # 5. Evaluation & Testing
    elif st.session_state.audio_step == "5. Evaluation & Testing":
        st.header("📊 Model Evaluation & Testing")
        
        if st.session_state.audio_model_results is None:
            st.warning("Please train the model first")
            st.session_state.audio_step = "4. Model Training"
            st.rerun()
        
        model_results = st.session_state.audio_model_results
        
        # Show evaluation metrics
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{model_results['accuracy']*100:.2f}%")
        with col2:
            st.metric("Loss", f"{model_results['loss']:.4f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        # Plot confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(model_results['y_true_classes'], model_results['y_pred_classes'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=model_results['emotion_labels'],
                yticklabels=model_results['emotion_labels']
            )
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Error displaying confusion matrix: {e}")
            st.info("Could not generate the confusion matrix visualization. This might be due to insufficient test data or visualization errors.")
        
        # Classification report
        st.subheader("Classification Report")
        
        try:
            from sklearn.metrics import classification_report
            
            report = classification_report(
                model_results['y_true_classes'],
                model_results['y_pred_classes'],
                target_names=model_results['emotion_labels'],
                output_dict=True
            )
            
            # Convert to dataframe for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        except Exception as e:
            st.error(f"Error generating classification report: {e}")
            st.info("Could not generate the classification report. This might be due to insufficient test data.")
        
        # Download model
        st.subheader("Download Trained Model")
        
        model_zip = save_trained_model(model_results)
        
        st.download_button(
            label="Download Emotion Recognition Model",
            data=model_zip,
            file_name="emotion_recognition_model.zip",
            mime="application/zip"
        )
        
        # Test with new audio
        st.subheader("Test with New Audio")
        
        uploaded_test_file = st.file_uploader("Upload an audio file for testing", type=["wav"])
        
        if uploaded_test_file:
            # Save temporarily
            temp_file = "./temp_test_audio.wav"
            with open(temp_file, "wb") as f:
                f.write(uploaded_test_file.getbuffer())
            
            # Allow playback
            st.audio(temp_file)
            
            # Predict
            try:
                # Load audio
                data, sample_rate = librosa.load(temp_file)
                
                # Show waveform
                st.subheader("Audio Waveform")
                try:
                    fig_waveform = plot_waveform(data, sample_rate, "Audio Waveform")
                    st.pyplot(fig_waveform)
                except Exception as e:
                    st.error(f"Error displaying waveform: {e}")
                
                # Predict emotion
                prediction_result = predict_emotion_from_audio(
                    temp_file,
                    model_results['model'],
                    model_results['scaler'],
                    model_results['encoder']
                )
                
                if 'error' in prediction_result:
                    st.error(f"Prediction error: {prediction_result['error']}")
                else:
                    # Show prediction
                    st.subheader("Prediction Result")
                    st.markdown(f"### Detected Emotion: {prediction_result['emotion'].upper()}")
                    st.markdown(f"Confidence: {prediction_result['confidence']*100:.2f}%")
                    
                    # Show all probabilities
                    st.markdown("### All Emotion Probabilities")
                    
                    # Create bar chart of probabilities
                    probs = prediction_result['all_probabilities']
                    emotions = list(probs.keys())
                    values = list(probs.values())
                    
                    try:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(emotions, values)
                        
                        # Highlight the predicted emotion
                        for i, emotion in enumerate(emotions):
                            if emotion == prediction_result['emotion']:
                                bars[i].set_color('red')
                        
                        plt.title('Emotion Probabilities')
                        plt.xlabel('Emotion')
                        plt.ylabel('Probability')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error displaying probability chart: {e}")
                        # Show a fallback table view of the probabilities
                        st.write("Emotion probabilities:")
                        prob_df = pd.DataFrame({
                            'Emotion': emotions,
                            'Probability': [f"{v*100:.2f}%" for v in values]
                        })
                        st.dataframe(prob_df)
            
            except Exception as e:
                st.error(f"Error analyzing test file: {e}")
        
        # Start over button
        if st.button("🔄 Start Over", use_container_width=True):
            # Reset audio-specific session states
            st.session_state.audio_step = "1. Select Data Source"
            st.session_state.audio_features = None
            st.session_state.audio_model_results = None
            st.rerun()

# Original ML Workflow mode
elif st.session_state.app_mode == "ml_workflow":
    # Create sidebar to show current workflow step
    st.sidebar.title("ML Workflow Steps")
    
    # Initialize current step in session state if not present
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "1. Upload Data"

    # Add a button to switch back to mode selection
    if st.sidebar.button("← Back to Mode Selection"):
        st.session_state.app_mode = None
        st.rerun()
    
    # Display the current step using markdown with visual cues
    st.sidebar.markdown("### Current Step:")
    for step in ["1. Upload Data", "2. Feature Selection", "3. Preprocessing", "4. Model Selection", "5. Train & Evaluate"]:
        if step == st.session_state.current_step:
            st.sidebar.markdown(f"**→ {step}** 🔵")
        else:
            st.sidebar.markdown(f"  {step}")
            
    st.title("Machine Learning Workflow Assistant")
    st.markdown("---")

    # Add a function to navigate to the next step
    def go_to_next_step(next_step):
        if next_step in ["1. Upload Data", "2. Feature Selection", "3. Preprocessing", "4. Model Selection", "5. Train & Evaluate"]:
            st.session_state.current_step = next_step
            st.rerun()

    # ML workflow steps
    if st.session_state.current_step == "1. Upload Data":
        st.header("📥 Upload Data")
        
        with st.expander("How to use", expanded=True):
            st.markdown("""
            - Upload CSV, Excel 
            - The data will be automatically analyzed
            - You'll see a preview of the first 5 rows and data types
            """)
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            try:
                data, file_type = load_data(uploaded_file)
                st.session_state.data = data
                st.session_state.original_data = data.copy()
                st.session_state.data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
                st.session_state.features = list(data.columns)
                
                st.success(f"✅ Successfully loaded {file_type} file!")
                
                # Preview data
                st.subheader("Data Preview")
                st.dataframe(get_data_preview(data))
                
                # Display data types
                st.subheader("Data Types")
                data_types_df = pd.DataFrame({
                    "Feature": st.session_state.data_types.keys(),
                    "Data Type": st.session_state.data_types.values()
                })
                st.dataframe(data_types_df)
                
                # Display basic statistics
                st.subheader("Data Summary")
                st.write(f"Number of rows: {data.shape[0]}")
                st.write(f"Number of columns: {data.shape[1]}")
                
                # Detect and show missing values
                missing_values = data.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning("⚠️ Missing values detected in the dataset!")
                    missing_df = pd.DataFrame({
                        "Feature": missing_values.index,
                        "Missing Values": missing_values.values,
                        "Missing Percentage": (missing_values.values / len(data) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df["Missing Values"] > 0].sort_values(
                        by="Missing Values", ascending=False
                    )
                    st.dataframe(missing_df)
                    
                # Add continue button
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("✅ Continue to Next Step", use_container_width=True):
                        # Update the current step and rerun
                        st.session_state.current_step = "2. Feature Selection"
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("Please upload a file to get started.")

    # 2. Feature Selection
    elif st.session_state.current_step == "2. Feature Selection":
        st.header("🔍 Feature Selection")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            st.stop()
        
        with st.expander("How feature selection works", expanded=True):
            st.markdown("""
            - Select features to include in your model
            - Choose a target variable (what you want to predict)
            - The system will highlight correlated features
            - A preliminary feature importance ranking will help you make better choices
            """)
        
        data = st.session_state.data
        
        # Calculate correlation matrix for numeric features
        numeric_data = data.select_dtypes(include=["number"])
        if not numeric_data.empty:
            st.subheader("Correlation Matrix")
            fig = plot_correlation_matrix(numeric_data)
            st.pyplot(fig)
        
        # Feature selection
        st.subheader("Select Features")
        
        # Target selection
        st.markdown("### Select Target Variable")
        st.markdown("This is the variable you want to predict.")
        
        target_col = st.selectbox(
            "Which feature is your target?",
            options=st.session_state.features,
            index=None,
            key="target_selector"
        )
        
        if target_col:
            st.session_state.target = target_col
            potential_features = [col for col in st.session_state.features if col != target_col]
            
            # Detect if target is categorical or numeric
            if pd.api.types.is_numeric_dtype(data[target_col]):
                if data[target_col].nunique() <= 10:
                    st.info(f"🎯 Target '{target_col}' is numeric but has only {data[target_col].nunique()} unique values. It might be treated as a classification problem.")
                else:
                    st.info(f"🎯 Target '{target_col}' is numeric - this will be a regression problem.")
            else:
                st.info(f"🎯 Target '{target_col}' is categorical - this will be a classification problem.")
            
            # Calculate feature importance
            if len(potential_features) > 0:
                try:
                    with st.spinner("Calculating preliminary feature importance..."):
                        feature_importances = get_feature_importance(data, potential_features, target_col)
                        st.session_state.feature_importances = feature_importances
                        
                        if feature_importances is not None:
                            st.subheader("Preliminary Feature Importance")
                            fig = plot_feature_importance(feature_importances)
                            st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not calculate feature importance: {str(e)}")
            
            # Feature selection with importance suggestions
            st.markdown("### Select Features to Use")
            st.markdown("Choose the features you want to include in your model:")
            
            if st.session_state.feature_importances is not None:
                # Sort features by importance
                sorted_features = [
                    feature for feature, _ in sorted(
                        st.session_state.feature_importances.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                ]
                selected_features = []
                
                for feature in sorted_features:
                    importance = st.session_state.feature_importances.get(feature, 0)
                    
                    # Highlight important features
                    if importance > 0.05:
                        selected = st.checkbox(
                            f"{feature} - Importance: {importance:.4f} 🌟",
                            value=True,
                            key=f"feature_{feature}"
                        )
                    elif importance > 0.01:
                        selected = st.checkbox(
                            f"{feature} - Importance: {importance:.4f}",
                            value=True,
                            key=f"feature_{feature}"
                        )
                    else:
                        selected = st.checkbox(
                            f"{feature} - Importance: {importance:.4f} (low importance)",
                            value=False,
                            key=f"feature_{feature}"
                        )
                    
                    if selected:
                        selected_features.append(feature)
            else:
                selected_features = st.multiselect(
                    "Select features to use for prediction:",
                    options=potential_features,
                    default=potential_features
                )
            
            if len(selected_features) > 0:
                st.session_state.selected_features = selected_features
                st.success(f"✅ Selected {len(selected_features)} features and '{target_col}' as target.")
                
                # Add continue button at the end of feature selection
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("✅ Continue to Preprocessing", use_container_width=True):
                        # Update the current step and rerun
                        st.session_state.current_step = "3. Preprocessing"
                        st.rerun()
            else:
                st.warning("⚠️ Please select at least one feature.")
        
        else:
            st.markdown("### Clustering Mode")
            st.info("📌 No target selected. The system will run in clustering mode if you proceed.")
            selected_features = st.multiselect(
                "Select features for clustering:",
                options=st.session_state.features,
                default=st.session_state.features[:min(5, len(st.session_state.features))]
            )
            
            if len(selected_features) > 0:
                st.session_state.selected_features = selected_features
                st.session_state.target = None
                st.success(f"✅ Selected {len(selected_features)} features for clustering.")
                
                # Add continue button for clustering mode too
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("✅ Continue to Preprocessing", use_container_width=True, key="clustering_continue"):
                        # Update the current step and rerun
                        st.session_state.current_step = "3. Preprocessing"
                        st.rerun()
            else:
                st.warning("⚠️ Please select at least one feature.")

    # 3. Preprocessing
    elif st.session_state.current_step == "3. Preprocessing":
        st.header("⚙️ Preprocessing")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            st.stop()
        
        if not st.session_state.selected_features:
            st.warning("⚠️ Please select features for your model first!")
            st.stop()
        
        with st.expander("About preprocessing", expanded=True):
            st.markdown("""
            - Handle missing values in your data
            - Encode categorical variables
            - Scale numeric features
            - Detect and handle outliers
            """)
        
        data = st.session_state.data.copy()
        
        # Get relevant columns (selected features + target if available)
        relevant_columns = st.session_state.selected_features.copy()
        if st.session_state.target:
            relevant_columns.append(st.session_state.target)
        
        # Filter the data to include only the relevant columns
        data = data[relevant_columns]
        
        # 1. Missing Value Handling
        st.subheader("Missing Value Handling")
        missing_info = detect_missing_values(data)
        
        if missing_info.empty:
            st.success("✅ No missing values detected in the selected columns.")
        else:
            st.warning("⚠️ Missing values detected in the selected columns:")
            
            # Create a more compact table interface for handling missing values
            missing_columns = []
            missing_percentages = []
            handling_methods = []
            constant_values = []
            
            for col in missing_info["Feature"].tolist():
                missing_pct = missing_info.loc[missing_info["Feature"] == col, "Missing Percentage"].values[0]
                missing_columns.append(col)
                missing_percentages.append(f"{missing_pct:.2f}%")
                
                # Create a unique key for each column's method selection
                col_key = f"method_{col}"
                method_options = ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with constant"]
                
                # Default method suggestion based on percentage and data type
                default_idx = 0  # Default to drop rows
                if missing_pct < 5:  # Very few missing values
                    if pd.api.types.is_numeric_dtype(data[col]):
                        default_idx = 2  # Default to median for numeric with few missing
                    else:
                        default_idx = 3  # Default to mode for categorical with few missing
                elif 5 <= missing_pct < 20:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        default_idx = 1  # Default to mean for numeric with moderate missing
                    else:
                        default_idx = 3  # Default to mode for categorical with moderate missing
                
                handling_methods.append(st.selectbox(
                    f"Method for '{col}'",
                    options=method_options,
                    index=default_idx,
                    key=col_key
                ))
                
                # Handle constant value if needed
                if handling_methods[-1] == "Fill with constant":
                    if pd.api.types.is_numeric_dtype(data[col]):
                        constant_values.append(st.number_input(
                            f"Value for '{col}'",
                            value=0,
                            key=f"const_{col}"
                        ))
                    else:
                        constant_values.append(st.text_input(
                            f"Value for '{col}'",
                            value="missing",
                            key=f"const_{col}"
                        ))
                else:
                    constant_values.append(None)
            
            # Display missing values table
            missing_table = pd.DataFrame({
                "Feature": missing_columns,
                "Missing": missing_percentages,
                "Method": handling_methods
            })
            st.dataframe(missing_table)
            
            # Process all missing values with selected methods
            for i, col in enumerate(missing_columns):
                data = handle_missing_values(
                    data,
                    col,
                    method=handling_methods[i],
                    constant_value=constant_values[i]
                )
        
        # 2. Encoding Categorical Features
        st.subheader("Categorical Encoding")
        
        categorical_columns = [
            col for col in data.columns 
            if not pd.api.types.is_numeric_dtype(data[col]) and col in st.session_state.selected_features
        ]
        
        if not categorical_columns:
            st.success("✅ No categorical features to encode among the selected features.")
        else:
            st.write("Categorical features detected:")
            for col in categorical_columns:
                st.write(f"- {col} (Unique values: {data[col].nunique()})")
            
            encoding_method = st.radio(
                "Select encoding method for categorical features:",
                options=["One-Hot Encoding", "Label Encoding"],
                index=0
            )
            
            # Encode categorical features
            data = encode_categorical_features(data, categorical_columns, method=encoding_method)
        
        # 3. Feature Scaling
        st.subheader("Feature Scaling")
        
        numeric_columns = [
            col for col in data.columns 
            if pd.api.types.is_numeric_dtype(data[col]) and col in st.session_state.selected_features
        ]
        
        if not numeric_columns:
            st.warning("⚠️ No numeric features found among the selected features.")
        else:
            scaling_needed = st.checkbox("Apply feature scaling", value=True)
            
            if scaling_needed:
                scaling_method = st.radio(
                    "Select scaling method:",
                    options=["StandardScaler (Z-score normalization)", "MinMaxScaler (0-1 scaling)"],
                    index=0
                )
                
                # Apply scaling based on the selected method
                method = "standard" if "StandardScaler" in scaling_method else "minmax"
                data = scale_features(data, numeric_columns, method=method)
                st.success("✅ Scaling applied to numeric features.")
        
        # 5. Task Type Detection
        st.subheader("Task Type Detection")
        
        if st.session_state.target:
            task_type = detect_task_type(data, st.session_state.target)
            st.session_state.task_type = task_type
            
            if task_type == "regression":
                st.info("📊 Detected task type: Regression")
            elif task_type == "classification":
                st.info("🔍 Detected task type: Classification")
        else:
            st.info("📚 No target selected. The system will run in clustering mode.")
            st.session_state.task_type = "clustering"
        
        # Save the preprocessed data and provide a Next button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✅ Apply Preprocessing & Continue", use_container_width=True):
                # Store the preprocessed data
                st.session_state.data = data
                st.session_state.preprocessing_done = True
                
                st.success("✅ Preprocessing completed successfully!")
                
                # Update the current step and rerun
                st.session_state.current_step = "4. Model Selection"
                st.rerun()

    # 4. Model Selection
    elif st.session_state.current_step == "4. Model Selection":
        st.header("🧠 Model Selection")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            st.stop()
        
        # Only check for preprocessing completion if using the standard workflow (not directly clicking in sidebar)
        if not st.session_state.preprocessing_done and not st.session_state.current_step == "4. Model Selection":
            st.warning("⚠️ Please complete the preprocessing step first!")
            st.session_state.current_step = "3. Preprocessing"
            st.rerun()
        
        with st.expander("About model selection", expanded=True):
            st.markdown("""
            - Choose a model appropriate for your task
            - Configure the train-test split ratio
            - You can let the system auto-select the best model
            """)
        
        # Train-Test Split Configuration
        st.subheader("Train-Test Split")
        
        test_size = st.slider(
            "Select test set size (%):",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        ) / 100
        
        # Model Selection based on task type
        st.subheader("Model Selection")
        
        if st.session_state.task_type:
            task_type = st.session_state.task_type
            st.info(f"Task type: {task_type.capitalize()}")
            
            # Get model suggestions based on task type
            model_options = get_model_suggestions(task_type)
            
            if task_type == "classification":
                with st.expander("About Classification Models", expanded=False):
                    st.markdown("""
                    **Classification Models Information:**
                    
                    - **K-Nearest Neighbors (KNN)**: Yes, this is a classification technique. It classifies data points based on the majority class of their k-nearest neighbors in the feature space.
                    - **Decision Tree**: Creates a tree-like structure to make decisions.
                    - **Random Forest**: Combines multiple decision trees for better accuracy.
                    - **SVM**: Finds an optimal hyperplane to separate classes.
                    """)
            
            selected_model = st.selectbox(
                "Select a model:",
                options=list(model_options.keys())
            )
            
            st.session_state.selected_model = selected_model
            
            # Display model description
            st.markdown(f"**About {selected_model}:**")
            st.markdown(model_options[selected_model]["description"])
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Continue to Training", use_container_width=True):
                    st.success("✅ Model selection complete!")
                    st.session_state.test_size = test_size
                    # Set default random state to 42
                    st.session_state.random_state = 42
                    
                    # Update the current step and rerun
                    st.session_state.current_step = "5. Train & Evaluate"
                    st.rerun()
        else:
            st.error("⚠️ Task type not detected. Please go back to preprocessing.")

    # 5. Train & Evaluate
    elif st.session_state.current_step == "5. Train & Evaluate":
        st.header("🚀 Model Training & Evaluation")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            st.stop()
        
        if not st.session_state.preprocessing_done:
            st.warning("⚠️ Please complete the preprocessing step first!")
            st.stop()
        
        if not hasattr(st.session_state, 'selected_model') or not st.session_state.selected_model:
            st.warning("⚠️ Please select a model first!")
            st.stop()
        
        with st.expander("About training & evaluation", expanded=True):
            st.markdown("""
            - Train your selected model on the preprocessed data
            - Evaluate model performance with appropriate metrics
            - Visualize the results
            - Download the trained model and predictions
            """)
        
        # Show selected configuration
        st.subheader("Model Configuration")
        st.write(f"Task Type: {st.session_state.task_type.capitalize()}")
        if st.session_state.selected_model == "auto":
            st.write("Model: Auto-select the best model")
        else:
            st.write(f"Model: {st.session_state.selected_model}")
        st.write(f"Test Size: {st.session_state.test_size * 100:.0f}%")
        
        # Train Model Button
        if not st.session_state.model_trained:
            if st.button("🚀 Train Model"):
                data = st.session_state.data
                
                # Create progress bar for training
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Training progress simulation
                for i in range(5):
                    progress_text.text(f"Training in progress... Step {i+1}/5")
                    progress_bar.progress((i+1) * 20)
                    time.sleep(0.5)
                
                try:
                    with st.spinner("Training model..."):
                        # Train the model
                        (
                            model, 
                            X_train, 
                            X_test, 
                            y_train, 
                            y_test, 
                            predictions, 
                            metrics, 
                            actual_model_name
                        ) = train_model(
                            data,
                            st.session_state.selected_features,
                            st.session_state.target,
                            st.session_state.task_type,
                            st.session_state.selected_model,
                            test_size=st.session_state.test_size,
                            random_state=st.session_state.random_state
                        )
                        
                        # Store results in session state
                        st.session_state.model = model
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.predictions = predictions
                        st.session_state.evaluation_metrics = metrics
                        st.session_state.actual_model_name = actual_model_name
                        st.session_state.model_trained = True
                    
                    progress_text.text("Training completed!")
                    st.success(f"✅ Model training completed successfully! Used model: {actual_model_name}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        
        # Display model evaluation results if model is trained
        if st.session_state.model_trained:
            st.subheader("Model Evaluation")
            
            if st.session_state.task_type == "regression":
                # Display regression metrics
                metrics = st.session_state.evaluation_metrics
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}")
                    st.markdown("*Lower is better. Average absolute difference between predicted and actual values.*")
                    
                    st.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.4f}")
                    st.markdown("*Lower is better. Average squared difference between predicted and actual values.*")
                    
                with col2:
                    st.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.4f}")
                    st.markdown("*Lower is better. Square root of MSE, in the same units as the target.*")
                    
                    st.metric("R² Score", f"{metrics['r2']:.4f}")
                    st.markdown("*Higher is better (max 1.0). Proportion of variance explained by the model.*")
            
            elif st.session_state.task_type == "classification":
                # Display classification metrics
                metrics = st.session_state.evaluation_metrics
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    st.markdown("*Higher is better. Proportion of correct predictions.*")
                    
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                    st.markdown("*Higher is better. Ability to not label a negative sample as positive.*")
                
                with col2:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                    st.markdown("*Higher is better. Ability to find all positive samples.*")
                    
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                    st.markdown("*Higher is better. Harmonic mean of precision and recall.*")
            
            # Plot confusion matrix
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(
                st.session_state.y_test, 
                st.session_state.predictions, 
                metrics['class_names']
            )
            st.pyplot(fig)
            
            # Download options
            st.subheader("Download Options")
            
            # Download model
            model_download = save_model(st.session_state.model)
            st.download_button(
                label="Download Trained Model",
                data=model_download,
                file_name=f"{st.session_state.actual_model_name}_model.joblib",
                mime="application/octet-stream"
            )
            
            # Option to start over
            if st.button("🔄 Start Over with New Model"):
                # Reset all session state variables for a completely fresh start
                st.session_state.data = None
                st.session_state.original_data = None
                st.session_state.data_types = None
                st.session_state.features = []
                st.session_state.target = None
                st.session_state.selected_features = []
                st.session_state.task_type = None
                st.session_state.preprocessing_done = False
                st.session_state.model_trained = False
                st.session_state.model = None
                st.session_state.predictions = None
                st.session_state.feature_importances = None
                st.session_state.evaluation_metrics = {}
                st.session_state.X_train = None
                st.session_state.X_test = None
                st.session_state.y_train = None
                st.session_state.y_test = None
                st.session_state.current_step = "1. Upload Data"
                st.rerun()
    else:
        st.info("Please select a workflow step from the sidebar to continue.")

else:
    st.info("Please select a mode to continue.")
