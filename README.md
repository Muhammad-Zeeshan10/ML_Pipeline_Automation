# AutoMachineLearning

A comprehensive machine learning workflow assistant with audio analysis capabilities, built using Streamlit.

## Features

- **ML Workflow**: Upload CSV/Excel data, preprocess, train models, and evaluate results
- **Audio Analysis**: Analyze speech emotion from audio files using the RAVDESS dataset
- **Interactive UI**: Step-by-step guided workflow with visualizations
- **Model Training**: Train and evaluate ML models with detailed performance metrics

## Prerequisites

- Python 3.11.x (TensorFlow is not compatible with Python 3.13)
- macOS, Linux, or Windows
- Homebrew (for macOS users)
- pyenv (recommended for Python version management)
- xz (required for lzma support)

## Installation

### Step 1: Set up Python environment

TensorFlow requires Python 3.11 or earlier. This project has been tested with Python 3.11.9.

#### Using pyenv (recommended)

```bash
# Install pyenv if you don't have it
brew install pyenv  # macOS
# or follow instructions at https://github.com/pyenv/pyenv#installation for other platforms

# Install xz (required for lzma module)
brew install xz  # macOS
# On Ubuntu/Debian: sudo apt-get install liblzma-dev
# On Fedora/RHEL: sudo dnf install xz-devel

# Install Python 3.11.9
pyenv install 3.11.9

# Create a virtual environment
pyenv virtualenv 3.11.9 tf-env
pyenv activate tf-env

# Verify lzma support
python -c "import lzma; print('lzma module available')"
```

If you see "lzma module available", you're good to go. If you get an error, try reinstalling Python with pyenv after installing xz:

```bash
pyenv uninstall 3.11.9
pyenv install 3.11.9
pyenv virtualenv 3.11.9 tf-env
pyenv activate tf-env
```

#### Using venv/conda

Alternatively, you can use venv or conda, but ensure you're using Python 3.11.x:

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Clone the repository

```bash
git clone https://github.com/yourusername/AutoMachineLearning.git
cd AutoMachineLearning
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt file includes a specific constraint for NumPy (<2.0.0) to ensure compatibility with TensorFlow.

## Running the Application

Once installed, run the application with:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your web browser.

## Usage

The application offers two main modes:

### ML Workflow Mode

1. **Upload Data**: Upload CSV/Excel files
2. **Feature Selection**: Choose target variable and features
3. **Preprocessing**: Handle missing values, encode categorical features, scale data
4. **Model Selection**: Choose an appropriate machine learning model
5. **Train & Evaluate**: Train the model and view performance metrics

### Audio Analysis Mode

1. **Select Data Source**: Use the RAVDESS dataset or upload your own audio files
2. **Explore & Visualize**: Visualize audio waveforms and spectrograms
3. **Feature Extraction**: Extract audio features for emotion recognition
4. **Model Training**: Train a CNN model for emotion classification
5. **Evaluation & Testing**: Evaluate model performance and test with new audio

## Troubleshooting

### Missing _lzma Module

If you encounter an error about the missing _lzma module:

```
ModuleNotFoundError: No module named '_lzma'
```

Fix it with these steps:

1. Install xz (if not already installed):
```bash
brew install xz  # macOS
# On Ubuntu/Debian: sudo apt-get install liblzma-dev
# On Fedora/RHEL: sudo dnf install xz-devel
```

2. Reinstall Python via pyenv (now that xz is available):
```bash
pyenv uninstall 3.11.9
pyenv install 3.11.9
```

3. Recreate your virtual environment:
```bash
pyenv virtualenv 3.11.9 tf-env
pyenv activate tf-env
```

4. Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### TensorFlow/NumPy Compatibility Issues

If you encounter errors related to NumPy and TensorFlow compatibility, ensure you're using a version of NumPy that's compatible with TensorFlow:

```bash
pip install numpy<2.0.0
```

The requirements.txt file already includes this constraint, but if you've installed a newer version manually, you may need to downgrade.

### Audio File Support

For audio analysis, ensure you have:
- WAV files
- For RAVDESS dataset, the files should follow the RAVDESS naming convention

## Credits

This project utilizes:
- Streamlit for the UI
- scikit-learn for ML algorithms
- TensorFlow for deep learning
- librosa for audio processing
- The RAVDESS dataset for emotional speech analysis

## License

[Your chosen license] 