# Complaint Routing System

This is a machine learning pipeline that automatically processes complaints submitted in text, audio, or video formats. The system uses local models to assign complaints to the correct officer, predict the priority level, estimate the resolution time (in days), and find similar past complaints.

All inference is done locally without external APIs. 

## Setup

1. Install the required Python packages:
```bash
pip install -r requirements.txt
pip install openai-whisper gradio
```

2. To support audio/video extraction, ensure `ffmpeg` is installed on your system.

## Usage

You can launch the web interface by running:
```bash
python app/web_app.py
```
Then open `http://localhost:7860` in your browser.

## Project Structure
- `app/` - Contains the web app and CLI interfaces.
- `models/` - Contains the training scripts and saved ML models (SVM, Random Forest, Gradient Boosting).
- `inference/` - The core inference engine and vector store for semantic similarity.
- `audio_video/` - Wrappers for local Whisper transcription.
- `data/` - Training data.

## Models Used
- **Officer Routing**: Support Vector Classifier (SVC)
- **Priority Prediction**: Random Forest Classifier
- **ETA Prediction**: Gradient Boosting Regressor
- **Embeddings**: SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Transcription**: Whisper (Offline)
