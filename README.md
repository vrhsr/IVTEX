# Complaint Auto-Routing System

An end-to-end machine learning system that processes citizen complaints submitted in text, audio, or video format. It is designed to run fully offline without external APIs.

It automatically performs the following tasks:
- **Officer Routing**: Assigns the complaint to the most suitable officer from an internal list using an SVM.
- **Priority Prediction**: Classifies the priority as High, Medium, or Low using a Random Forest.
- **ETA Prediction**: Estimates resolution time in days using a Gradient Boosting Regressor.
- **Similarity Search**: Retrieves similar past complaints using cosine similarity over text embeddings.

---

## Project Structure

```
complaint-routing-system/
├── data/
│   ├── generate_data.py          # Synthetic multilingual complaint generator
│   └── synthetic_complaints.csv  # 800 labelled complaints (auto-generated)
├── models/
│   ├── train.py                  # End-to-end training pipeline
│   └── saved/                    # Trained model artifacts
├── inference/
│   ├── embedding_engine.py       # TF-IDF+SVD or sentence-transformers
│   ├── vector_store.py           # NumPy cosine search
│   └── engine.py                 # Core inference engine (load + predict)
├── audio_video/
│   └── transcriber.py            # Whisper-based offline ASR (audio + video)
├── app/
│   ├── cli.py                    # Command-line interface
│   └── web_app.py                # Gradio web UI
├── evaluation/
│   └── evaluate.py               # Full evaluation suite
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# Core requirements
pip install scikit-learn numpy pandas scipy joblib

# Embedding model (multilingual support)
pip install sentence-transformers

# Web UI
pip install gradio

# Audio/Video transcription (requires ffmpeg installed on your system)
pip install openai-whisper
```

### 2. Generate training data and train models

```bash
python data/generate_data.py
python models/train.py
```

### 3. Run inference

```bash
# Interactive CLI
python app/cli.py

# Direct text
python app/cli.py --text "Pothole on MG Road near hospital causing accidents. URGENT!"

# Web UI
python app/web_app.py
```

### 4. Run evaluation

```bash
python evaluation/evaluate.py
```

---

## Architecture

The system takes in text, audio, or video. If audio or video is provided, it uses the local Whisper model to transcribe the speech offline.

The text is then passed to an embedding engine (using sentence-transformers for multilingual support). The resulting vector is passed to three separate scikit-learn models:
1. Support Vector Machine (Officer Routing)
2. Random Forest (Priority)
3. Gradient Boosting Regressor (ETA)

The vector is also compared against a local NumPy store of historical complaints using cosine similarity to fetch the most relevant past issues.

---

## Evaluation Results (5-fold Cross-Validation)

### Officer Routing (SVM)
| Metric | Value |
|--------|-------|
| CV Accuracy | 1.000 ± 0.000 |
| CV F1-macro | 1.000 ± 0.000 |

Note: The synthetic data has very clear departmental boundaries (e.g., "pothole" maps strictly to roads). In a real-world scenario, accuracy would be closer to 85-90%.

### Priority Prediction (Random Forest)
| Metric | Value |
|--------|-------|
| CV Accuracy | 0.686 ± 0.024 |
| CV F1-macro | 0.661 ± 0.041 |

### ETA Prediction (Gradient Boosting)
| Metric | Value |
|--------|-------|
| CV MAE | 5.47 days |
| CV RMSE | 7.69 days |

---

## Replacing Synthetic Data with Real Data

1. Prepare a CSV with columns: `text`, `officer_id`, `priority`, `eta_days`
2. Replace `data/synthetic_complaints.csv`
3. Run `python models/train.py`

The pipeline will automatically retrain all models on the new data and overwrite the old artifacts in `models/saved`.
