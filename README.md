# 🏛️ Complaint Auto-Routing System



An end-to-end AI/ML system that processes citizen complaints submitted in **text, audio, or video** format and automatically:

| Task | Output | Model |
|------|--------|-------|
| Officer Routing | Most suitable officer from 8 officers | SVM (RBF) |
| Priority Prediction | High / Medium / Low | Random Forest |
| ETA Prediction | Estimated resolution time in days | Gradient Boosting Regressor |
| Similarity Search | Top-K past similar complaints | Cosine / FAISS vector search |

> ✅ **No external APIs. Fully offline. Multilingual.**

---

## 📁 Project Structure

```
complaint-routing-system/
├── data/
│   ├── generate_data.py          # Synthetic multilingual complaint generator
│   └── synthetic_complaints.csv  # 800 labelled complaints (auto-generated)
│
├── models/
│   ├── train.py                  # End-to-end training pipeline
│   └── saved/                    # Trained model artifacts
│       ├── officer_classifier.pkl
│       ├── priority_classifier.pkl
│       ├── eta_regressor.pkl
│       ├── embedding_engine.pkl
│       ├── vector_store.pkl
│       ├── label_encoders.pkl
│       ├── metrics.json
│       └── evaluation_report.json
│
├── inference/
│   ├── embedding_engine.py       # TF-IDF+SVD (offline) or sentence-transformers
│   ├── vector_store.py           # NumPy cosine or FAISS IndexFlatIP
│   └── engine.py                 # Core inference engine (load + predict)
│
├── audio_video/
│   └── transcriber.py            # Whisper-based offline ASR (audio + video)
│
├── app/
│   ├── cli.py                    # Command-line interface
│   └── web_app.py                # Gradio web UI
│
├── evaluation/
│   └── evaluate.py               # Full evaluation suite (all 4 tasks)
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
# Core (required)
pip install scikit-learn numpy pandas scipy joblib

# Embedding upgrade — multilingual, ~120 MB download (highly recommended)
pip install sentence-transformers

# Web UI (optional)
pip install gradio

# Audio/Video transcription (optional — local Whisper model, NOT the API)
pip install openai-whisper
sudo apt install ffmpeg          # for video audio extraction
```

### 2. Generate training data

```bash
python data/generate_data.py
# → generates data/synthetic_complaints.csv (800 multilingual complaints)
```

### 3. Train all models

```bash
python models/train.py
# → trains SVM, Random Forest, GBR, builds vector store
# → saves all artifacts to models/saved/
```

### 4. Run inference

```bash
# Interactive CLI
python app/cli.py

# Direct text
python app/cli.py --text "Pothole on MG Road near hospital causing accidents. URGENT!"

# Audio file
python app/cli.py --audio path/to/complaint.mp3

# Video file
python app/cli.py --video path/to/complaint.mp4

# JSON output
python app/cli.py --text "Sewage overflow near park" --json

# Web UI (requires gradio)
python app/web_app.py
```

### 5. Run evaluation

```bash
python evaluation/evaluate.py
# → prints full metrics report and saves evaluation_report.json
```

---

## 🏗️ Architecture

```
Input (text | audio | video)
         │
         ▼
  ┌──────────────────────┐
  │  Modality Detection  │
  └──────────┬───────────┘
             │
     ┌───────┴────────┐
     │                │
  [Audio/Video]    [Text]
     │                │
     ▼                │
 Whisper (local)      │    ← 99 languages, fully offline
     │                │
     └───────┬────────┘
             │
             ▼
     Normalised Text
             │
             ▼
  ┌──────────────────────────────┐
  │  EmbeddingEngine             │
  │  Option A: TF-IDF + SVD      │  ← offline baseline (char n-grams)
  │  Option B: sentence-trans.   │  ← multilingual upgrade, ~120 MB
  └──────────────┬───────────────┘
                 │
        256-dim L2-normalised vector
                 │
     ┌───────────┼───────────────┐
     │           │               │
     ▼           ▼               ▼
  SVM clf   RF clf         GBR regressor
  (officer) (priority)    (ETA days)
     │           │               │
     ▼           ▼               ▼
  OFF001-8  H/M/L + conf   ETA + conf

  + NumpyVectorStore / FAISS
     → top-K similar past complaints
```

---

## 🧪 Evaluation Results (5-fold Cross-Validation)

### T1 — Officer Routing (SVM)
| Metric | Value |
|--------|-------|
| CV Accuracy | 1.000 ± 0.000 |
| CV F1-macro | 1.000 ± 0.000 |
| Recall@5 Similarity | 1.000 |

> Officer routing achieves perfect CV scores because each department uses domain-specific vocabulary (e.g. "pothole" → Roads, "voltage" → Electricity). In production, expect 85–95% with real data and a multilingual transformer.

### T2 — Priority Prediction (Random Forest)
| Metric | Value |
|--------|-------|
| CV Accuracy | 0.686 ± 0.024 |
| CV F1-macro | 0.661 ± 0.041 |

> Priority signal comes from urgency language ("emergency", "life-threatening" vs "minor", "when convenient"). CV performance reflects realistic generalisation. With real complaint data, transformer fine-tuning typically reaches 80–90% F1.

### T3 — ETA Prediction (Gradient Boosting)
| Metric | Value |
|--------|-------|
| CV MAE | 5.47 days |
| CV RMSE | 7.69 days |
| CV R² | 0.529 |

> ETA is a noisy regression target in practice. Improvement paths: stratified sampling by department, priority-aware features, historical officer resolution data.

### T4 — Similarity Search
| Metric | Value |
|--------|-------|
| Recall@1 | 0.990 |
| Recall@5 | 1.000 |
| Recall@10 | 1.000 |

---

## 🌍 Multilingual Support

**Two-tier strategy:**

| Tier | Component | Languages |
|------|-----------|-----------|
| Embeddings | TF-IDF + char n-grams (offline baseline) | Any script via char-level tokenisation |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (upgrade) | 50+ languages, proper semantics |
| ASR | Whisper (local) | 99 languages, automatic detection |

**Tested language mixes in synthetic data:**
- English
- Hinglish (Hindi + English): "Bahut bada problem hai. …"
- Hinglish (transliterated): "Mera complaint yeh hai ki …"
- Tamil-English: "Romba kastam aaguthu. …"
- Spanish-English: "Es urgente. … Por favor actúe rápidamente."

---

## 👮 Officer Registry

| ID | Name | Department |
|----|------|-----------|
| OFF001 | Rahul Sharma | Infrastructure & Roads |
| OFF002 | Priya Mehta | Water & Sanitation |
| OFF003 | Amit Verma | Electricity & Utilities |
| OFF004 | Sunita Patel | Public Safety & Security |
| OFF005 | Vijay Kumar | Health & Environment |
| OFF006 | Anjali Singh | Land & Property |
| OFF007 | Ravi Nair | Transport & Traffic |
| OFF008 | Meena Reddy | Administrative Services |

---

## ⚖️ Design Trade-offs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding backbone | TF-IDF+SVD (offline) vs sentence-transformers | Zero-dependency baseline; drop-in upgrade path defined |
| Officer classifier | SVM (RBF) | Best generalisation on small corpora; probability calibration via `predict_proba` |
| Priority classifier | Random Forest | Handles class imbalance; interpretable feature importances |
| ETA model | Gradient Boosting Regressor | Captures non-linear ETA distributions per department/priority |
| Similarity search | NumPy cosine vs FAISS | NumPy for ≤50k docs (zero deps); FAISS for production scale |
| Audio/Video | openai-whisper (local) | 99 languages, 4 model sizes, runs on CPU, truly offline |
| Data | Synthetic (800 samples) | For demo; plug in real complaint CSV and retrain in 1 command |

---

## 🔄 Replacing Synthetic Data with Real Data

1. Prepare a CSV with columns: `text`, `officer_id`, `priority`, `eta_days`
2. Replace `data/synthetic_complaints.csv`
3. Run `python models/train.py`

The entire pipeline retrains in under 2 minutes on CPU.

---

## 🚀 Production Upgrade Path

```
Current (offline baseline)            Production upgrade
─────────────────────────────────     ───────────────────────────────────
TF-IDF + SVD (256-dim)           →   paraphrase-multilingual-MiniLM-L12-v2
NumPy cosine search              →   FAISS IndexFlatIP (millions of docs)
SVM / RF / GBR (sklearn)         →   Fine-tuned mBERT / XLM-RoBERTa
Whisper 'base' (74 MB)           →   Whisper 'medium' (769 MB) on GPU
Gradio (demo)                    →   FastAPI + React frontend
```

---

## 📄 License

MIT License
