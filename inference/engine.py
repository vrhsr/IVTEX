"""
engine.py
────────────────────────────────────────────────────────────────
Inference engine for the Complaint Auto-Routing System.

Loads all trained artifacts once and exposes:
    predict(text)       → officer, priority, ETA, similar complaints
    transcribe(path)    → text  (requires openai-whisper installed)

Whisper note:
    pip install openai-whisper   # local model, NOT the API
    The model weights (~140MB for 'base') download once and run offline.
    Audio: .wav / .mp3 / .m4a / .ogg
    Video: .mp4 / .mkv / .avi  (audio track extracted automatically)
"""

import os
import sys
import math
import joblib
import numpy as np
from typing import Optional

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
SAVE_DIR  = os.path.join(BASE_DIR, "models", "saved")

sys.path.insert(0, BASE_DIR)
from inference.vector_store import NumpyVectorStore

# ─── Officer registry (mirrors generate_data.py) ──────────────
OFFICERS = [
    {"id": "OFF001", "name": "Rahul Sharma",  "department": "Infrastructure & Roads"},
    {"id": "OFF002", "name": "Priya Mehta",   "department": "Water & Sanitation"},
    {"id": "OFF003", "name": "Amit Verma",    "department": "Electricity & Utilities"},
    {"id": "OFF004", "name": "Sunita Patel",  "department": "Public Safety & Security"},
    {"id": "OFF005", "name": "Vijay Kumar",   "department": "Health & Environment"},
    {"id": "OFF006", "name": "Anjali Singh",  "department": "Land & Property"},
    {"id": "OFF007", "name": "Ravi Nair",     "department": "Transport & Traffic"},
    {"id": "OFF008", "name": "Meena Reddy",   "department": "Administrative Services"},
]
OFFICER_MAP = {o["id"]: o for o in OFFICERS}


class ComplaintRoutingEngine:
    """Singleton-style inference engine; call load() before predict()."""

    def __init__(self):
        self.embedding_engine  = None
        self.officer_model     = None
        self.priority_model    = None
        self.eta_model         = None
        self.vector_store      = None
        self.label_encoders    = None
        self._loaded           = False

    # ─── Loading ──────────────────────────────────────────────
    def load(self, save_dir: str = SAVE_DIR) -> "ComplaintRoutingEngine":
        print("[Engine] Loading models...")
        self.embedding_engine = joblib.load(
            os.path.join(save_dir, "embedding_engine.pkl"))
        self.officer_model    = joblib.load(
            os.path.join(save_dir, "officer_classifier.pkl"))
        self.priority_model   = joblib.load(
            os.path.join(save_dir, "priority_classifier.pkl"))
        self.eta_model        = joblib.load(
            os.path.join(save_dir, "eta_regressor.pkl"))
        self.label_encoders   = joblib.load(
            os.path.join(save_dir, "label_encoders.pkl"))
        self.vector_store     = NumpyVectorStore().load(
            os.path.join(save_dir, "vector_store.pkl"))
        self._loaded = True
        print("[Engine] All models loaded.")
        return self

    # ─── Core prediction ──────────────────────────────────────
    def predict(self, text: str, top_k_similar: int = 5) -> dict:
        """
        Parameters
        ----------
        text            : complaint text (any language)
        top_k_similar   : number of similar past complaints to retrieve

        Returns
        -------
        dict with keys:
            officer, priority, eta_days, confidence, similar_complaints
        """
        if not self._loaded:
            raise RuntimeError("Call .load() first.")

        # 1. Embed
        vec = self.embedding_engine.encode_single(text)

        # 2. Officer routing
        le_off      = self.label_encoders["officer"]
        off_proba   = self.officer_model.predict_proba([vec])[0]
        off_idx     = int(np.argmax(off_proba))
        officer_id  = le_off.inverse_transform([off_idx])[0]
        officer_info = OFFICER_MAP[officer_id]
        confidence  = float(off_proba[off_idx])

        # 3. Priority prediction
        le_pri      = self.label_encoders["priority"]
        pri_proba   = self.priority_model.predict_proba([vec])[0]
        pri_idx     = int(np.argmax(pri_proba))
        priority    = le_pri.inverse_transform([pri_idx])[0]
        pri_conf    = float(pri_proba[pri_idx])

        # 4. ETA prediction (round to nearest half-day, min 1)
        eta_raw  = float(self.eta_model.predict([vec])[0])
        eta_days = max(1, round(eta_raw * 2) / 2)   # round to 0.5-day granularity

        # 5. Similarity search
        similar = self.vector_store.search(vec, top_k=top_k_similar + 1)
        # strip identical-text match if present
        similar = [s for s in similar
                   if s["text"].strip() != text.strip()][:top_k_similar]

        return {
            "officer": {
                "id":         officer_id,
                "name":       officer_info["name"],
                "department": officer_info["department"],
                "confidence": round(confidence * 100, 1),
            },
            "priority": {
                "level":      priority,
                "confidence": round(pri_conf * 100, 1),
            },
            "eta_days":         eta_days,
            "similar_complaints": [
                {
                    "complaint_id":     s["complaint_id"],
                    "text_snippet":     s["text"][:120] + "…",
                    "officer_name":     s["officer_name"],
                    "priority":         s["priority"],
                    "eta_days":         s["eta_days"],
                    "similarity_score": round(s["similarity_score"], 4),
                }
                for s in similar
            ],
        }

    # ─── Audio / Video transcription ──────────────────────────
    def transcribe(self, file_path: str, whisper_model: str = "base") -> str:
        """
        Transcribe audio or video file to text using openai-whisper (local).

        Install:  pip install openai-whisper
        Models:   tiny | base | small | medium | large
                  'base' ~140 MB, supports 99 languages, runs on CPU.

        For video, Whisper extracts the audio track automatically via ffmpeg.
        """
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper not installed.\n"
                "Run: pip install openai-whisper\n"
                "Note: This is a LOCAL model — no API key required."
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        model  = whisper.load_model(whisper_model)
        result = model.transcribe(file_path, task="transcribe")
        return result["text"].strip()

    # ─── End-to-end multimodal ────────────────────────────────
    def process(self, text: Optional[str] = None,
                audio_path: Optional[str] = None,
                video_path: Optional[str] = None,
                top_k: int = 5) -> dict:
        """
        One-shot entry point for text / audio / video input.
        Exactly one of text / audio_path / video_path should be non-None.
        """
        source_text = None
        modality    = "text"

        if text:
            source_text = text
            modality    = "text"
        elif audio_path:
            print(f"[Engine] Transcribing audio: {audio_path}")
            source_text = self.transcribe(audio_path)
            modality    = "audio"
        elif video_path:
            print(f"[Engine] Transcribing video: {video_path}")
            source_text = self.transcribe(video_path)
            modality    = "video"
        else:
            raise ValueError("Provide text, audio_path, or video_path.")

        result = self.predict(source_text, top_k_similar=top_k)
        result["modality"]     = modality
        result["source_text"]  = source_text
        return result


# ─── Module-level singleton ────────────────────────────────────
_engine: Optional[ComplaintRoutingEngine] = None

def get_engine() -> ComplaintRoutingEngine:
    global _engine
    if _engine is None:
        _engine = ComplaintRoutingEngine().load()
    return _engine


# ─── Quick smoke test ──────────────────────────────────────────
if __name__ == "__main__":
    engine = get_engine()

    samples = [
        "There is a massive pothole on MG Road near the hospital. "
        "Three accidents have already happened. Emergency! Urgent action needed.",
        "My ration card application has been pending for 45 days. "
        "Not urgent, but the matter requires attention when convenient.",
        "Sewage water is overflowing onto the street near Central Park. "
        "This is a serious problem affecting daily life.",
        "Bahut bada problem hai. A live electric wire has fallen near the school. "
        "People are in immediate danger. Urgent action needed!",
    ]

    for i, text in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"COMPLAINT #{i}")
        print(f"Text: {text[:80]}…")
        res = engine.predict(text)
        print(f"  -> Officer   : {res['officer']['name']} ({res['officer']['department']}) "
              f"[{res['officer']['confidence']}%]")
        print(f"  -> Priority  : {res['priority']['level']} [{res['priority']['confidence']}%]")
        print(f"  -> ETA       : {res['eta_days']} days")
        print(f"  -> Similar   : {len(res['similar_complaints'])} retrieved")
        if res['similar_complaints']:
            top = res['similar_complaints'][0]
            print(f"     Best match [{top['similarity_score']:.4f}]: {top['text_snippet'][:60]}…")
