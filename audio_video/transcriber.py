"""
transcriber.py
────────────────────────────────────────────────────────────────
Multimodal transcription module.

Supported input types:
    Audio  → .wav, .mp3, .m4a, .ogg, .flac
    Video  → .mp4, .mkv, .avi, .mov  (audio track extracted via ffmpeg)

Backend: openai-whisper (LOCAL model — NOT the API)
    pip install openai-whisper
    pip install ffmpeg-python  (for video processing)
    apt install ffmpeg         (system dependency)

Whisper model sizes (all run fully offline):
    tiny    ~39 MB   — fastest, lower accuracy
    base    ~74 MB   — good balance for demos
    small   ~244 MB  — better for noisy audio
    medium  ~769 MB  — high accuracy
    large   ~1.5 GB  — best quality, multilingual

Language detection is automatic — no explicit lang flag needed.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


class WhisperTranscriber:
    """
    Thin wrapper around openai-whisper for offline speech-to-text.
    Handles both audio and video (extracts audio via ffmpeg internally).
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Parameters
        ----------
        model_size  : "tiny" | "base" | "small" | "medium" | "large"
        device      : "cpu" | "cuda"  (auto-detected if torch available)
        """
        try:
            import whisper
            self.whisper   = whisper
            self.model     = whisper.load_model(model_size, device=device)
            self.model_size = model_size
            self._available = True
            print(f"[Transcriber] whisper '{model_size}' model loaded on {device}.")
        except ImportError:
            self._available = False
            print(
                "[Transcriber] openai-whisper not installed.\n"
                "  pip install openai-whisper\n"
                "  This is a LOCAL model — no API key required."
            )

    @property
    def available(self) -> bool:
        return self._available

    def _extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio track from video using ffmpeg.
        Returns path to temporary .wav file.
        """
        tmp_audio = tempfile.mktemp(suffix=".wav")
        cmd = (
            f"ffmpeg -i \"{video_path}\" -vn -acodec pcm_s16le "
            f"-ar 16000 -ac 1 \"{tmp_audio}\" -y -loglevel error"
        )
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract audio from {video_path}.\n"
                "Install ffmpeg: sudo apt install ffmpeg"
            )
        return tmp_audio

    def transcribe(self, file_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe a file to text.

        Parameters
        ----------
        file_path   : path to audio or video file
        language    : ISO-639-1 code e.g. "en", "hi", "ta" — None = auto-detect

        Returns
        -------
        dict with keys: text, language, segments
        """
        if not self._available:
            raise RuntimeError("openai-whisper not installed.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        tmp_path = None

        if ext in VIDEO_EXTS:
            print(f"[Transcriber] Extracting audio from video …")
            tmp_path  = self._extract_audio_from_video(file_path)
            audio_in  = tmp_path
        elif ext in AUDIO_EXTS:
            audio_in = file_path
        else:
            raise ValueError(
                f"Unsupported extension: {ext}\n"
                f"Audio: {AUDIO_EXTS}\nVideo: {VIDEO_EXTS}"
            )

        try:
            kw = {"task": "transcribe"}
            if language:
                kw["language"] = language

            result = self.model.transcribe(audio_in, **kw)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {
            "text":     result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end":   seg["end"],
                    "text":  seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        }


# ─── Design note for reviewers ──────────────────────────────────
"""
MULTIMODAL PIPELINE DESIGN
────────────────────────────
Input (text / audio / video)
         │
         ▼
  ┌──────────────────────┐
  │  Modality Detection  │  → check file extension
  └──────────┬───────────┘
             │
     ┌───────┴────────┐
     │                │
  [Audio/Video]    [Text]
     │                │
     ▼                │
 WhisperTranscriber   │
 (local, offline)     │
     │                │
     └───────┬────────┘
             │
             ▼
      Normalized Text
             │
             ▼
  EmbeddingEngine.encode()
  (TF-IDF+SVD  or  sentence-transformers)
             │
             ▼
  ┌──────────────────────────────┐
  │  Parallel Inference          │
  │  T1: Officer SVM classifier  │
  │  T2: Priority RF classifier  │
  │  T3: ETA GBR regressor       │
  │  T4: VectorStore top-K       │
  └──────────────────────────────┘
             │
             ▼
       Structured JSON result

Language Support:
  • TF-IDF + SVD  : char n-grams (2-5) → language-agnostic tokenisation
  • sentence-transformers: paraphrase-multilingual-MiniLM-L12-v2 supports 50+ languages
  • Whisper: 99-language ASR with automatic language detection

All processing is fully offline after the one-time model download.
"""


if __name__ == "__main__":
    # Demo: print design overview and check availability
    t = WhisperTranscriber(model_size="base")
    print(f"\nWhisper available: {t.available}")
    print("\nSee docstring above for full pipeline design.")
