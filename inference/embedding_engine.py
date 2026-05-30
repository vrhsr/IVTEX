"""
embedding_engine.py
────────────────────────────────────────────────────────────────
Offline embedding strategy using TF-IDF + SVD (LSA) for
environments where sentence-transformers cannot be installed
due to disk/network constraints.

When sentence-transformers IS available (recommended for
production), swap EmbeddingEngine with SentenceTransformerEngine
below.  The rest of the pipeline is identical.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


# ─────────────────────────────────────────────────────────────
# Baseline: TF-IDF + Latent Semantic Analysis (offline, fast)
# ─────────────────────────────────────────────────────────────
class TFIDFEmbeddingEngine:
    """
    Produces 256-dim L2-normalised embeddings via TF-IDF → SVD.
    Works completely offline with no extra downloads.
    Handles multilingual input through char n-grams.
    """

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer      = "char_wb",   # char n-grams → language-agnostic
                ngram_range   = (2, 5),
                max_features  = 50_000,
                sublinear_tf  = True,
                strip_accents = "unicode",
                lowercase     = True,
            )),
            ("svd",  TruncatedSVD(n_components=n_components, random_state=42)),
            ("norm", Normalizer(norm="l2")),
        ])
        self.is_fitted = False

    def fit(self, texts: list[str]) -> "TFIDFEmbeddingEngine":
        self.pipeline.fit(texts)
        self.is_fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return (N, n_components) float32 array."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .encode()")
        return self.pipeline.transform(texts).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ─────────────────────────────────────────────────────────────
# Upgrade path: sentence-transformers (recommended)
# ─────────────────────────────────────────────────────────────
class SentenceTransformerEngine:
    """
    Drop-in replacement using paraphrase-multilingual-MiniLM-L12-v2.
    Supports 50+ languages out-of-the-box.
    Install: pip install sentence-transformers
    The model is downloaded once and cached locally (~120 MB).

    Usage:
        engine = SentenceTransformerEngine()
        vectors = engine.encode(texts)
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.is_fitted = True
        except ImportError:
            raise ImportError(
                "pip install sentence-transformers  # then retry"
            )

    def fit(self, texts):          # no-op: pretrained
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True,
                                 show_progress_bar=False, batch_size=32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedding_engine(prefer_transformer: bool = True):
    """Factory: use sentence-transformers if available, else TF-IDF+SVD."""
    if prefer_transformer:
        try:
            engine = SentenceTransformerEngine()
            print("[EmbeddingEngine] Using sentence-transformers (multilingual).")
            return engine
        except (ImportError, Exception) as e:
            print(f"[EmbeddingEngine] sentence-transformers unavailable ({e})."
                  " Falling back to TF-IDF+SVD.")
    print("[EmbeddingEngine] Using TF-IDF + SVD (offline baseline).")
    return TFIDFEmbeddingEngine()
