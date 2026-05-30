"""
vector_store.py
────────────────────────────────────────────────────────────────
Similarity search backend.

Two implementations:
  1. NumpyVectorStore  – pure NumPy cosine search (default, zero deps)
  2. FAISSVectorStore  – FAISS IndexFlatIP for large corpora (optional)

Both expose the same API so the training script / engine can
swap them via a flag.
"""

import numpy as np
from typing import List, Dict, Any


class NumpyVectorStore:
    """
    Brute-force cosine similarity search over a numpy matrix.
    Suitable for corpora up to ~50k documents (sub-ms on modern hardware).
    """

    def __init__(self):
        self.vectors: np.ndarray = None   # (N, D) float32, L2-normalised
        self.metadata: List[Dict] = []

    def build(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        vectors  : (N, D) L2-normalised float32
        metadata : list of dicts with complaint info
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        self.vectors  = (vectors / norms).astype(np.float32)
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Return top-k most similar complaints with scores."""
        q = query_vector.astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        scores = self.vectors @ q                   # cosine similarity
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                **self.metadata[idx],
                "similarity_score": float(scores[idx]),
            })
        return results

    def save(self, path: str):
        import joblib
        joblib.dump({"vectors": self.vectors, "metadata": self.metadata}, path)
        print(f"[VectorStore] Saved {len(self.metadata)} entries -> {path}")

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.vectors  = data["vectors"]
        self.metadata = data["metadata"]
        print(f"[VectorStore] Loaded {len(self.metadata)} entries from {path}")
        return self


class FAISSVectorStore:
    """
    FAISS-based vector store.  Faster for large corpora (>50k docs).
    Requires: pip install faiss-cpu
    """

    def __init__(self, dim: int = 256):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("pip install faiss-cpu  # then retry")
        self.dim      = dim
        self.index    = None
        self.metadata = []

    def build(self, vectors: np.ndarray, metadata: List[Dict]):
        import faiss
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        vecs  = (vectors / norms).astype(np.float32)
        self.dim   = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        self.metadata = metadata
        print(f"[FAISSVectorStore] Built index with {self.index.ntotal} vectors (dim={self.dim})")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        q = query_vector.astype(np.float32).reshape(1, -1)
        q /= (np.linalg.norm(q) + 1e-9)
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({**self.metadata[idx], "similarity_score": float(score)})
        return results

    def save(self, path: str):
        import faiss, joblib
        faiss.write_index(self.index, path + ".faiss")
        joblib.dump(self.metadata, path + ".meta")

    def load(self, path: str):
        import faiss, joblib
        self.index    = faiss.read_index(path + ".faiss")
        self.metadata = joblib.load(path + ".meta")
        self.dim      = self.index.d
        return self


def get_vector_store(use_faiss: bool = False, dim: int = 256):
    """Factory: return FAISS store if available, else NumPy store."""
    if use_faiss:
        try:
            store = FAISSVectorStore(dim=dim)
            print("[VectorStore] Using FAISS IndexFlatIP.")
            return store
        except ImportError:
            print("[VectorStore] FAISS not installed. Falling back to NumPy cosine search.")
    return NumpyVectorStore()
