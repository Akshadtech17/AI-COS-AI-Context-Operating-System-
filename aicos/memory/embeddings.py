"""
Fast local embedding engine using feature hashing.

Produces 512-dimensional dense vectors via character n-gram hashing + feature
hashing trick. No external model required — suitable for semantic cache and
memory retrieval with sub-millisecond performance.

For production deployments requiring higher accuracy, drop-in replacement with
sentence-transformers or OpenAI embeddings is supported via the EmbeddingEngine
interface.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """Protocol for pluggable embedding backends."""

    def encode(self, texts: list[str]) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class HashEmbeddingModel:
    """
    Feature-hashing embedding model. Produces stable 512-d vectors using:
      - Unigrams, bigrams (word-level)
      - Character 3-grams, 4-grams
      - Multiple hash functions for lower collision rate

    No training required. Consistent across processes.
    """

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text.lower())
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_features(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        words = normalized.split()

        features: list[str] = []

        # Word unigrams
        features.extend(words)

        # Word bigrams
        features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))

        # Character n-grams (3-4 grams) from full text
        compact = normalized.replace(" ", "")
        features.extend(compact[i:i+3] for i in range(len(compact) - 2))
        features.extend(compact[i:i+4] for i in range(len(compact) - 3))

        return features

    def _hash_to_index(self, feature: str, seed: int) -> int:
        h = int(hashlib.md5(f"{seed}:{feature}".encode()).hexdigest(), 16)
        return h % self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self._dim), dtype=np.float32)

        for i, text in enumerate(texts):
            features = self._extract_features(text)
            vec = result[i]

            for feat in features:
                # Two hash functions to reduce collision impact
                idx1 = self._hash_to_index(feat, 0)
                idx2 = self._hash_to_index(feat, 1)
                sign = 1 if self._hash_to_index(feat, 2) % 2 == 0 else -1
                vec[idx1] += sign * 1.0
                vec[idx2] += sign * 0.5

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

        return result


class EmbeddingEngine:
    """
    High-level embedding engine with caching and optional backend upgrade.

    Priority:
    1. sentence-transformers (if installed and configured)
    2. HashEmbeddingModel (always available, no deps)
    """

    def __init__(self, model_name: str | None = None, dim: int = 512) -> None:
        self._model: EmbeddingModel = self._load_model(model_name, dim)
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self, model_name: str | None, dim: int) -> EmbeddingModel:
        # Use sentence-transformers when available for true semantic similarity.
        # Falls back to HashEmbeddingModel when ST is not installed (zero-dep path).
        target = model_name or "all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer

            class STModel:
                def __init__(self, name: str) -> None:
                    self._model = SentenceTransformer(name)

                @property
                def dim(self) -> int:
                    return int(self._model.get_sentence_embedding_dimension() or 384)

                def encode(self, texts: list[str]) -> np.ndarray:
                    return np.array(self._model.encode(texts, normalize_embeddings=True))

            return STModel(target)
        except (ImportError, Exception):
            pass

        return HashEmbeddingModel(dim=dim)

    @property
    def dim(self) -> int:
        return self._model.dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text, using in-memory cache."""
        if text not in self._cache:
            self._cache[text] = self._model.encode([text])[0]
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts, bypassing cache for memory efficiency."""
        return self._model.encode(texts)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors."""
        return float(np.dot(a, b))

    def batch_similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Vectorized cosine similarity: query (d,) × candidates (N, d) → (N,).
        Assumes both are L2-normalized.
        """
        return np.dot(candidates, query)

    def top_k(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        k: int,
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Return indices and scores of top-k most similar candidates above threshold."""
        scores = self.batch_similarity(query, candidates)
        # Partial sort — much faster than full sort for large candidate sets
        if len(scores) > k:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if float(scores[idx]) >= threshold
        ]

    def clear_cache(self) -> None:
        self._cache.clear()
