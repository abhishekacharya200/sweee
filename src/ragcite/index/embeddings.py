"""Dense embedding backends.

Default is a fully offline TF-IDF + truncated-SVD "local" embedding so the
whole pipeline (ingest -> index -> retrieve -> generate -> eval) runs with
zero network calls and zero API keys. Set EMBEDDING_BACKEND=sentence-transformers
to swap in real sentence embeddings when that dependency + model download
is available.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class EmbeddingBackend(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalTfidfEmbedding:
    """Offline dense-ish embedding: TF-IDF followed by truncated SVD
    (latent semantic analysis). Captures co-occurrence structure beyond raw
    lexical overlap without requiring any model download.
    """

    name = "local-tfidf-svd"

    def __init__(self, n_components: int = 128, random_state: int = 13):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=20000
        )
        self.svd: TruncatedSVD | None = None
        self.random_state = random_state

    def fit(self, texts: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.n_components, max(2, min(tfidf.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.svd.fit(tfidf)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("LocalTfidfEmbedding.fit() must be called before encode()")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return normalize(dense)


class SentenceTransformerEmbedding:
    """Real sentence embeddings via sentence-transformers. Lazily imported
    so the dependency is optional."""

    name = "sentence-transformers"

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when opted in
            raise RuntimeError(
                "sentence-transformers is not installed. Install extras: "
                "pip install -e '.[embeddings]'"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        return None  # pretrained model, nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors)


def build_embedding_backend(backend_name: str, model_name: str) -> EmbeddingBackend:
    if backend_name == "local":
        return LocalTfidfEmbedding()
    if backend_name == "sentence-transformers":
        return SentenceTransformerEmbedding(model_name)
    raise ValueError(f"Unknown embedding backend: {backend_name}")
