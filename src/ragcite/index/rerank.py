"""Rerankers applied to the top fused candidates before final selection."""

from __future__ import annotations

from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragcite.index.sparse import tokenize
from ragcite.models import Chunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Chunk]) -> list[tuple[Chunk, float]]: ...


class LexicalReranker:
    """Offline cross-encoder substitute: per-pair TF-IDF cosine similarity
    blended with raw query-term coverage. Cheap, deterministic, and needs
    no model download -- a reasonable default for an on-prem deployment."""

    name = "lexical"

    def rerank(self, query: str, candidates: list[Chunk]) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        texts = [c.text for c in candidates]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        matrix = vectorizer.fit_transform(texts + [query])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

        query_terms = set(tokenize(query))
        scored = []
        for chunk, sim in zip(candidates, sims):
            chunk_terms = set(tokenize(chunk.text))
            coverage = len(query_terms & chunk_terms) / max(1, len(query_terms))
            blended = 0.7 * float(sim) + 0.3 * coverage
            scored.append((chunk, blended))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class CrossEncoderReranker:
    """Real cross-encoder reranker via sentence-transformers. Lazily
    imported so the dependency stays optional."""

    name = "cross-encoder"

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Install extras: "
                "pip install -e '.[embeddings]'"
            ) from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Chunk]) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def build_reranker(backend_name: str, model_name: str) -> Reranker:
    if backend_name == "lexical":
        return LexicalReranker()
    if backend_name == "cross-encoder":
        return CrossEncoderReranker(model_name)
    raise ValueError(f"Unknown rerank backend: {backend_name}")
