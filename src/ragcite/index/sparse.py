from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def fit(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = list(chunk_ids)
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25Index.fit() must be called before search()")
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]
