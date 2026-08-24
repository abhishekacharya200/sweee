from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from ragcite.config import settings
from ragcite.index.embeddings import EmbeddingBackend, build_embedding_backend
from ragcite.index.hybrid import reciprocal_rank_fusion
from ragcite.index.rerank import Reranker, build_reranker
from ragcite.index.sparse import BM25Index
from ragcite.models import Chunk, ScoredChunk

INDEX_FILE = "index.pkl"


class IndexStore:
    """Owns the chunk corpus plus the fitted sparse + dense indexes, and
    runs hybrid retrieval (BM25 + dense, RRF-fused) followed by reranking.
    """

    def __init__(
        self,
        embedding_backend: str | None = None,
        rerank_backend: str | None = None,
    ):
        self.embedding_backend_name = embedding_backend or settings.embedding_backend
        self.rerank_backend_name = rerank_backend or settings.rerank_backend

        self.chunks: dict[str, Chunk] = {}
        self.chunk_order: list[str] = []
        self.bm25 = BM25Index()
        self.embedder: EmbeddingBackend | None = None
        self.dense_matrix: np.ndarray | None = None
        self._reranker: Reranker | None = None

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = build_reranker(self.rerank_backend_name, settings.cross_encoder_model)
        return self._reranker

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.chunk_order = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]

        self.bm25.fit(self.chunk_order, texts)

        self.embedder = build_embedding_backend(
            self.embedding_backend_name, settings.sentence_transformers_model
        )
        self.embedder.fit(texts)
        self.dense_matrix = self.embedder.encode(texts)

    def save(self, index_dir: Path | None = None) -> Path:
        index_dir = index_dir or settings.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / INDEX_FILE
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": {cid: c.to_dict() for cid, c in self.chunks.items()},
                    "chunk_order": self.chunk_order,
                    "bm25": self.bm25,
                    "embedder": self.embedder,
                    "dense_matrix": self.dense_matrix,
                    "embedding_backend_name": self.embedding_backend_name,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, index_dir: Path | None = None, rerank_backend: str | None = None) -> IndexStore:
        index_dir = index_dir or settings.index_dir
        path = index_dir / INDEX_FILE
        with open(path, "rb") as f:
            state = pickle.load(f)
        store = cls(embedding_backend=state["embedding_backend_name"], rerank_backend=rerank_backend)
        store.chunks = {cid: Chunk.from_dict(d) for cid, d in state["chunks"].items()}
        store.chunk_order = state["chunk_order"]
        store.bm25 = state["bm25"]
        store.embedder = state["embedder"]
        store.dense_matrix = state["dense_matrix"]
        return store

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.embedder is None or self.dense_matrix is None:
            raise RuntimeError("IndexStore.build() or .load() must be called first")
        query_vec = self.embedder.encode([query])[0]
        sims = self.dense_matrix @ query_vec
        order = np.argsort(-sims)[:top_k]
        return [(self.chunk_order[i], float(sims[i])) for i in order if sims[i] > 0]

    def search(
        self,
        query: str,
        top_k_sparse: int | None = None,
        top_k_dense: int | None = None,
        top_k_fused: int | None = None,
        top_k_final: int | None = None,
    ) -> tuple[list[ScoredChunk], dict[str, float]]:
        top_k_sparse = top_k_sparse or settings.top_k_sparse
        top_k_dense = top_k_dense or settings.top_k_dense
        top_k_fused = top_k_fused or settings.top_k_fused
        top_k_final = top_k_final or settings.top_k_final

        t0 = time.perf_counter()
        sparse_hits = self.bm25.search(query, top_k_sparse)
        t1 = time.perf_counter()
        dense_hits = self._dense_search(query, top_k_dense)
        t2 = time.perf_counter()

        fused = reciprocal_rank_fusion([sparse_hits, dense_hits])[:top_k_fused]
        candidates = [self.chunks[cid] for cid, _score, _ranks in fused]
        fused_scores = {cid: score for cid, score, _ranks in fused}
        rank_detail = {cid: ranks for cid, _score, ranks in fused}
        t3 = time.perf_counter()

        reranked = self.reranker.rerank(query, candidates)[:top_k_final]
        t4 = time.perf_counter()

        results = [
            ScoredChunk(
                chunk=chunk,
                score=score,
                rank_scores={
                    "fused_rrf": fused_scores.get(chunk.chunk_id, 0.0),
                    "sparse_rank": rank_detail.get(chunk.chunk_id, {}).get(0),
                    "dense_rank": rank_detail.get(chunk.chunk_id, {}).get(1),
                    "rerank_score": score,
                },
            )
            for chunk, score in reranked
        ]

        latency = {
            "sparse_s": t1 - t0,
            "dense_s": t2 - t1,
            "fuse_s": t3 - t2,
            "rerank_s": t4 - t3,
            "retrieval_total_s": t4 - t0,
        }
        return results, latency
