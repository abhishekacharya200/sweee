"""Reciprocal Rank Fusion of sparse (BM25) and dense retrieval rankings."""

from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float, dict[int, int]]]:
    """Fuse N ranked lists of (chunk_id, score) into one ranking.

    Returns list of (chunk_id, fused_score, {ranking_index: rank}) sorted
    by fused_score descending. RRF is scale-free, so it composes a lexical
    BM25 score with a cosine similarity score without any normalization.
    """
    fused: dict[str, float] = {}
    per_ranking_rank: dict[str, dict[int, int]] = {}

    for ranking_idx, ranking in enumerate(rankings):
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
            per_ranking_rank.setdefault(chunk_id, {})[ranking_idx] = rank

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [(cid, score, per_ranking_rank[cid]) for cid, score in ordered]
