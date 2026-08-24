from ragcite.index.hybrid import reciprocal_rank_fusion
from ragcite.index.sparse import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("HbA1c of 6.5%!") == ["hba1c", "of", "6", "5"]


def test_bm25_finds_exact_keyword_match():
    idx = BM25Index()
    idx.fit(
        ["a", "b", "c"],
        [
            "the capital conservation buffer is 2.5 percent",
            "unrelated passage about foot examinations",
            "another unrelated passage about data breaches",
        ],
    )
    hits = idx.search("capital conservation buffer", top_k=3)
    assert hits[0][0] == "a"


def test_reciprocal_rank_fusion_prefers_items_ranked_highly_in_both_lists():
    sparse = [("x", 5.0), ("y", 3.0), ("z", 1.0)]
    dense = [("y", 0.9), ("x", 0.5), ("z", 0.1)]
    fused = reciprocal_rank_fusion([sparse, dense])
    fused_ids = [f[0] for f in fused]
    # x and y are top-2 in both rankings; z is last in both, so it must be last.
    assert fused_ids[-1] == "z"
    assert set(fused_ids[:2]) == {"x", "y"}


def test_index_store_retrieves_relevant_chunk_at_top_rank(index_store):
    results, latency = index_store.search(
        "What is the minimum Common Equity Tier 1 capital ratio a covered depository "
        "institution must maintain?"
    )
    assert results
    assert "risk-weighted assets" in results[0].chunk.text or "Common Equity Tier 1" in results[0].chunk.text
    assert latency["retrieval_total_s"] >= 0
