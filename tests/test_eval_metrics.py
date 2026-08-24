from ragcite.eval.metrics import (
    citation_accuracy,
    context_precision,
    context_recall,
    keyword_recall,
)
from ragcite.models import AnswerResult, Citation


def _result(citations):
    return AnswerResult(
        question="q",
        answer="a",
        citations=citations,
        retrieved=[],
        used_chunk_ids=[c.chunk_id for c in citations],
        provider="mock",
        model="extractive-v1",
        latency_s={"total_s": 0.0},
        tokens={"input": 0, "output": 0},
        cost_usd=0.0,
        grounded=True,
        ungrounded_sentences=[],
    )


def test_context_precision_and_recall():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert context_precision(retrieved, relevant) == 0.5
    assert context_recall(retrieved, relevant) == 1.0
    assert context_recall(retrieved, {"a", "z"}) == 0.5


def test_context_recall_is_1_when_nothing_is_marked_relevant():
    assert context_recall(["a"], set()) == 1.0


def test_citation_accuracy_counts_only_valid_hits():
    citations = [
        Citation(marker=1, chunk_id="a", doc_title="D", page=1, quote="..."),
        Citation(marker=2, chunk_id="z", doc_title="D", page=1, quote="..."),
    ]
    result = _result(citations)
    assert citation_accuracy(result, {"a"}) == 0.5


def test_citation_accuracy_is_zero_with_no_citations():
    assert citation_accuracy(_result([]), {"a"}) == 0.0


def test_keyword_recall_is_case_insensitive():
    assert keyword_recall("The HbA1c target is 7.0%", ["HbA1c", "7.0%"]) == 1.0
    assert keyword_recall("Nothing relevant here", ["HbA1c"]) == 0.0
