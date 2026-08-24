"""RAGAS-style metrics, implemented locally (no `ragas` dependency) so the
whole eval loop runs offline and deterministically against the mock LLM in
CI, while still being usable against a real LLM provider.

These are lexical/TF-IDF approximations of the metrics RAGAS popularized
(faithfulness, answer relevancy, context precision/recall), plus a
citation-specific metric this project's brief calls out explicitly.
Swap in real NLI/LLM-judge scoring by replacing `_similarity` with a
provider call if higher-fidelity numbers are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragcite.generation.answer import split_sentences
from ragcite.models import AnswerResult


def _similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(lowercase=True, stop_words="english").fit_transform([a, b])
    except ValueError:
        return 0.0  # e.g. both strings are pure stopwords
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


@dataclass
class QuestionScore:
    id: str
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    citation_accuracy: float
    keyword_recall: float
    grounded: bool
    cost_usd: float
    latency_s: float
    retrieved_chunk_ids: list = field(default_factory=list)
    relevant_chunk_ids: list = field(default_factory=list)


def faithfulness(result: AnswerResult) -> float:
    """Mean, over answer sentences, of the max lexical similarity between
    that sentence and any chunk it cites. A sentence with no valid
    citation scores 0 (this is what check_grounding also flags)."""
    citation_by_marker = {c.marker: c for c in result.citations}
    sentences = split_sentences(result.answer)
    if not sentences:
        return 0.0

    scores = []
    for sentence in sentences:
        import re

        markers = [int(m) for m in re.findall(r"\[(\d+)\]", sentence)]
        cited_texts = []
        for m in markers:
            cite = citation_by_marker.get(m)
            if cite is None:
                continue
            match = next((sc.chunk for sc in result.retrieved if sc.chunk.chunk_id == cite.chunk_id), None)
            if match:
                cited_texts.append(match.text)
        if not cited_texts:
            scores.append(0.0)
            continue
        scores.append(max(_similarity(sentence, t) for t in cited_texts))
    return sum(scores) / len(scores)


def answer_relevancy(result: AnswerResult) -> float:
    return _similarity(result.question, result.answer)


def context_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return hits / len(retrieved_ids)


def context_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 1.0
    hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return hits / len(relevant_ids)


def citation_accuracy(result: AnswerResult, relevant_ids: set[str]) -> float:
    """Of the chunks actually cited in the answer, what fraction are
    chunks the golden set says are truly relevant to the question? This
    is the "citations point to evidence" check."""
    if not result.citations:
        return 0.0
    hits = sum(1 for c in result.citations if c.chunk_id in relevant_ids)
    return hits / len(result.citations)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    answer_l = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_l)
    return hits / len(keywords)


def score_question(golden: dict, result: AnswerResult) -> QuestionScore:
    relevant_ids = set(golden["relevant_chunk_ids"])
    retrieved_ids = [sc.chunk.chunk_id for sc in result.retrieved]

    return QuestionScore(
        id=golden["id"],
        question=golden["question"],
        answer=result.answer,
        faithfulness=faithfulness(result),
        answer_relevancy=answer_relevancy(result),
        context_precision=context_precision(retrieved_ids, relevant_ids),
        context_recall=context_recall(retrieved_ids, relevant_ids),
        citation_accuracy=citation_accuracy(result, relevant_ids),
        keyword_recall=keyword_recall(result.answer, golden["keywords"]),
        grounded=result.grounded,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s["total_s"],
        retrieved_chunk_ids=retrieved_ids,
        relevant_chunk_ids=list(relevant_ids),
    )
