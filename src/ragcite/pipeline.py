from __future__ import annotations

import time

from ragcite.cost import estimate_cost_usd
from ragcite.generation.answer import check_grounding, extract_citations, used_chunk_ids
from ragcite.generation.llm import LLMClient, build_llm_client
from ragcite.index.store import IndexStore
from ragcite.models import AnswerResult


class RagPipeline:
    def __init__(self, index: IndexStore, llm: LLMClient | None = None):
        self.index = index
        self.llm = llm or build_llm_client()

    def answer(self, question: str) -> AnswerResult:
        retrieved, retrieval_latency = self.index.search(question)

        t0 = time.perf_counter()
        response = self.llm.generate(question, retrieved)
        generation_latency = time.perf_counter() - t0

        citations = extract_citations(response.text, retrieved)
        grounded, ungrounded_sentences = check_grounding(response.text, len(retrieved))
        cost = estimate_cost_usd(response.provider, response.model, response.input_tokens, response.output_tokens)

        latency = dict(retrieval_latency)
        latency["generation_s"] = generation_latency
        latency["total_s"] = retrieval_latency["retrieval_total_s"] + generation_latency

        return AnswerResult(
            question=question,
            answer=response.text,
            citations=citations,
            retrieved=retrieved,
            used_chunk_ids=used_chunk_ids(citations),
            provider=response.provider,
            model=response.model,
            latency_s=latency,
            tokens={"input": response.input_tokens, "output": response.output_tokens},
            cost_usd=cost,
            grounded=grounded,
            ungrounded_sentences=ungrounded_sentences,
        )
