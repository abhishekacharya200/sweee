from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ragcite.config import settings
from ragcite.cost import count_tokens
from ragcite.generation.prompt import build_citation_prompt
from ragcite.index.sparse import tokenize
from ragcite.models import ScoredChunk

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    provider: str
    model: str


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, question: str, chunks: list[ScoredChunk]) -> LLMResponse: ...


def _sentences_of(text: str) -> list[str]:
    sentences = []
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return sentences or [text.strip()]


def _best_sentence(question: str, text: str) -> tuple[str, int]:
    query_terms = set(tokenize(question))
    best, best_score = None, -1
    for sentence in _sentences_of(text):
        score = len(query_terms & set(tokenize(sentence)))
        if score > best_score:
            best, best_score = sentence, score
    return (best or text.strip()), max(best_score, 0)


class MockLLM:
    """Deterministic, offline, zero-cost extractive answerer.

    For each of the top retrieved passages, it pulls the single sentence
    with the highest lexical overlap with the question and cites it. This
    is what powers the CI eval gate: no API key, no network call, no
    non-determinism -- while still exercising real citation-grounding
    logic end to end.
    """

    provider = "mock"
    model = "extractive-v1"
    min_confidence = 0.15

    def generate(self, question: str, chunks: list[ScoredChunk]) -> LLMResponse:
        if not chunks or chunks[0].score < self.min_confidence:
            text = "The provided documents do not contain a clear answer to this question."
            return LLMResponse(text=text, input_tokens=count_tokens(question), output_tokens=count_tokens(text),
                                provider=self.provider, model=self.model)

        top = chunks[: min(3, len(chunks))]
        candidates = [(*_best_sentence(question, sc.chunk.text), sc) for sc in top]

        parts = []
        top_score = candidates[0][1]
        for i, (sentence, kw_score, _sc) in enumerate(candidates, start=1):
            # Only cite additional passages once the top hit is picked; a
            # second/third passage is included only if it is nearly as
            # relevant, so single-fact questions get a focused one-citation
            # answer instead of padding with tangential context.
            if i > 1 and (kw_score == 0 or kw_score < top_score):
                break
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            parts.append(f"{sentence} [{i}]")
        text = " ".join(parts)

        prompt = build_citation_prompt(question, chunks)
        return LLMResponse(
            text=text,
            input_tokens=count_tokens(prompt),
            output_tokens=count_tokens(text),
            provider=self.provider,
            model=self.model,
        )


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, question: str, chunks: list[ScoredChunk]) -> LLMResponse:
        prompt = build_citation_prompt(question, chunks)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = response.usage
        return LLMResponse(
            text=text.strip(),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            provider=self.provider,
            model=self.model,
        )


class OpenAILLM:
    provider = "openai"

    def __init__(self, model: str, api_key: str):
        import openai

        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, question: str, chunks: list[ScoredChunk]) -> LLMResponse:
        prompt = build_citation_prompt(question, chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text.strip(),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            provider=self.provider,
            model=self.model,
        )


def build_llm_client(provider: str | None = None) -> LLMClient:
    provider = provider or settings.resolve_llm_provider()
    if provider == "mock":
        return MockLLM()
    if provider == "anthropic":
        return AnthropicLLM(settings.anthropic_model, settings.anthropic_api_key or "")
    if provider == "openai":
        return OpenAILLM(settings.openai_model, settings.openai_api_key or "")
    raise ValueError(f"Unknown LLM provider: {provider}")
