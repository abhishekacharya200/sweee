from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A retrievable unit of text with provenance for citation."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    page: int
    text: str
    order: int

    def citation_label(self) -> str:
        return f"{self.doc_title}, p.{self.page}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "page": self.page,
            "text": self.text,
            "order": self.order,
        }

    @staticmethod
    def from_dict(d: dict) -> Chunk:
        return Chunk(**d)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    rank_scores: dict = field(default_factory=dict)


@dataclass
class Citation:
    marker: int
    chunk_id: str
    doc_title: str
    page: int
    quote: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[ScoredChunk]
    used_chunk_ids: list[str]
    provider: str
    model: str
    latency_s: dict
    tokens: dict
    cost_usd: float
    grounded: bool
    ungrounded_sentences: list[str]
