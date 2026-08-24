from __future__ import annotations

from pydantic import BaseModel

from ragcite.models import AnswerResult


class QueryRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    doc_title: str
    page: int
    quote: str


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    doc_title: str
    section: str
    page: int
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    ungrounded_sentences: list[str]
    citations: list[CitationOut]
    retrieved: list[RetrievedChunkOut]
    provider: str
    model: str
    tokens: dict
    cost_usd: float
    latency_s: dict

    @classmethod
    def from_result(cls, result: AnswerResult) -> QueryResponse:
        return cls(
            question=result.question,
            answer=result.answer,
            grounded=result.grounded,
            ungrounded_sentences=result.ungrounded_sentences,
            citations=[
                CitationOut(marker=c.marker, chunk_id=c.chunk_id, doc_title=c.doc_title, page=c.page, quote=c.quote)
                for c in result.citations
            ],
            retrieved=[
                RetrievedChunkOut(
                    chunk_id=sc.chunk.chunk_id,
                    doc_title=sc.chunk.doc_title,
                    section=sc.chunk.section,
                    page=sc.chunk.page,
                    text=sc.chunk.text,
                    score=sc.score,
                )
                for sc in result.retrieved
            ],
            provider=result.provider,
            model=result.model,
            tokens=result.tokens,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
        )
