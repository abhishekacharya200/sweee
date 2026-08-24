from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ragcite.api.schemas import QueryRequest, QueryResponse
from ragcite.config import settings
from ragcite.generation.llm import build_llm_client
from ragcite.index.store import IndexStore
from ragcite.ingest import chunk_document, parse_document
from ragcite.pipeline import RagPipeline

_state: dict = {}


def build_index_from_corpus() -> IndexStore:
    chunks = []
    for path in sorted(settings.corpus_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        parsed = parse_document(path)
        chunks.extend(chunk_document(parsed))
    store = IndexStore()
    store.build(chunks)
    store.save()
    return store


def get_or_build_index() -> IndexStore:
    index_path = settings.index_dir / "index.pkl"
    if index_path.exists():
        return IndexStore.load()
    return build_index_from_corpus()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_or_build_index()
    _state["pipeline"] = RagPipeline(store, llm=build_llm_client())
    yield
    _state.clear()


app = FastAPI(
    title="ragcite: Citation-Grounded RAG over a Regulated Corpus",
    version="0.1.0",
    lifespan=lifespan,
)


def get_pipeline() -> RagPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not loaded yet")
    return pipeline


@app.get("/health")
def health() -> dict:
    pipeline = _state.get("pipeline")
    return {
        "status": "ok" if pipeline else "starting",
        "provider": pipeline.llm.provider if pipeline else None,
        "model": pipeline.llm.model if pipeline else None,
        "n_chunks": len(pipeline.index.chunks) if pipeline else 0,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    pipeline = get_pipeline()
    result = pipeline.answer(request.question)
    return QueryResponse.from_result(result)


@app.post("/ingest")
def ingest() -> dict:
    """Re-parse data/corpus/ and rebuild the hybrid index from scratch."""
    store = build_index_from_corpus()
    _state["pipeline"] = RagPipeline(store, llm=build_llm_client())
    return {"status": "rebuilt", "n_chunks": len(store.chunks)}


@app.get("/corpus")
def corpus() -> dict:
    pipeline = get_pipeline()
    docs: dict[str, dict] = {}
    for chunk in pipeline.index.chunks.values():
        d = docs.setdefault(chunk.doc_id, {"doc_title": chunk.doc_title, "n_chunks": 0, "pages": 0})
        d["n_chunks"] += 1
        d["pages"] = max(d["pages"], chunk.page)
    return {"documents": docs, "n_chunks": len(pipeline.index.chunks)}
