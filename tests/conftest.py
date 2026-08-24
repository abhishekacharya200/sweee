from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from ragcite.corpus_facts import DOCS
from ragcite.index import IndexStore
from ragcite.ingest import chunk_document, parse_document


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    path = ROOT / "data" / "corpus"
    missing = [meta["filename"] for meta in DOCS.values() if not (path / meta["filename"]).exists()]
    if missing:
        pytest.skip(f"corpus not generated (run `make corpus` first): missing {missing}")
    return path


@pytest.fixture(scope="session")
def all_chunks(corpus_dir):
    chunks = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        parsed = parse_document(path)
        chunks.extend(chunk_document(parsed))
    return chunks


@pytest.fixture(scope="session")
def index_store(all_chunks) -> IndexStore:
    store = IndexStore()
    store.build(all_chunks)
    return store


@pytest.fixture(scope="session")
def world():
    """The seeded reconciliation world shared by the Project B tests."""
    from reconagent.world import build_world

    return build_world(20260301)


@pytest.fixture
def ledger(world):
    """A fresh clone per test — tool calls mutate the store."""
    store, _ = world
    return store.clone()


@pytest.fixture
def registry(ledger):
    from reconagent.tools.registry import build_registry

    return build_registry(ledger)
