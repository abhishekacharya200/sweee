#!/usr/bin/env python3
"""Parse data/corpus/, chunk it, and build+persist the hybrid (BM25 + dense) index."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragcite.config import settings
from ragcite.index import IndexStore
from ragcite.ingest import chunk_document, parse_document


def main() -> None:
    chunks = []
    for path in sorted(settings.corpus_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        parsed = parse_document(path)
        doc_chunks = chunk_document(parsed)
        chunks.extend(doc_chunks)
        print(f"{path.name}: {len(parsed.paragraphs)} paragraphs, {len(parsed.pages)} pages, "
              f"{len(doc_chunks)} chunks")

    store = IndexStore()
    store.build(chunks)
    path = store.save()
    print(f"\nBuilt index with {len(chunks)} chunks -> {path.relative_to(ROOT)}")
    print(f"embedding_backend={store.embedding_backend_name} rerank_backend={store.rerank_backend_name}")


if __name__ == "__main__":
    main()
