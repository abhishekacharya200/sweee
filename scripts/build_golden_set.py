#!/usr/bin/env python3
"""Build eval/golden_set.json from src/ragcite/corpus_facts.py by actually
running the ingestion + chunking pipeline over data/corpus/ and locating,
for each Fact, the real chunk_id(s) whose text contains its statement.

This keeps the golden set honest: "relevant_chunk_ids" always reflects
what the current chunker actually produces, not a hand-typed guess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragcite.corpus_facts import DOCS, FACTS
from ragcite.ingest import chunk_document, parse_document

OUT_PATH = ROOT / "eval" / "golden_set.json"


def main() -> None:
    chunks_by_doc = {}
    for doc_id, meta in DOCS.items():
        path = ROOT / "data" / "corpus" / meta["filename"]
        parsed = parse_document(path)
        chunks_by_doc[doc_id] = chunk_document(parsed)

    entries = []
    unmatched = []
    for fact in FACTS:
        doc_chunks = chunks_by_doc[fact.doc_id]
        matches = [c for c in doc_chunks if fact.statement in c.text]
        if not matches:
            unmatched.append(fact.id)
            continue
        entries.append(
            {
                "id": fact.id,
                "doc_id": fact.doc_id,
                "doc_title": DOCS[fact.doc_id]["title"],
                "section": fact.section,
                "question": fact.question,
                "reference_answer": fact.answer,
                "keywords": list(fact.keywords),
                "relevant_chunk_ids": [c.chunk_id for c in matches],
            }
        )

    if unmatched:
        raise SystemExit(f"ERROR: {len(unmatched)} facts not found in chunked corpus: {unmatched}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} golden questions to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
