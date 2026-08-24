"""Parse an LLM's raw citation-marked answer into structured Citations and
flag any sentence that is not grounded by a citation marker."""

from __future__ import annotations

import re

from ragcite.models import Citation, ScoredChunk

_CITATION_MARKER = re.compile(r"\[(\d+)\]")
_TRAILING_MARKERS = re.compile(r"\s*\[\d+\]")

REFUSAL_PHRASE = "the provided documents do not contain a clear answer"


def split_sentences(answer_text: str) -> list[str]:
    """Split into sentences, keeping trailing citation markers (e.g.
    "...at all times. [1]") attached to the sentence they ground rather
    than letting them start the next sentence."""
    text = answer_text.strip()
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        is_decimal_point = (
            text[i] == "."
            and 0 < i < n - 1
            and text[i - 1].isdigit()
            and text[i + 1].isdigit()
        )
        if text[i] in ".!?" and not is_decimal_point:
            j = i + 1
            while True:
                m = _TRAILING_MARKERS.match(text, j)
                if not m:
                    break
                j = m.end()
            sentences.append(text[start:j].strip())
            while j < n and text[j].isspace():
                j += 1
            start = j
            i = j
        else:
            i += 1
    if start < n:
        sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def extract_citations(answer_text: str, chunks: list[ScoredChunk]) -> list[Citation]:
    citations: list[Citation] = []
    seen_markers: set[int] = set()
    for match in _CITATION_MARKER.finditer(answer_text):
        marker = int(match.group(1))
        if marker in seen_markers:
            continue
        seen_markers.add(marker)
        if 1 <= marker <= len(chunks):
            chunk = chunks[marker - 1].chunk
            citations.append(
                Citation(marker=marker, chunk_id=chunk.chunk_id, doc_title=chunk.doc_title, page=chunk.page,
                          quote=chunk.text[:200])
            )
    return sorted(citations, key=lambda c: c.marker)


def check_grounding(answer_text: str, num_context_chunks: int) -> tuple[bool, list[str]]:
    """A sentence is "grounded" if it carries at least one in-range
    citation marker. Refusal answers are trivially grounded (they cite
    nothing on purpose)."""
    if REFUSAL_PHRASE in answer_text.lower():
        return True, []

    ungrounded = []
    for sentence in split_sentences(answer_text):
        markers = [int(m) for m in _CITATION_MARKER.findall(sentence)]
        valid = [m for m in markers if 1 <= m <= num_context_chunks]
        if not valid:
            ungrounded.append(sentence)
    return (len(ungrounded) == 0), ungrounded


def used_chunk_ids(citations: list[Citation]) -> list[str]:
    seen: list[str] = []
    for c in citations:
        if c.chunk_id not in seen:
            seen.append(c.chunk_id)
    return seen
