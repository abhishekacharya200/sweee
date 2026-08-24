"""Paragraph-aware chunker: never splits a paragraph across chunk
boundaries, and stamps every chunk with the section heading and page
number it actually came from."""

from __future__ import annotations

import hashlib

from ragcite.config import settings
from ragcite.ingest.parsers import ParsedDocument
from ragcite.models import Chunk


def _word_count(text: str) -> int:
    return len(text.split())


def _chunk_id(doc_id: str, order: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{order}:{text}".encode()).hexdigest()[:10]
    return f"{doc_id}-{order:03d}-{digest}"


def chunk_document(
    parsed: ParsedDocument,
    max_tokens: int | None = None,
    overlap_paragraphs: int | None = None,
) -> list[Chunk]:
    max_tokens = max_tokens if max_tokens is not None else settings.chunk_max_tokens
    overlap_paragraphs = (
        overlap_paragraphs if overlap_paragraphs is not None else settings.chunk_overlap_paragraphs
    )

    chunks: list[Chunk] = []
    current_section = "Preamble"
    buffer: list[str] = []
    buffer_page: int | None = None
    order = 0

    def flush() -> None:
        nonlocal buffer, buffer_page, order
        if not buffer:
            return
        text = "\n\n".join(buffer)
        chunk = Chunk(
            chunk_id=_chunk_id(parsed.doc_id, order, text),
            doc_id=parsed.doc_id,
            doc_title=parsed.title,
            section=current_section,
            page=buffer_page or 1,
            text=text,
            order=order,
        )
        chunks.append(chunk)
        order += 1
        # keep the last N paragraphs as overlap context for the next chunk
        buffer = buffer[-overlap_paragraphs:] if overlap_paragraphs else []
        buffer_page = None

    for para in parsed.paragraphs:
        if para.is_heading:
            flush()
            current_section = para.text
            continue

        prospective = buffer + [para.text]
        if buffer and _word_count("\n\n".join(prospective)) > max_tokens:
            flush()
            buffer = [para.text]
        else:
            buffer.append(para.text)
        buffer_page = buffer_page or para.page

    flush()
    return chunks
