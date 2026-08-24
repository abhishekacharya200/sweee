from __future__ import annotations

from ragcite.models import ScoredChunk

SYSTEM_INSTRUCTIONS = (
    "You are a compliance assistant answering questions strictly from the "
    "provided excerpts of regulated documents (clinical guidelines, financial "
    "regulations, statutes, insurance policies, and SOPs).\n"
    "Rules:\n"
    "1. Use ONLY information contained in the numbered context passages below. "
    "Never use outside knowledge.\n"
    "2. Every sentence of your answer MUST end with a citation marker such as "
    "[1] or [2][3] referencing the passage(s) it is drawn from.\n"
    "3. If the passages do not contain a clear answer, reply exactly: "
    '"The provided documents do not contain a clear answer to this question." '
    "and cite nothing.\n"
    "4. Be concise: answer in 1-4 sentences, no preamble."
)


def format_context(chunks: list[ScoredChunk]) -> str:
    lines = []
    for i, sc in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({sc.chunk.doc_title}, p.{sc.chunk.page}) {sc.chunk.text}")
    return "\n\n".join(lines)


def build_citation_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    context = format_context(chunks)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
