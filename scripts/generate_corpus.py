#!/usr/bin/env python3
"""Render the synthetic regulated corpus (data/corpus/*.pdf, *.docx) from
src/ragcite/corpus_facts.py. Every paragraph in the generated documents is
exactly one Fact.statement, so downstream chunking/citation/eval code can be
verified against ground truth rather than eyeballed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docx import Document as DocxDocument
from docx.enum.text import WD_BREAK
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from ragcite.corpus_facts import DOCS, facts_for_doc, sections_for_doc

OUT_DIR = ROOT / "data" / "corpus"


def render_pdf(doc_id: str) -> Path:
    meta = DOCS[doc_id]
    out_path = OUT_DIR / meta["filename"]
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=16, spaceAfter=6
    )
    domain_style = ParagraphStyle(
        "Domain", parent=styles["Normal"], fontSize=10, textColor="#555555", spaceAfter=24
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], spaceBefore=18, spaceAfter=10
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=10
    )

    story = [
        Paragraph(meta["title"], title_style),
        Paragraph(f"Domain: {meta['domain']} &mdash; Synthetic document generated for demo purposes.", domain_style),
    ]

    sections = sections_for_doc(doc_id)
    for i, section in enumerate(sections):
        if i > 0:
            story.append(PageBreak())
        story.append(Paragraph(section, section_style))
        for fact in facts_for_doc(doc_id):
            if fact.section != section:
                continue
            story.append(Paragraph(fact.statement, body_style))
        story.append(Spacer(1, 0.1 * inch))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=meta["title"],
    )
    doc.build(story)
    return out_path


def render_docx(doc_id: str) -> Path:
    meta = DOCS[doc_id]
    out_path = OUT_DIR / meta["filename"]
    document = DocxDocument()

    document.add_heading(meta["title"], level=0)
    p = document.add_paragraph(
        f"Domain: {meta['domain']} — Synthetic document generated for demo purposes."
    )
    p.italic = True

    sections = sections_for_doc(doc_id)
    for i, section in enumerate(sections):
        if i > 0:
            page_break_para = document.add_paragraph()
            page_break_para.add_run().add_break(WD_BREAK.PAGE)
        document.add_heading(section, level=1)
        for fact in facts_for_doc(doc_id):
            if fact.section != section:
                continue
            document.add_paragraph(fact.statement)

    document.save(str(out_path))
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, meta in DOCS.items():
        if meta["kind"] == "pdf":
            path = render_pdf(doc_id)
        else:
            path = render_docx(doc_id)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
