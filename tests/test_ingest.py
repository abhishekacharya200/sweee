from ragcite.corpus_facts import FACTS
from ragcite.ingest import parse_document


def test_parse_pdf_recovers_title_and_pages(corpus_dir):
    doc = parse_document(corpus_dir / "clinical_guideline_diabetes.pdf")
    assert doc.title == "Clinical Practice Guideline: Management of Type 2 Diabetes Mellitus"
    assert len(doc.pages) >= 5


def test_parse_docx_recovers_title_and_page_breaks(corpus_dir):
    doc = parse_document(corpus_dir / "insurance_policy_group402.docx")
    assert doc.title == "Comprehensive Health Insurance Policy Document, Group Plan 402"
    assert len(doc.pages) >= 5


def test_every_fact_survives_chunking(all_chunks):
    all_text = "\n".join(c.text for c in all_chunks)
    missing = [f.id for f in FACTS if f.statement not in all_text]
    assert missing == []


def test_chunks_have_valid_provenance(all_chunks):
    for c in all_chunks:
        assert c.page >= 1
        assert c.doc_title
        assert c.text.strip()
        assert c.chunk_id.startswith(c.doc_id)
