from ragcite.ingest.chunker import chunk_document
from ragcite.ingest.parsers import ParsedDocument, ParsedParagraph


def _doc(paragraphs):
    return ParsedDocument(doc_id="doc", title="Doc Title", source_path="doc.pdf", paragraphs=paragraphs, pages=[])


def test_heading_starts_new_section_and_is_excluded_from_chunk_text():
    paragraphs = [
        ParsedParagraph(text="Section One", is_heading=True, page=1),
        ParsedParagraph(text="First body sentence.", is_heading=False, page=1),
        ParsedParagraph(text="Section Two", is_heading=True, page=2),
        ParsedParagraph(text="Second body sentence.", is_heading=False, page=2),
    ]
    chunks = chunk_document(_doc(paragraphs), max_tokens=100, overlap_paragraphs=0)
    assert [c.section for c in chunks] == ["Section One", "Section Two"]
    assert "Section One" not in chunks[0].text
    assert chunks[0].page == 1
    assert chunks[1].page == 2


def test_never_splits_a_single_paragraph_mid_text():
    long_para = "word " * 500
    paragraphs = [ParsedParagraph(text=long_para.strip(), is_heading=False, page=1)]
    chunks = chunk_document(_doc(paragraphs), max_tokens=10, overlap_paragraphs=0)
    assert len(chunks) == 1
    assert chunks[0].text == long_para.strip()


def test_respects_max_tokens_budget_across_paragraphs():
    paragraphs = [
        ParsedParagraph(text="one two three four five", is_heading=False, page=1),
        ParsedParagraph(text="six seven eight nine ten", is_heading=False, page=1),
        ParsedParagraph(text="eleven twelve thirteen fourteen fifteen", is_heading=False, page=1),
    ]
    chunks = chunk_document(_doc(paragraphs), max_tokens=10, overlap_paragraphs=0)
    assert len(chunks) == 2
    assert chunks[0].text == "one two three four five\n\nsix seven eight nine ten"
    assert chunks[1].text == "eleven twelve thirteen fourteen fifteen"


def test_stable_chunk_ids_are_deterministic():
    paragraphs = [ParsedParagraph(text="Stable text.", is_heading=False, page=1)]
    c1 = chunk_document(_doc(paragraphs))
    c2 = chunk_document(_doc(paragraphs))
    assert c1[0].chunk_id == c2[0].chunk_id
