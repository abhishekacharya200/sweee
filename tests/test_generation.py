from ragcite.generation.answer import check_grounding, extract_citations, split_sentences
from ragcite.generation.llm import MockLLM
from ragcite.models import Chunk, ScoredChunk


def _scored_chunk(text: str, chunk_id: str = "c1") -> ScoredChunk:
    chunk = Chunk(chunk_id=chunk_id, doc_id="doc", doc_title="Doc", section="Sec", page=1, text=text, order=0)
    return ScoredChunk(chunk=chunk, score=0.5)


def test_split_sentences_keeps_decimal_points_intact():
    text = "The ratio is 4.5 percent. [1] The buffer is 2.5 percent. [2]"
    sentences = split_sentences(text)
    assert sentences == ["The ratio is 4.5 percent. [1]", "The buffer is 2.5 percent. [2]"]


def test_split_sentences_attaches_marker_to_preceding_sentence():
    text = "First fact here. [1] Second fact here. [2]"
    sentences = split_sentences(text)
    assert sentences[0].endswith("[1]")
    assert sentences[1].endswith("[2]")


def test_extract_citations_maps_markers_to_chunks_in_order():
    chunks = [_scored_chunk("alpha text", "c1"), _scored_chunk("beta text", "c2")]
    citations = extract_citations("Alpha claim [1]. Beta claim [2].", chunks)
    assert [c.chunk_id for c in citations] == ["c1", "c2"]


def test_check_grounding_flags_sentence_without_citation():
    grounded, ungrounded = check_grounding("Claim one [1]. Claim two with no citation.", num_context_chunks=1)
    assert grounded is False
    assert len(ungrounded) == 1


def test_check_grounding_passes_refusal_with_no_citations():
    grounded, ungrounded = check_grounding(
        "The provided documents do not contain a clear answer to this question.", num_context_chunks=3
    )
    assert grounded is True
    assert ungrounded == []


def test_mock_llm_is_deterministic_and_cites_top_chunk():
    chunks = [
        _scored_chunk("Adults aged 35 or older should be screened for diabetes every 3 years.", "c1"),
        _scored_chunk("Unrelated passage about insurance deductibles.", "c2"),
    ]
    llm = MockLLM()
    r1 = llm.generate("At what age should adults be screened for diabetes?", chunks)
    r2 = llm.generate("At what age should adults be screened for diabetes?", chunks)
    assert r1.text == r2.text
    assert "[1]" in r1.text
    assert r1.provider == "mock"


def test_mock_llm_refuses_when_nothing_relevant_is_retrieved():
    llm = MockLLM()
    r = llm.generate("Anything?", [])
    assert "do not contain a clear answer" in r.text
