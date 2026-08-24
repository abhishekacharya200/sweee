"""Streamlit chat UI for citation-grounded RAG over the regulated corpus.

Run: streamlit run src/ragcite/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from ragcite.api.main import get_or_build_index
from ragcite.config import settings
from ragcite.generation.llm import build_llm_client
from ragcite.pipeline import RagPipeline

st.set_page_config(page_title="ragcite — Citation-Grounded RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading hybrid index (BM25 + dense, building on first run)...")
def get_pipeline() -> RagPipeline:
    store = get_or_build_index()
    return RagPipeline(store, llm=build_llm_client())


def render_answer(result) -> None:
    if result.grounded:
        st.success("All sentences are citation-grounded.")
    else:
        st.warning(f"{len(result.ungrounded_sentences)} sentence(s) lack a valid citation.")

    st.markdown(f"### Answer\n{result.answer}")

    if result.citations:
        st.markdown("### Citations")
        for c in result.citations:
            with st.expander(f"[{c.marker}] {c.doc_title} — p.{c.page}"):
                st.write(c.quote)

    cols = st.columns(4)
    cols[0].metric("Cost", f"${result.cost_usd:.6f}")
    cols[1].metric("Latency", f"{result.latency_s['total_s']:.2f}s")
    cols[2].metric("Tokens in/out", f"{result.tokens['input']}/{result.tokens['output']}")
    cols[3].metric("Provider", f"{result.provider}/{result.model}")

    with st.expander("Retrieved passages (hybrid search + rerank)"):
        for sc in result.retrieved:
            st.markdown(f"**{sc.chunk.doc_title}** — {sc.chunk.section}, p.{sc.chunk.page} "
                        f"(score {sc.score:.3f})")
            st.caption(sc.chunk.text)

    with st.expander("Latency breakdown"):
        st.json(result.latency_s)


def main() -> None:
    st.title("📚 Citation-Grounded RAG over a Regulated Corpus")
    st.caption(
        "Ask a question about the demo corpus (clinical guidelines, financial regulation, "
        "a data-privacy statute, an insurance policy, and a lab SOP). Every claim in the "
        "answer is cited back to a specific document and page."
    )

    pipeline = get_pipeline()

    with st.sidebar:
        st.header("Configuration")
        st.write(f"**LLM provider:** `{pipeline.llm.provider}`")
        st.write(f"**Model:** `{pipeline.llm.model}`")
        st.write(f"**Embedding backend:** `{settings.embedding_backend}`")
        st.write(f"**Rerank backend:** `{settings.rerank_backend}`")
        st.write(f"**Chunks indexed:** {len(pipeline.index.chunks)}")
        st.divider()
        st.caption(
            "Default provider is `mock` (offline, $0, deterministic). Set "
            "`LLM_PROVIDER=anthropic` or `openai` with an API key in `.env` for real generation."
        )
        if st.button("Rebuild index from data/corpus/"):
            get_pipeline.clear()
            st.rerun()

    example_questions = [
        "At what age should asymptomatic adults begin routine screening for type 2 diabetes?",
        "What is the minimum Common Equity Tier 1 capital ratio a covered depository institution must maintain?",
        "Within how many hours of discovering a data breach must the regulator be notified?",
        "What is the annual individual deductible under Group Plan 402?",
        "How often must a controlled document undergo periodic review at minimum?",
    ]

    if "history" not in st.session_state:
        st.session_state.history = []

    picked = st.selectbox("Try an example question", ["(pick one)"] + example_questions)
    default_q = picked if picked != "(pick one)" else ""
    question = st.text_input("Ask a question about the regulated corpus", value=default_q)

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving + generating..."):
            result = pipeline.answer(question)
        st.session_state.history.insert(0, result)

    for result in st.session_state.history:
        st.divider()
        st.markdown(f"**Q: {result.question}**")
        render_answer(result)


if __name__ == "__main__":
    main()
