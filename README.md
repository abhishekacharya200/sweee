# ragcite — Citation-Grounded RAG over a Regulated Corpus

A retrieval-augmented QA system for domains where being wrong is unacceptable:
clinical guidelines, financial regulations, legal statutes, and insurance
policy documents. Every answer is grounded in specific, numbered citations
back to a source document and page, and every change to the pipeline is
checked against a RAGAS-style eval gate before it can ship.

**Measured on the shipped 67-question golden set, offline mock LLM, $0 cost:**

> **86.0% faithfulness, 91.3% citation accuracy, 100% context recall, $0.0000 and 0.008s per query.**

(Numbers are reproduced by `make eval` — see [Evaluation](#evaluation) for how
this compares to running with a real LLM, and why $0/0.008s is a fair
baseline rather than a marketing number.)

## Why this exists

The brief: pick a domain where wrong answers are costly, and build a RAG
system that never states a claim without pointing to the exact passage it
came from. That constrains the whole design:

| Requirement | Where it lives |
|---|---|
| PDF/DOCX parsing pipeline | `src/ragcite/ingest/` — page- and section-aware parsers for both formats |
| Hybrid search + reranking | `src/ragcite/index/` — BM25 + dense embeddings fused via Reciprocal Rank Fusion, then reranked |
| Citations pointing to evidence | `src/ragcite/generation/` — every sentence must carry a `[n]` marker resolved to a real chunk + page |
| RAGAS-style eval table | `src/ragcite/eval/` — faithfulness, answer relevancy, context precision/recall, citation accuracy |
| Cost + latency per query | `src/ragcite/cost.py`, tracked on every `AnswerResult` and aggregated in the eval report |
| Streamlit frontend | `src/ragcite/app/streamlit_app.py` |
| Docker + CI with eval gate | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Architecture

```
data/corpus/*.pdf,*.docx
        │  parse (pypdf / python-docx, page+section aware)
        ▼
   ingest.chunk_document        (paragraph-aware, never splits mid-sentence)
        │
        ▼
   index.IndexStore.build       ── BM25Okapi (sparse)
        │                       └─ TF-IDF+SVD or sentence-transformers (dense)
        ▼
   IndexStore.search(query)
        │  RRF-fuse sparse + dense rankings → rerank top-N
        ▼
   pipeline.RagPipeline.answer
        │  LLM generates answer with inline [n] citation markers
        │  answer.py validates every sentence resolves to a retrieved chunk
        ▼
   AnswerResult  (answer, citations, cost_usd, latency_s, grounded)
        │
        ├──► FastAPI  /query  /health  /ingest  /corpus
        └──► Streamlit chat UI (citations, retrieved passages, cost/latency)

   eval.run_eval(golden_set) → faithfulness / citation accuracy / context
   recall / … → eval.gate() → CI fails the build if scores regress
```

### Design choices that matter

- **Everything runs offline with zero API keys by default.** Embeddings
  default to a TF-IDF + truncated-SVD backend (`EMBEDDING_BACKEND=local`),
  reranking defaults to a lexical TF-IDF + term-overlap reranker
  (`RERANK_BACKEND=lexical`), and generation defaults to a deterministic
  extractive `mock` LLM. This is what makes the CI eval gate possible without
  secrets in the repo, and it's a legitimate on-prem-friendly config, not just
  a demo shortcut — swap in `sentence-transformers`/`cross-encoder` or a real
  LLM provider via `.env` (see below) without touching any other code.
- **The golden set is generated, not hand-typed.** `src/ragcite/corpus_facts.py`
  is the single source of truth: each of 67 `Fact` records becomes one
  paragraph in the generated corpus *and* one golden question
  (`scripts/build_golden_set.py` looks up the real chunk_id(s) that fact ended
  up in after chunking). This means faithfulness/citation-accuracy numbers are
  checked against ground truth that's actually grounded in the corpus, not
  eyeballed.
- **Citation grounding is enforced structurally, not just prompted.**
  `generation/answer.py` parses every answer into sentences and rejects any
  sentence lacking a `[n]` marker that resolves to a real retrieved chunk
  (`AnswerResult.grounded`). A CI-visible refusal phrase is used when context
  is insufficient, and the mock LLM includes it automatically when the top
  retrieval score is below a confidence floor.
- **Cost/latency are tracked per-stage on every query** — sparse search,
  dense search, fusion, rerank, generation — via `AnswerResult.latency_s`, and
  cost via an explicit, editable `$/1M tokens` pricing table
  (`src/ragcite/cost.py`) rather than hidden inside a provider SDK.

## Quickstart

```bash
make setup                 # venv + install
make corpus                # render the synthetic regulated corpus (PDF/DOCX)
make golden                # build eval/golden_set.json from the corpus
make index                 # parse + chunk + build the hybrid index
make eval                  # run the RAGAS-style eval gate (mock LLM, $0, offline)
make api                   # FastAPI on :8000
make ui                    # Streamlit on :8501
```

Or with Docker:

```bash
docker compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

### Using a real LLM

Copy `.env.example` to `.env` and set:

```bash
LLM_PROVIDER=anthropic       # or "openai"
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_MODEL=claude-sonnet-5
```

If the key is missing, `Settings.resolve_llm_provider()` silently falls back
to `mock` — the app never crashes for lack of a key, it just runs in $0
offline mode.

## The demo corpus

Five synthetic-but-realistic regulated documents, generated from
`src/ragcite/corpus_facts.py` (67 discrete, checkable facts) so every golden
answer is traceable to an exact source paragraph:

| Document | Format | Domain |
|---|---|---|
| Clinical Practice Guideline: Management of Type 2 Diabetes Mellitus | PDF | Life sciences / clinical |
| Regulation FR-14: Capital Adequacy and Liquidity Requirements | PDF | Financial regulation |
| Data Privacy Compliance Statute, Sections 101–140 | PDF | Legal / statute |
| Comprehensive Health Insurance Policy, Group Plan 402 | DOCX | Insurance policy |
| SOP-QA-011: Document Control and Change Management | DOCX | Life sciences / quality SOP |

Drop real regulatory PDFs/DOCX files into `data/corpus/` and rerun
`make index` — the parsers use generic heuristics (heading detection,
page/paragraph tracking), not anything specific to the synthetic corpus.

## Evaluation

`make eval` runs all 67 golden questions through the full pipeline and
scores them with locally-implemented, RAGAS-inspired metrics (no `ragas`
dependency, so eval stays offline and deterministic in CI):

| Metric | What it measures | This run |
|---|---|---|
| Faithfulness | lexical support between each cited sentence and the chunk(s) it cites | 0.860 |
| Answer Relevancy | similarity between the answer and the question | 0.429 |
| Context Precision | fraction of retrieved chunks that are actually relevant | 0.200 |
| Context Recall | fraction of the truly relevant chunk(s) that were retrieved | 1.000 |
| Citation Accuracy | fraction of cited chunks that are actually relevant evidence | 0.913 |
| Grounded Rate | fraction of answers where every sentence carries a valid citation | 1.000 |

Full per-question breakdown: [`eval/report.md`](eval/report.md) /
[`eval/results.json`](eval/results.json) (regenerated by every `make eval` /
CI run).

**Reading context precision honestly:** it's low (0.20) by construction, not
by bug — `top_k_final=5` retrieves 5 chunks per query so recall stays at
100% even on single-fact questions where only 1 chunk is truly relevant
(1/5 = 0.20 is close to the ceiling here). Tune `TOP_K_FINAL` down if your
corpus has enough near-duplicate chunks that precision matters more than
recall for your use case.

**Eval gate** (`ragcite.eval.runner.DEFAULT_GATE_THRESHOLDS`, enforced by
`scripts/run_eval.py` exit code and wired into CI):

```python
{"faithfulness": 0.55, "citation_accuracy": 0.85, "context_recall": 0.70, "grounded_rate": 0.90}
```

### Scaling past 67 questions

The brief's headline references a 120-question golden set. Add more `Fact`
entries to `corpus_facts.py` (or more documents to `DOCS`), rerun
`make corpus golden index eval` — the golden set, chunk citations, and eval
table all regenerate from that one source of truth with no manual
relabeling.

### Running eval against a real LLM

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... python scripts/run_eval.py
```

This exercises the same gate against `build_citation_prompt()` and a real
model instead of the extractive mock — expect materially different
cost/latency numbers and (usually) higher answer relevancy, at nonzero $/query.

## Project layout

```
src/ragcite/
  corpus_facts.py     single source of truth: facts → corpus paragraphs → golden questions
  config.py           Settings (env-driven: providers, backends, top_k, chunk size)
  models.py           Chunk / ScoredChunk / Citation / AnswerResult
  ingest/             PDF & DOCX parsing (page+heading aware) and paragraph-safe chunking
  index/              BM25, dense embeddings (local TF-IDF+SVD or sentence-transformers),
                       RRF fusion, reranking (lexical or cross-encoder)
  generation/         prompt construction, LLM clients (mock/anthropic/openai), citation parsing
  eval/               RAGAS-style metrics, golden-set runner, markdown/json reports
  api/                FastAPI app (/query /health /ingest /corpus)
  app/                Streamlit chat UI
  cost.py             token counting + $ pricing table
  pipeline.py          RagPipeline: retrieval → generation → grounding check → AnswerResult
scripts/              generate_corpus, build_golden_set, build_index, run_eval
tests/                pytest unit tests (ingest, chunker, hybrid search, generation, eval metrics)
data/corpus/          generated PDF/DOCX documents (checked in so the repo is runnable as-is)
eval/                 golden_set.json + generated report.md / results.json
```

## Testing

```bash
make test    # pytest — ingest, chunker, hybrid search, citation grounding, eval metrics
make lint    # ruff
```

CI (`.github/workflows/ci.yml`) runs lint → tests → corpus generation →
golden-set build → index build → **eval gate** → Docker image builds (api +
ui targets) on every push/PR, so a change that drops citation accuracy or
faithfulness below threshold fails the build before it can merge.

## Known limitations

- The demo corpus is synthetic (generated, not scraped from real regulatory
  filings) — this is by design, both for licensing cleanliness and so the
  golden set can be mechanically verified against ground truth.
- The default `local`/`lexical` backends are lexical/co-occurrence based, not
  neural. They're deliberately chosen for zero-dependency offline operation;
  swap to `sentence-transformers` embeddings and a `cross-encoder` reranker
  (`pip install -e ".[embeddings]"`, set `EMBEDDING_BACKEND` /
  `RERANK_BACKEND` in `.env`) for materially better retrieval quality against
  real-world, higher-volume corpora.
- Faithfulness is a lexical-overlap heuristic, not an NLI/LLM-judge score —
  documented in `eval/metrics.py`. It correlates with grounding quality but
  isn't a substitute for human review before any regulated deployment.
- Out-of-corpus question refusal uses a single confidence-score threshold on
  the mock LLM (`MockLLM.min_confidence`); it isn't perfectly precise on
  adversarial queries. A real LLM provider with the citation-enforcing prompt
  in `generation/prompt.py` handles this far better.
