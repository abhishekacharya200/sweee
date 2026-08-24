# Two systems for domains where a wrong answer is expensive

| | What it is | Headline |
|---|---|---|
| **[Project A — `ragcite`](#project-a--ragcite-citation-grounded-rag-over-a-regulated-corpus)** | Citation-grounded RAG over clinical, financial and legal documents. Every sentence carries a `[n]` marker resolved to a real page. | 86.0% faithfulness, 91.3% citation accuracy, 100% context recall on 67 golden questions |
| **[Project B — `reconagent`](#project-b--reconagent-a-reconciliation-agent-with-a-real-tool-surface)** | An agent that triages a ledger reconciliation queue across two data sources, over 12 Pydantic-typed tools that are also served as an MCP server. | 100% task success vs a 25.7% baseline on 70 tasks; 90% under injected tool failures, with zero misbookings |

Both run offline, deterministically, at $0, with no API key — which is what
lets both eval gates run in CI. Both take a real model instead via `.env`.

```bash
make setup
make corpus golden index eval    # Project A: build the corpus and run the RAG eval gate
make agent-eval                  # Project B: run the agent eval gate
make test lint
```

Full write-ups: [SDK trade-off analysis](docs/AGENT_SDK_TRADEOFFS.md) ·
[RAG eval report](eval/report.md) · [agent eval report](eval/agent/report.md)

---

## Project A — ragcite: citation-grounded RAG over a regulated corpus

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

### Why this exists

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

### Architecture

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

#### Design choices that matter

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

### Quickstart

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

#### Using a real LLM

Copy `.env.example` to `.env` and set:

```bash
LLM_PROVIDER=anthropic       # or "openai"
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_MODEL=claude-sonnet-5
```

If the key is missing, `Settings.resolve_llm_provider()` silently falls back
to `mock` — the app never crashes for lack of a key, it just runs in $0
offline mode.

### The demo corpus

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

### Evaluation

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

#### Scaling past 67 questions

The brief's headline references a 120-question golden set. Add more `Fact`
entries to `corpus_facts.py` (or more documents to `DOCS`), rerun
`make corpus golden index eval` — the golden set, chunk citations, and eval
table all regenerate from that one source of truth with no manual
relabeling.

#### Running eval against a real LLM

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... python scripts/run_eval.py
```

This exercises the same gate against `build_citation_prompt()` and a real
model instead of the extractive mock — expect materially different
cost/latency numbers and (usually) higher answer relevancy, at nonzero $/query.

### Project layout

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

### Testing

```bash
make test    # pytest — 109 tests across both projects
make lint    # ruff
```

CI (`.github/workflows/ci.yml`) runs lint → tests → corpus generation →
golden-set build → index build → **RAG eval gate** → tool-surface export check
→ **agent eval gate** → Docker image builds (api + ui targets) on every
push/PR. A change that drops citation accuracy or faithfulness below
threshold, or one that makes the agent start booking guesses, fails the build
before it can merge.

### Known limitations

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

---

## Project B — reconagent: a reconciliation agent with a real tool surface

Month-end reconciliation between an AR ledger and a bank statement feed leaves
a queue of exceptions: credits that did not match, invoices that did not
settle, amounts that disagree. Somebody has to work that queue and decide, for
each one, whether it is a clean match, a bank fee, an FX residual, a short
payment, a duplicate — or something a human needs to look at.

`reconagent` is an agent that does that, over a real tool surface, with the
loop bounded and the failure behaviour defined.

**Measured on the shipped 70-task queue, offline, deterministic, $0:**

> **100% task success against a 25.7% baseline; 90% under 12% injected tool
> failures, with zero misbookings and zero exceptions left unowned.**

Reproduced by `make agent-eval` → [`eval/agent/report.md`](eval/agent/report.md).

### What it does

```
        AR ledger (invoices)        Bank feed (credits)
                 \                        /
                  \                      /
                   nightly matcher → exception queue (70 open)
                                            │
                            ┌───────────────┴───────────────┐
                            │  one episode per exception    │
                            │                               │
                     gather evidence ──► 10 read tools      │
                            │            (search, fuzzy-match memo,
                            │             FX rate, contract terms,
                            │             materiality thresholds)
                            ▼                               │
                     decide disposition                     │
                            │                               │
              ┌─────────────┴─────────────┐                 │
              ▼                           ▼                 │
     record_resolution              escalate_exception       │
     (match / bank_fee /            (evidence missing,       │
      fx_variance /                  contradictory, or       │
      short_payment /                above the write-off     │
      duplicate_payment)             limit)                  │
              └─────────────┬─────────────┘                 │
                            ▼                               │
                  exactly one terminal write  ◄──────────────┘
                  (guaranteed, even if every budget blows)
```

Every exception is generated from one of eight archetypes whose correct
disposition is known by construction, so eval scoring is exact rather than
eyeballed — the same trick Project A uses for its RAG golden set.

| Archetype | Correct answer | What the agent has to notice |
|---|---|---|
| `clean_match` | `match_payment` | The memo reference is mangled, but amount and currency agree |
| `bank_fee` | `bank_fee` | The shortfall equals the *contracted* wire fee for a customer who bears fees |
| `fx_variance` | `fx_variance` | Invoice in EUR/GBP, cash in USD; the residual at the value-date rate is inside tolerance |
| `short_payment` | `short_payment` | Shortfall is unexplained but under the write-off limit |
| `duplicate_payment` | `duplicate_payment` | The invoice was already settled by an earlier credit inside the lookback window |
| `material_short_payment` | **escalate** | Same shape as `short_payment`, but over the limit — a human credit decision |
| `ambiguous_multi_match` | **escalate** | No usable reference, and two customers have invoices at that exact amount |
| `no_matching_invoice` | **escalate** | The referenced invoice does not exist and no amount matches |

The last three are 34% of the queue, so neither "always escalate" nor "never
escalate" is a winning strategy — which is the point.

### The tool surface

Twelve tools, declared once as Pydantic models in `src/reconagent/tools/`:

| Tool | Terminal | Writes | Purpose |
|---|---|---|---|
| `list_open_exceptions` | no | no | List exceptions still awaiting a disposition, oldest first |
| `get_exception` | no | no | Fetch one exception with whichever invoice and credit the matcher already linked |
| `get_invoice` | no | no | Fetch one invoice from the AR ledger by id |
| `get_bank_transaction` | no | no | Fetch one bank statement line by id |
| `search_invoices` | no | no | Search by customer, amount (with tolerance), currency, status or issue date |
| `search_bank_transactions` | no | no | Search by memo text, amount, currency, value date or existing match |
| `extract_invoice_reference` | no | no | Mine a memo for references and rank invoice ids, tolerating missing separators and OCR digit look-alikes |
| `get_customer_policy` | no | no | Contract terms: payment terms, settlement currency, who bears wire fees, the fee amount |
| `get_accounting_policy` | no | no | Entity thresholds: FX tolerance, fee variance, write-off limit, duplicate lookback |
| `get_fx_rate` | no | no | Published rate for a pair on a date (business days only) |
| `record_resolution` | **yes** | **yes** | Close an exception by booking a disposition |
| `escalate_exception` | **yes** | **yes** | Hand an exception to a human queue with a reason |

Regenerate this table with `make agent-tools`; dump the raw schemas with
`python -m reconagent.cli tools --json`.

### One registry, three consumers

`TOOL_SPECS` in `tools/registry.py` is the only declaration. It projects into:

- **`anthropic_tool_schemas()`** — `tools=[...]` for the Messages API,
- **`mcp_tool_definitions()`** — `inputSchema`/`outputSchema` for MCP,
- **`Tool.from_schema(...)`** — the pydantic-ai agent,

with one validation path and one error-normalisation path behind all three. A
test asserts the two schema dialects are byte-identical, because the moment a
tool schema exists in two places one of them is wrong.

Pydantic's `$defs`/`$ref` output is flattened by `inline_defs()` before export:
some tool-calling APIs and MCP clients quietly ignore a referenced subschema,
which surfaces as a model inventing enum values.

### The MCP server

`src/reconagent/mcp_server.py` (189 lines, redeclaring nothing) serves the same
12 tools over stdio, plus 3 resources and a prompt:

```bash
make agent-mcp            # or: python scripts/mcp_server.py
```

```json
{
  "mcpServers": {
    "reconagent": {
      "command": "python",
      "args": ["scripts/mcp_server.py"],
      "cwd": "/path/to/this/repo"
    }
  }
}
```

Or containerised — MCP speaks JSON-RPC over stdio, so the image is driven by
an attached client rather than a published port:

```bash
docker build --target mcp -t reconagent-mcp .
docker run -i --rm reconagent-mcp
```

| | |
|---|---|
| Tools | all 12, with `readOnlyHint`/`idempotentHint` annotations derived from `ToolSpec.mutates` |
| Resources | `recon://queue/open`, `recon://policy/accounting`, `recon://ledger/summary` |
| Prompts | `triage_exception(exception_id)` |

Argument validation is deliberately left to the registry rather than the MCP
SDK's JSON Schema check. Both reject the same calls, but only the registry
returns a repair instruction — which field, why, and what the valid fields are
— and an MCP client deserves the same quality of error as the in-process loop.
`tests/test_agent_mcp.py` exercises all of this over a real client session.

### Bounded loops and failure handling

The hand-rolled loop (`agent/loop.py`) holds two invariants:

1. **Every episode ends in exactly one terminal write.** When a bound fires,
   the loop escalates on the agent's behalf. An exception that silently falls
   out of the queue is the one outcome an AR team cannot detect.
2. **The safety-net escalation is not subject to the budget that triggered
   it.** A guard that can be blocked by the guard it fired for is not a guard.

Seven bounds, all in one `Budget` dataclass, all tested with policies that
misbehave on purpose:

| Bound | Fires when | Test |
|---|---|---|
| `max_steps` | the agent keeps gathering evidence forever | `_Spinner` |
| `max_tool_calls` | including retries — bounds real upstream load | `_Spinner` |
| `max_tool_errors` | six *different* failures means the evidence is not there | fault injector at 100% |
| `max_identical_calls` | the same call with the same arguments — cheapest loop detector there is | `_Repeater` |
| `max_cost_usd` | the episode is no longer worth what a human costs | budget check |
| `max_wall_clock_s` | the nightly batch window is closing | budget check |
| policy gave up / raised | the model answered in prose, or the policy crashed | `_Quitter`, `_Exploder` |

Tool failures are split in two, deliberately: a **transport failure** is
retried with exponential backoff and the model never learns it happened; a
**semantic failure** (invoice does not exist, arguments invalid) comes back as
a tool result the model has to reason about. Collapsing them means a flaky
upstream burns the model's reasoning budget and shows up in the transcript as
indecision.

The chaos eval profile injects failures into 12% of read calls, one in five of
them non-retryable, on a fixed seed.

### Evaluation

`make agent-eval` runs three suites over the same 70-task queue and gates the
build on the first:

| Metric | rules / clean | naive / clean | rules / chaos |
|---|---|---|---|
| Task success | **1.000** | 0.257 | 0.900 |
| Escalation recall | 1.000 | 0.333 | 1.000 |
| Escalation precision | 1.000 | 1.000 | 0.774 |
| Missed escalations (booked a guess) | 0.000 | 0.229 | **0.000** |
| Wrong linkage | 0.000 | 0.000 | **0.000** |
| Unhandled (no terminal write) | 0.000 | 0.000 | **0.000** |
| Steps/task | 4.79 | 2.50 | 4.64 |
| Retries/task | 0.00 | 0.00 | 0.27 |
| Projected $/task on `claude-sonnet-5` | 0.0482 | 0.0243 | 0.0467 |

Three things that table is designed to show:

- **The eval discriminates.** `naive` is a deliberately under-engineered
  baseline that matches on amount and never escalates. An eval that cannot tell
  a careful agent from a careless one is not measuring anything, so the
  careless one ships alongside as the floor.
- **Task success is all-or-nothing.** A reconciliation booked against the right
  invoice with the wrong adjustment is not 80% correct; it is a wrong journal
  entry. Partial credit would let exactly that regression through the gate.
- **Under chaos, accuracy degrades but safety does not.** All ten losses are
  `over_escalation` — the agent could not get the evidence and handed the case
  to a human. Missed escalations, wrong linkage and unhandled episodes stay at
  zero. That asymmetry is the property worth having, and it is measured rather
  than asserted.

`rules` is the deterministic control policy: a real policy that reads only tool
output and never ground truth, written so the harness has a known ceiling and
so CI can gate agent behaviour with no API key. It scores 100% on the clean
profile because it was written against this world — the number is a ceiling,
not a claim about LLM performance. Run the same suites against a model with:

```bash
ANTHROPIC_API_KEY=sk-... python -m reconagent.cli eval --policy anthropic
ANTHROPIC_API_KEY=sk-... python -m reconagent.cli eval --runner framework   # pydantic-ai
```

Gate thresholds (`evaluation/harness.py`, enforced by `scripts/run_agent_eval.py`):

```python
{"task_success_min": 0.90, "escalation_recall_min": 0.85,
 "missed_escalation_rate_max": 0.05, "wrong_linkage_rate_max": 0.02,
 "unhandled_rate_max": 0.0}
```

### Where the money actually goes

Roughly 92% of input tokens are the same ~2,900-token prefix — the tool
schemas plus the system prompt — re-sent on each of ~4.8 turns. Nothing about
the loop choice changes that; it is a property of a 12-tool surface and a
short episode.

| | Per task | Per 10k exceptions/month |
|---|---|---|
| As implemented | ~$0.05 | ~$500 |
| With prompt caching on the fixed prefix | ~$0.02 | ~$220 |

Exact figures, and the token counter that produced them, are in the **Cost
basis** table of [`eval/agent/report.md`](eval/agent/report.md) — regenerated
on every run rather than transcribed here. They move a few percent with the
counter: `budget.py` uses tiktoken when its encoding is available and falls
back to a ~4-chars-per-token estimate when the network is not, so a sandboxed
run and a CI run disagree slightly. The report names which one it used.

The second-largest lever is that episodes deliberately do not share context: an
exception is a unit of work with its own budget, and letting turn history
accumulate across a shift is how a queue-draining agent's cost goes quadratic.

### Hand-rolled loop vs framework

Both are implemented. The hand-rolled loop ships;
[`docs/AGENT_SDK_TRADEOFFS.md`](docs/AGENT_SDK_TRADEOFFS.md) is the argument,
built out of what the pydantic-ai version actually cost:

- pydantic-ai has **no terminal tool** — its stop condition is "the model
  produced final output", so ending on a successful write means raising an
  exception to unwind the run, and every correct episode exits through an
  exception handler.
- `UsageLimits` covers **three of the seven bounds**; wall clock, tool-error
  budget and identical-call detection have no equivalent.
- When a limit fires it **raises**, so `framework.py` imports `escalate_as_safety_net`
  back out of the hand-rolled loop to keep invariant #1.

The framework is not worse — it is aimed at streaming, multi-provider,
multi-agent problems this task does not have. The doc says what would change
my mind.

### Trying it

```bash
make agent-queue                      # what's in the queue
make agent-run EXC=EXC-0015           # work one exception, print the full step trace
make agent-drain                      # work the whole open queue, one bounded episode each
make agent-tools                      # the tool table above
make agent-mcp                        # start the MCP server on stdio
make agent-eval                       # all three suites + the gate
```

```
$ make agent-run EXC=EXC-0015
[EXC-0015] EXC-0015 via rules (rules-v1)
   1. get_exception({"exception_id": "EXC-0015"}) -> ok
      why: Load the exception under review.
   2. get_accounting_policy({}) -> ok
      why: Read the thresholds before judging any variance.
   3. get_fx_rate({"base": "EUR", "on_date": "2026-04-06", "quote": "USD"}) -> ok
      why: Amounts are in different currencies; convert at the value-date rate.
   4. record_resolution({...,"resolution_type": "fx_variance", "adjustment_cents": -3758}) -> ok
      why: 1445000 EUR converts to 1556420 USD at the 2026-04-06 rate; the
           -3758 cent residual is inside the 75bp FX tolerance.
  => resolved

  tool calls 4 (errors 0, retries 0) · tokens 12361/199 · $0.0000
  (projected $0.0401 on claude-sonnet-5) · 0.358s
```

### Project B layout

```
src/reconagent/
  models.py         Pydantic domain: Invoice, BankTransaction, ReconciliationException,
                      Resolution, CustomerPolicy, AccountingPolicy, FxRate
  world.py          deterministic world + golden set generator (the only place ground truth lives)
  store.py          the two data sources and the exception queue behind one API
  matching.py       memo reference normalisation and fuzzy ranking (stdlib only, deterministic)
  budget.py         Budget (seven bounds) + CostLedger (tokens, $, projected $)
  tools/
    schemas.py      Pydantic input/output models — the contract, and the only contract
    ledger_tools.py handlers: (store, validated_input) -> output_model
    registry.py     TOOL_SPECS + schema projection + validation + error normalisation
    faults.py       seeded fault injection (retryable and hard)
  agent/
    loop.py         the hand-rolled loop: budgets, retries, repeat detection, safety net
    policies.py     RulesPolicy (control) and NaivePolicy (floor)
    llm_policy.py   AnthropicPolicy over the Messages API
    framework.py    the same tool surface driven by pydantic-ai
    prompts.py      system prompt, task prompt, transcript rendering
    trace.py        AgentStep / AgentRun — the failure taxonomy the eval is built on
  evaluation/
    scoring.py      one episode vs its known-correct disposition
    harness.py      suites, profiles, metrics, gate
    report.py       markdown + JSON reports
  mcp_server.py     MCP: 12 tools, 3 resources, 1 prompt
  cli.py            queue / run / tools / eval
scripts/            mcp_server.py, run_agent_eval.py
eval/agent/         generated report.md + results.json
docs/               AGENT_SDK_TRADEOFFS.md
```

### Known limitations

- **The world is generated, not a real ERP extract.** Deliberate: it is the
  only way the golden set can be mechanically verified against ground truth,
  and it keeps the eval free and deterministic. It also means the archetypes
  are cleaner than production, where a single exception can be a bank fee *and*
  a partial payment at once. `LedgerStore` is the seam where real adapters go.
- **`RulesPolicy` scoring 100% is a ceiling, not a result.** It is a control
  written against this world. The honest reading of the eval table is the gap
  between it and `naive`, and the shape of the chaos-profile degradation — not
  the headline number.
- **Fuzzy reference matching is `difflib`, not a learned matcher.** Chosen for
  determinism and zero dependencies. It handles missing separators, OCR digit
  look-alikes and transposed digits; it will not handle a memo in a different
  language or a customer's internal PO number.
- **The Anthropic and pydantic-ai policies are not exercised in CI**, because
  CI has no key. They are covered by offline tests that assert the tool surface
  projects correctly into both, which catches the failure that actually
  happens: a tool added to the registry and silently missing from one consumer.
- **Cost figures are projections**, priced from the table in `budget.py` at
  token volumes measured offline. Real runs will differ on output tokens, which
  the offline policies can only estimate — and on input tokens, which depend on
  whether tiktoken's encoding downloaded. The eval report states which counter
  it used; treat a figure quoted without one as approximate.
