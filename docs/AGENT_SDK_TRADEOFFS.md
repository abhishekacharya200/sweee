# Choosing the loop: why this agent is hand-rolled

**The decision:** the reconciliation agent runs on a hand-written loop
(`src/reconagent/agent/loop.py`, 272 lines) over the raw Messages API. A
pydantic-ai implementation of the same agent ships alongside it
(`src/reconagent/agent/framework.py`) and is kept, tested and runnable — not as
a hedge, but because building it is what produced the argument below.

This is not a general verdict on frameworks. It is a verdict for *this* task,
and the reasoning transfers better than the conclusion does.

---

## 1. What this task actually demands

Reconciliation is not a chat product. The properties that decide the
architecture are unusual enough that they are worth stating before naming any
library:

1. **Every episode must end in exactly one terminal write.** An exception that
   falls out of the queue is invisible — nobody gets a ticket, nobody notices
   for a month. Ending "successfully with no disposition" is the worst
   available outcome, worse than a wrong answer, because a wrong answer at
   least shows up in a reconciliation report.
2. **The stop condition is a tool call, not a message.** The run is over when
   `record_resolution` or `escalate_exception` succeeds. Prose after that point
   is noise.
3. **Abandonment needs a defined behaviour, not an exception.** When a budget
   blows, "raise" is not a policy. The exception still exists and still needs
   an owner.
4. **Cost is per unit of work, on a queue with tens of thousands of units.**
   $0.05 versus $0.02 per exception is a real line item, not a rounding error.
5. **Every decision has to be reconstructible.** "Why did it write off $412?"
   is a question an auditor asks six months later, and the answer has to be a
   trace, not a log level.

Note what is *absent*: no streaming to a user, no multi-agent handoff, no
long-lived conversation, no human in the loop mid-task. Frameworks are priced
for those features, and this task buys none of them.

---

## 2. The options

| | Hand-rolled | Anthropic `tool_runner` | pydantic-ai | LangGraph | Claude Agent SDK |
|---|---|---|---|---|---|
| Loop ownership | mine | SDK | framework | mine, as a graph | SDK |
| Terminal-tool stop | native | needs a break | needs a raise | native (edge to END) | needs a break |
| Custom bounds | all seven | few | three of seven | all, hand-written | few |
| Safe default on abandon | native | bolt-on | bolt-on | native | bolt-on |
| Structured tool schemas | from Pydantic | from Pydantic | native | manual | from Pydantic |
| Step trace with reasoning | native | partial | via events | native | partial |
| Cost to first working agent | highest | lowest | low | medium | low |
| Cost to *correct* agent here | lowest | high | medium | medium | high |
| Dependency surface | `anthropic` | `anthropic` | ~15 packages | large | SDK + Node runtime |

The last two rows are the whole argument. Every option gets you to a working
agent quickly. The question is what it costs to get from working to *correct*
under the five constraints above.

---

## 3. What building the framework version actually taught me

I wrote `framework.py` expecting it to replace the loop. Three things came out
of it, and all three are visible in the file:

### 3.1 There is no terminal tool, so you raise

pydantic-ai ends a run when the model produces final output. This agent ends
when a write succeeds. Bridging them means throwing an exception out of the
tool to unwind the run:

```python
class _TerminalReached(Exception):
    """Raised out of a terminal tool to end the framework's run."""
```

It works. It also means the framework's own completion path — the one its
retries, output validation and result object are built around — is now the
path never taken. Every run of a correct episode exits through an exception
handler. That is not a small smell; it means the framework's model of "done"
and mine disagree, and mine is the one the business cares about.

The alternative is to let the model produce structured output as the
disposition instead of calling a tool. That reads better in pydantic-ai — and
it moves the write out of the tool surface, which breaks the property that the
MCP server and the loop expose the *same* twelve tools. I would rather raise.

### 3.2 Usage limits are not the bounds I need

`UsageLimits` covers three of the seven bounds in `Budget`:

| Bound | pydantic-ai | Why it matters here |
|---|---|---|
| `max_steps` | `request_limit` | ✅ |
| `max_tool_calls` | `tool_calls_limit` | ✅ |
| `max_cost_usd` | `cost_limit` | ✅ |
| `max_wall_clock_s` | — | A nightly batch has a window |
| `max_tool_errors` | — | Distinct from retries: six *different* failures means the evidence isn't there |
| `max_identical_calls` | — | The cheapest loop detector there is, and it caught a real repeat during development |
| `tool_max_attempts` + backoff | partly, via `ModelRetry` | Retrying a 503 is not the same event as telling the model it was wrong |

That last row is the interesting one. pydantic-ai's `ModelRetry` is a good
abstraction — it is the framework-native way to say "that call was wrong, try
differently", and it is what `framework.py` uses. But it merges two things the
hand-rolled loop keeps apart: a **transport failure** (retry the same call,
silently, with backoff — the model never needs to know) and a **semantic
failure** (the model asked for an invoice that does not exist — it must know).
Collapsing them means a flaky upstream burns the model's reasoning budget and
shows up in the transcript as indecision.

The hand-rolled loop separates them explicitly. `execute_with_retries` retries
only `retryable` outcomes and never tells the model; everything else comes back
as a tool result the model has to reason about. Under the chaos profile that
distinction shows up as **0.27 retries per task that the model never saw**.

### 3.3 The safety net has to be reattached by hand

When a usage limit fires, pydantic-ai raises `UsageLimitExceeded`. Correct
behaviour for a library. But the exception is still open, so `framework.py`
ends with:

```python
if guard is not None:
    escalate_as_safety_net(registry, exception_id, guard, run, ledger, detail=detail)
```

— importing the safety net back out of the hand-rolled loop. The framework
version cannot satisfy requirement #1 without the module it was supposed to
replace. At that point the framework is handling message assembly and tool
dispatch, and I am still handling every property that makes this an agent
rather than a chat loop.

---

## 4. What hand-rolling actually cost

Honestly, and in order of how much it hurt:

- **Message assembly.** `llm_policy.py` (115 lines) rebuilds the messages array
  from the turn log every call, including `tool_use`/`tool_result` id pairing.
  This is the part most likely to break on a provider change, and the part a
  framework genuinely does better.
- **No streaming.** Fine here, would not be fine in a UI.
- **No observability integration.** pydantic-ai ships Logfire/OTel wiring; I
  have `AgentRun.transcript()`. For an audit trail that is arguably what I
  want — a domain trace rather than spans — but I built it, and I maintain it.
- **Provider lock.** The loop speaks Anthropic tool-use shapes. A second
  provider means a second policy class. pydantic-ai would have absorbed that.
- **~390 lines to own** (`loop.py` + `llm_policy.py`) versus ~200 for the
  framework adapter. That is real, and it is the honest cost of the decision.

What it bought: all seven bounds, a terminal-tool stop condition that is the
loop's actual exit, a safe default that cannot be bypassed by the budget that
triggered it, and a step trace with per-step reasoning, attempts, tokens and
error kind, which is what the eval harness is built on top of. Requirement #1
is not a feature I added; it is the loop's structure.

---

## 5. Where the money is (and it is not the loop)

Measured on the shipped 70-task queue, projected at `claude-sonnet-5` rates:

| | Per task | Per 10k exceptions/month |
|---|---|---|
| Naive implementation (schemas re-sent every turn) | ~$0.05 | ~$500 |
| With prompt caching on the fixed prefix | ~$0.02 | ~$210 |

**Roughly 93% of input tokens are the same ~2,900-token prefix** — the tool
schemas plus the system prompt — re-sent on each of ~4.8 turns. Nothing about
the loop choice changes that number; it is a property of a 12-tool surface and
a short episode.

(Exact values, and the token counter behind them, are in the Cost basis table
of the generated eval report. They shift a few percent depending on whether
tiktoken's encoding was reachable, which is why the report names the counter
and this document rounds.) The single highest-leverage cost decision
in this system is caching that prefix, and the second is **not** letting
context accumulate across the queue:

```python
# loop.py — run_queue
"""Episodes deliberately do not share context: an exception is a unit of work
with its own budget, and letting turn history accumulate across a shift is
how a queue-draining agent's cost goes quadratic."""
```

This is the part of the trade-off analysis that matters most in production and
that the framework-versus-hand-rolled question does not touch at all.

---

## 6. Why pydantic-ai and not the others

- **Anthropic's `tool_runner`** is the lowest-effort path to a working loop and
  I would reach for it first on a task with no terminal-tool semantics. Here it
  loses on the same points as pydantic-ai without the compensating typed-tool
  ergonomics.
- **LangGraph** would actually model this well — a terminal node is a native
  concept, and the safe default is just another edge. I did not choose it
  because this agent is a single linear episode with one decision point;
  expressing that as a graph adds a large dependency and a second mental model
  to buy back properties the 272-line loop already has. If the task grew a
  human-approval step, a parallel evidence-gathering fan-out, or resumable
  state across a batch window, that calculus flips and LangGraph is where I
  would go.
- **Claude Agent SDK** is built for a different shape of problem — a coding
  agent with a filesystem, a shell and a long horizon. Excellent at that;
  oversized for eleven read tools and two writes.
- **pydantic-ai** won the comparison slot because the tools were already
  Pydantic models, so `Tool.from_schema` consumes the registry's existing JSON
  schema directly. That keeps the comparison honest: the two implementations
  differ in the loop and *only* in the loop, which is the thing under
  examination.

---

## 7. The decision that mattered more than the loop

Neither. It was making the tool surface a first-class artifact instead of a
detail of whichever loop happened to be running.

`TOOL_SPECS` in `tools/registry.py` is declared once and projected into:

- `anthropic_tool_schemas()` → the Messages API,
- `mcp_tool_definitions()` → `inputSchema` for MCP,
- `Tool.from_schema(...)` → pydantic-ai,

with one validation path and one error-normalisation path behind all three. A
test asserts the two schema dialects are byte-identical:

```python
assert anthropic == mcp, "the two dialects must project from one schema, not two"
```

Because of that, `mcp_server.py` — a fully working MCP server with 12 tools, 3
resources and a prompt — is 189 lines and redeclares nothing. Swapping the loop
is a day's work. Discovering that your MCP server and your agent have drifted
apart on `adjustment_cents` sign convention is a month of wrong journal
entries.

---

## 8. What would change my mind

I would move to a framework if any of these became true:

- **A second provider is required.** The message-assembly code is the weakest
  part of what I own and the strongest part of what a framework gives.
- **Human-in-the-loop approval mid-episode.** Suspend/resume is genuinely hard
  to hand-roll correctly, and LangGraph's checkpointing solves it properly.
- **The episode stops being linear** — parallel evidence gathering, sub-agents
  per disposition type, retry-with-different-strategy branches.
- **The team grows past the point where 390 lines of loop is cheaper to read
  than a framework's docs.** This is a real threshold and it is about people,
  not code.

None of those are true today, so the loop stays.

---

## Scorecard

| Requirement | Hand-rolled | pydantic-ai as shipped |
|---|---|---|
| Ends in exactly one terminal write | structural | bolt-on (`escalate_as_safety_net` reimported) |
| Terminal tool ends the run | native | via raised exception |
| All seven bounds enforced | yes | three of seven |
| Transport retries hidden from the model | yes | merged into `ModelRetry` |
| Step trace with reasoning + attempts + tokens | native | reconstructed by `_Recorder` |
| Tool schemas declared once | yes (shared registry) | yes (shared registry) |
| Lines I maintain | ~390 | ~200 + the loop's safety net |
| Streaming, OTel, multi-provider | no | yes |

The framework is not worse. It is aimed somewhere else, and this task is not
there.
