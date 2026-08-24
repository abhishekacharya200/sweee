"""One registry, three consumers.

The hand-rolled loop, the pydantic-ai agent and the MCP server are all built
from `TOOL_SPECS`. That is the point of the module: a tool surface defined
once in Pydantic, projected into whatever schema dialect the consumer needs,
with a single implementation of argument validation and error normalisation
behind it. Adding a tool is one entry here and nothing else.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from ..store import LedgerError, LedgerStore
from . import ledger_tools as impl
from . import schemas as s
from .faults import FaultInjector, HardToolError, TransientToolError

Handler = Callable[[LedgerStore, BaseModel], BaseModel]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Handler
    terminal: bool = False
    mutates: bool = False


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_open_exceptions",
        description=(
            "List reconciliation exceptions still awaiting a disposition, oldest first. "
            "Use this to pick up work; use get_exception once you know which one you are on."
        ),
        input_model=s.ListOpenExceptionsInput,
        output_model=s.ListOpenExceptionsOutput,
        handler=impl.list_open_exceptions,
    ),
    ToolSpec(
        name="get_exception",
        description=(
            "Fetch one exception with whichever invoice and bank transaction the nightly matcher "
            "already linked to it. Always the first call for a given exception."
        ),
        input_model=s.GetExceptionInput,
        output_model=s.GetExceptionOutput,
        handler=impl.get_exception,
    ),
    ToolSpec(
        name="get_invoice",
        description="Fetch one invoice from the AR ledger by id.",
        input_model=s.GetInvoiceInput,
        output_model=s.GetInvoiceOutput,
        handler=impl.get_invoice,
    ),
    ToolSpec(
        name="get_bank_transaction",
        description="Fetch one bank statement line by id.",
        input_model=s.GetBankTransactionInput,
        output_model=s.GetBankTransactionOutput,
        handler=impl.get_bank_transaction,
    ),
    ToolSpec(
        name="search_invoices",
        description=(
            "Search the AR ledger by customer, amount (with tolerance), currency, status or issue date. "
            "Searching on the received amount is the cheapest way to confirm a memo reference."
        ),
        input_model=s.SearchInvoicesInput,
        output_model=s.SearchInvoicesOutput,
        handler=impl.search_invoices,
    ),
    ToolSpec(
        name="search_bank_transactions",
        description=(
            "Search the bank statement feed by memo text, amount, currency, value date or existing match. "
            "Use it to find an earlier credit that already settled an invoice."
        ),
        input_model=s.SearchBankTransactionsInput,
        output_model=s.SearchBankTransactionsOutput,
        handler=impl.search_bank_transactions,
    ),
    ToolSpec(
        name="extract_invoice_reference",
        description=(
            "Mine a bank memo for invoice references and rank known invoice ids by similarity, "
            "tolerating missing separators and OCR digit look-alikes. Returns candidates, not answers."
        ),
        input_model=s.ExtractInvoiceReferenceInput,
        output_model=s.ExtractInvoiceReferenceOutput,
        handler=impl.extract_invoice_reference,
    ),
    ToolSpec(
        name="get_customer_policy",
        description=(
            "Contract terms for one customer: payment terms, settlement currency, whether the customer "
            "bears wire fees, and the contracted fee amount. This is what makes a shortfall explainable."
        ),
        input_model=s.GetCustomerPolicyInput,
        output_model=s.GetCustomerPolicyOutput,
        handler=impl.get_customer_policy,
    ),
    ToolSpec(
        name="get_accounting_policy",
        description=(
            "Entity-wide thresholds: FX tolerance, fee variance tolerance, the maximum shortfall that may "
            "be written off without human approval, and the duplicate lookback window."
        ),
        input_model=s.GetAccountingPolicyInput,
        output_model=s.GetAccountingPolicyOutput,
        handler=impl.get_accounting_policy,
    ),
    ToolSpec(
        name="get_fx_rate",
        description=(
            "Published rate for a currency pair on a date. Rates exist for business days only; "
            "a missing rate is a real answer, not a bug."
        ),
        input_model=s.GetFxRateInput,
        output_model=s.GetFxRateOutput,
        handler=impl.get_fx_rate,
    ),
    ToolSpec(
        name="record_resolution",
        description=(
            "Close an exception by booking a disposition against an invoice and a transaction. "
            "Terminal: it writes to the ledger and ends the task. Only call it when the evidence supports it."
        ),
        input_model=s.RecordResolutionInput,
        output_model=s.RecordResolutionOutput,
        handler=impl.record_resolution,
        terminal=True,
        mutates=True,
    ),
    ToolSpec(
        name="escalate_exception",
        description=(
            "Hand an exception to a human queue with a reason. Terminal. "
            "This is the correct answer when the evidence is missing, contradictory or above your limit — "
            "it is never a failure to escalate, but it is a failure to guess."
        ),
        input_model=s.EscalateExceptionInput,
        output_model=s.EscalateExceptionOutput,
        handler=impl.escalate_exception,
        terminal=True,
        mutates=True,
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
TERMINAL_TOOLS = frozenset(spec.name for spec in TOOL_SPECS if spec.terminal)


@dataclass
class ToolOutcome:
    """What every consumer of the registry gets back — success or failure."""

    tool_name: str
    ok: bool
    latency_s: float
    payload: dict | None = None
    error: str | None = None
    error_kind: str | None = None
    retryable: bool = False
    terminal: bool = False
    attempts: int = 1

    def to_model_text(self) -> str:
        """The exact string handed back to a model as the tool result."""
        if self.ok:
            return json.dumps(self.payload, separators=(",", ":"), sort_keys=True)
        return json.dumps(
            {"error": self.error, "error_kind": self.error_kind, "retryable": self.retryable},
            separators=(",", ":"),
            sort_keys=True,
        )


def inline_defs(schema: dict) -> dict:
    """Flatten `$ref`/`$defs` so the schema is portable across providers.

    Pydantic emits `$defs` for every nested enum. Some tool-calling APIs and
    MCP clients handle that and some quietly ignore the referenced subschema,
    which shows up as a model inventing enum values. Inlining removes the
    question.
    """
    schema = deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: object, seen: frozenset[str]) -> object:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                if name in seen or name not in defs:
                    return {"type": "string"}
                target = resolve(defs[name], seen | {name})
                overrides = {k: v for k, v in node.items() if k != "$ref"}
                return {**target, **overrides}  # type: ignore[dict-item]
            return {key: resolve(value, seen) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        return node

    resolved = resolve(schema, frozenset())
    assert isinstance(resolved, dict)
    resolved.setdefault("type", "object")
    resolved.setdefault("properties", {})
    return resolved


def input_json_schema(spec: ToolSpec) -> dict:
    return inline_defs(spec.input_model.model_json_schema())


def anthropic_tool_schemas() -> list[dict]:
    """`tools=[...]` for the Anthropic Messages API."""
    return [
        {"name": spec.name, "description": spec.description, "input_schema": input_json_schema(spec)}
        for spec in TOOL_SPECS
    ]


def mcp_tool_definitions() -> list[dict]:
    """`tools/list` payloads for MCP, including the output schema MCP allows."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": input_json_schema(spec),
            "outputSchema": inline_defs(spec.output_model.model_json_schema()),
            "annotations": {
                "readOnlyHint": not spec.mutates,
                "destructiveHint": False,
                "idempotentHint": not spec.mutates,
            },
        }
        for spec in TOOL_SPECS
    ]


class ToolRegistry:
    """Validates, executes, and normalises the failure of every tool call."""

    def __init__(self, store: LedgerStore, faults: FaultInjector | None = None) -> None:
        self.store = store
        self.faults = faults
        self.call_log: list[ToolOutcome] = []

    @property
    def tool_names(self) -> list[str]:
        return [spec.name for spec in TOOL_SPECS]

    def call(self, tool_name: str, raw_args: dict | None = None) -> ToolOutcome:
        started = time.perf_counter()
        spec = TOOLS_BY_NAME.get(tool_name)
        if spec is None:
            return self._log(
                ToolOutcome(
                    tool_name=tool_name,
                    ok=False,
                    latency_s=time.perf_counter() - started,
                    error=f"Unknown tool {tool_name!r}. Available tools: {', '.join(self.tool_names)}.",
                    error_kind="unknown_tool",
                )
            )

        try:
            payload = spec.input_model.model_validate(raw_args or {})
        except ValidationError as exc:
            return self._log(
                ToolOutcome(
                    tool_name=tool_name,
                    ok=False,
                    latency_s=time.perf_counter() - started,
                    error=_validation_message(spec, exc),
                    error_kind="invalid_arguments",
                )
            )

        try:
            if self.faults is not None:
                self.faults.maybe_fail(tool_name)
            result = spec.handler(self.store, payload)
        except TransientToolError as exc:
            return self._log(
                ToolOutcome(
                    tool_name=tool_name,
                    ok=False,
                    latency_s=time.perf_counter() - started,
                    error=str(exc),
                    error_kind="transient",
                    retryable=True,
                )
            )
        except HardToolError as exc:
            return self._log(
                ToolOutcome(
                    tool_name=tool_name,
                    ok=False,
                    latency_s=time.perf_counter() - started,
                    error=str(exc),
                    error_kind="unavailable",
                )
            )
        except LedgerError as exc:
            return self._log(
                ToolOutcome(
                    tool_name=tool_name,
                    ok=False,
                    latency_s=time.perf_counter() - started,
                    error=str(exc),
                    error_kind="domain_error",
                )
            )

        return self._log(
            ToolOutcome(
                tool_name=tool_name,
                ok=True,
                latency_s=time.perf_counter() - started,
                payload=result.model_dump(mode="json"),
                terminal=spec.terminal,
            )
        )

    def _log(self, outcome: ToolOutcome) -> ToolOutcome:
        self.call_log.append(outcome)
        return outcome


def _validation_message(spec: ToolSpec, exc: ValidationError) -> str:
    """A repair instruction, not a stack trace.

    A model that gets `1 validation error for GetInvoiceInput` back will guess
    again; one that is told which field, why, and what the field list is will
    usually fix it on the next turn.
    """
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()[:5]
    )
    fields = ", ".join(spec.input_model.model_fields)
    return f"Invalid arguments for {spec.name}: {problems}. Accepted fields: {fields or '(none)'}."


def build_registry(store: LedgerStore, faults: FaultInjector | None = None) -> ToolRegistry:
    return ToolRegistry(store, faults=faults)


def tool_table_markdown() -> str:
    """Generated docs, so the README cannot drift from the registry."""
    lines = ["| Tool | Terminal | Writes | Purpose |", "|---|---|---|---|"]
    for spec in TOOL_SPECS:
        summary = spec.description.split(". ")[0].rstrip(".")
        lines.append(
            f"| `{spec.name}` | {'yes' if spec.terminal else 'no'} | "
            f"{'yes' if spec.mutates else 'no'} | {summary} |"
        )
    return "\n".join(lines)
