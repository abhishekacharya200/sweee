"""Policies: what to do next, given what we know.

Three of them ship:

- `RulesPolicy` is the deterministic control. It is a real policy — it reads
  only tool output, never ground truth — and it exists so the harness has a
  known ceiling and so CI can gate on agent behaviour without an API key.
- `NaivePolicy` is the deliberately under-engineered baseline. An eval that
  cannot tell a good agent from a careless one is not measuring anything, so
  the careless one is shipped alongside as the floor.
- `AnthropicPolicy` (in `llm_policy.py`) is the real thing, behind a key.

The loop does not care which it is holding.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Protocol

from ..budget import count_tokens
from ..models import ResolutionType
from ..tools.registry import anthropic_tool_schemas
from .prompts import render_transcript
from .state import AgentState, PolicyDecision

# A memo reference must look this much like an invoice id before it is worth
# testing. Below it, bogus references from other years still score ~0.78.
REF_ACCEPT_SCORE = 0.88

_TOOL_SCHEMA_TOKENS = None


def _schema_tokens() -> int:
    """Tool schemas are re-sent every turn, so they are re-counted every turn."""
    global _TOOL_SCHEMA_TOKENS
    if _TOOL_SCHEMA_TOKENS is None:
        _TOOL_SCHEMA_TOKENS = count_tokens(json.dumps(anthropic_tool_schemas()))
    return _TOOL_SCHEMA_TOKENS


def fixed_prefix_tokens() -> int:
    """System prompt plus tool schemas: the part of every request that repeats.

    Without prompt caching this is paid once per turn, per episode, forever.
    """
    from .prompts import SYSTEM_PROMPT

    return count_tokens(SYSTEM_PROMPT) + _schema_tokens()


class Policy(Protocol):
    name: str
    model: str

    def next_action(self, state: AgentState) -> PolicyDecision: ...


class _OfflinePolicy:
    """Shared token accounting for the policies that never call a model."""

    name = "offline"
    model = "offline-v1"

    def _account(self, state: AgentState, decision: PolicyDecision) -> PolicyDecision:
        prompt = state.system_prompt + state.task_prompt + render_transcript(state.turns)
        decision.input_tokens = count_tokens(prompt) + _schema_tokens()
        decision.output_tokens = count_tokens(
            json.dumps({"tool": decision.tool_name, "args": decision.arguments}, default=str)
            + decision.reasoning
        )
        return decision


class _Evidence:
    """Everything the policy has learned so far, pulled out of the turn log."""

    def __init__(self, state: AgentState) -> None:
        exc_payload = state.last_payload("get_exception") or {}
        self.exception = exc_payload.get("exception")
        self.transaction = exc_payload.get("transaction")
        self.invoice = exc_payload.get("invoice")

        txn_payload = state.last_payload("get_bank_transaction")
        if self.transaction is None and txn_payload:
            self.transaction = txn_payload["transaction"]
        inv_payload = state.last_payload("get_invoice")
        if self.invoice is None and inv_payload:
            self.invoice = inv_payload["invoice"]

        acct = state.last_payload("get_accounting_policy")
        self.accounting = acct["policy"] if acct else None
        cust = state.last_payload("get_customer_policy")
        self.customer = cust["policy"] if cust else None
        refs = state.last_payload("extract_invoice_reference")
        self.ref_candidates = refs["candidates"] if refs else None
        search = state.last_payload("search_invoices")
        self.amount_matches = search["invoices"] if search else None
        txns = state.last_payload("search_bank_transactions")
        self.prior_txns = txns["transactions"] if txns else None
        self.fx = state.last_payload("get_fx_rate")


def _call(tool: str, arguments: dict, reasoning: str) -> PolicyDecision:
    return PolicyDecision(tool_name=tool, arguments=arguments, reasoning=reasoning)


def _escalate(
    exception_id: str, reason: str, *, severity: str = "medium", owner: str = "ar_operations"
) -> PolicyDecision:
    return _call(
        "escalate_exception",
        {
            "exception_id": exception_id,
            "reason": reason,
            "severity": severity,
            "suggested_owner": owner,
        },
        reasoning=f"Evidence does not support any disposition: {reason}",
    )


def _resolve(
    exception_id: str,
    resolution: ResolutionType,
    invoice_id: str,
    txn_id: str,
    adjustment_cents: int,
    rationale: str,
    confidence: float,
) -> PolicyDecision:
    return _call(
        "record_resolution",
        {
            "exception_id": exception_id,
            "resolution_type": resolution.value,
            "invoice_id": invoice_id,
            "txn_id": txn_id,
            "adjustment_cents": adjustment_cents,
            "rationale": rationale,
            "confidence": confidence,
        },
        reasoning=rationale,
    )


class RulesPolicy(_OfflinePolicy):
    """Cheapest disambiguating evidence first, escalate the moment it runs out."""

    name = "rules"
    model = "rules-v1"

    def next_action(self, state: AgentState) -> PolicyDecision:
        return self._account(state, self._decide(state))

    def _decide(self, state: AgentState) -> PolicyDecision:
        ev = _Evidence(state)
        exception_id = state.exception_id

        if ev.exception is None:
            if state.attempted("get_exception"):
                return _escalate(exception_id, "The exception record could not be loaded from the queue.")
            return _call("get_exception", {"exception_id": exception_id}, "Load the exception under review.")

        if ev.accounting is None:
            if state.attempted("get_accounting_policy"):
                return _escalate(
                    exception_id,
                    "Entity materiality thresholds are unavailable, so no disposition can be justified.",
                )
            return _call("get_accounting_policy", {}, "Read the thresholds before judging any variance.")

        if ev.transaction is None:
            return _escalate(exception_id, "No bank credit is attached to this exception to reconcile.")

        if ev.invoice is None:
            return self._identify_invoice(state, ev)
        return self._classify(state, ev)

    def _identify_invoice(self, state: AgentState, ev: _Evidence) -> PolicyDecision:
        exception_id = state.exception_id
        txn = ev.transaction

        # Narrow on the exact amount before fuzzy-matching. Ranking a memo
        # against the whole ledger buries the right invoice under near-ties
        # from unrelated ids; ranking it against a two-row shortlist does not.
        if not state.attempted("search_invoices"):
            return _call(
                "search_invoices",
                {
                    "amount_cents": txn["amount_cents"],
                    "tolerance_cents": 0,
                    "currency": txn["currency"],
                    "limit": 10,
                },
                "Find every invoice the credit could settle exactly.",
            )

        by_amount = {inv["invoice_id"]: inv for inv in (ev.amount_matches or [])}
        if not by_amount:
            return _escalate(
                exception_id,
                f"No invoice matches the {txn['amount_cents']} cent credit in {txn['currency']}, "
                "so there is nothing to reconcile it against.",
            )

        if not state.attempted("extract_invoice_reference"):
            return _call(
                "extract_invoice_reference",
                {"memo": txn["memo"], "candidate_invoice_ids": sorted(by_amount), "top_k": 10},
                "Check whether the memo supports any of the amount matches.",
            )

        supported = sorted(
            c["invoice_id"] for c in (ev.ref_candidates or []) if c["score"] >= REF_ACCEPT_SCORE
        )
        if len(supported) == 1:
            return _call(
                "get_invoice",
                {"invoice_id": supported[0]},
                "Reference and amount agree on one invoice; pull the record to confirm.",
            )
        if len(supported) > 1:
            return _escalate(
                exception_id,
                f"The memo reference supports {len(supported)} invoices that all match the amount: "
                f"{', '.join(supported)}.",
            )
        return _escalate(
            exception_id,
            f"The credit carries no remittance reference matching any of the {len(by_amount)} "
            f"invoice(s) at this amount ({', '.join(sorted(by_amount))}), so it cannot be attributed.",
        )

    def _classify(self, state: AgentState, ev: _Evidence) -> PolicyDecision:
        exception_id = state.exception_id
        inv, txn = ev.invoice, ev.transaction

        if inv["currency"] != txn["currency"]:
            return self._classify_fx(state, ev)

        delta = inv["amount_cents"] - txn["amount_cents"]
        if delta == 0:
            return self._classify_exact(state, ev)
        if delta < 0:
            return _escalate(
                exception_id,
                f"The credit exceeds invoice {inv['invoice_id']} by {-delta} cents, which needs a "
                "credit note or a refund decision.",
                severity="high",
                owner="credit_control",
            )
        return self._classify_shortfall(state, ev, delta)

    def _classify_fx(self, state: AgentState, ev: _Evidence) -> PolicyDecision:
        exception_id = state.exception_id
        inv, txn, acct = ev.invoice, ev.transaction, ev.accounting

        if ev.fx is None:
            if state.attempted("get_fx_rate"):
                failure = state.last_failure("get_fx_rate")
                return _escalate(
                    exception_id,
                    f"Invoice is in {inv['currency']} and the credit in {txn['currency']}, but no rate "
                    f"could be obtained for {txn['value_date']} ({failure.error_kind if failure else 'unknown'}).",
                )
            return _call(
                "get_fx_rate",
                {"base": inv["currency"], "quote": txn["currency"], "on_date": txn["value_date"]},
                "Amounts are in different currencies; convert at the value-date rate.",
            )

        converted = (inv["amount_cents"] * ev.fx["rate_ppm"] + 500_000) // 1_000_000
        residual = converted - txn["amount_cents"]
        tolerance = (converted * acct["fx_variance_tolerance_bps"]) // 10_000
        if abs(residual) <= tolerance:
            return _resolve(
                exception_id,
                ResolutionType.FX_VARIANCE,
                inv["invoice_id"],
                txn["txn_id"],
                residual,
                f"{inv['amount_cents']} {inv['currency']} converts to {converted} {txn['currency']} at "
                f"the {txn['value_date']} rate; the {residual} cent residual is inside the "
                f"{acct['fx_variance_tolerance_bps']}bp FX tolerance.",
                0.9,
            )
        return _escalate(
            exception_id,
            f"After converting at the {txn['value_date']} rate the residual is {residual} cents, "
            f"outside the {tolerance} cent FX tolerance, so the gap is not exchange movement.",
            severity="high",
            owner="credit_control",
        )

    def _classify_exact(self, state: AgentState, ev: _Evidence) -> PolicyDecision:
        exception_id = state.exception_id
        inv, txn, acct = ev.invoice, ev.transaction, ev.accounting
        other_settlements = [t for t in inv["settled_txn_ids"] if t != txn["txn_id"]]

        if not other_settlements:
            return _resolve(
                exception_id,
                ResolutionType.MATCH_PAYMENT,
                inv["invoice_id"],
                txn["txn_id"],
                0,
                f"Credit {txn['txn_id']} settles invoice {inv['invoice_id']} exactly "
                f"({txn['amount_cents']} cents, {txn['currency']}) and the invoice is otherwise unsettled.",
                0.95,
            )

        if not state.attempted("search_bank_transactions"):
            return _call(
                "search_bank_transactions",
                {"matched_invoice_id": inv["invoice_id"], "limit": 10},
                "The invoice is already settled; find the earlier credit before booking anything.",
            )

        lookback = acct["duplicate_lookback_days"]
        value_date = date.fromisoformat(txn["value_date"])
        duplicates = [
            prior
            for prior in (ev.prior_txns or [])
            if prior["txn_id"] != txn["txn_id"]
            and prior["amount_cents"] == txn["amount_cents"]
            and abs((value_date - date.fromisoformat(prior["value_date"])).days) <= lookback
        ]
        if duplicates:
            first = duplicates[0]
            return _resolve(
                exception_id,
                ResolutionType.DUPLICATE_PAYMENT,
                inv["invoice_id"],
                txn["txn_id"],
                -txn["amount_cents"],
                f"Invoice {inv['invoice_id']} was already settled in full by {first['txn_id']} on "
                f"{first['value_date']}, inside the {lookback}-day window, so {txn['txn_id']} is a "
                "duplicate and the cash is owed back.",
                0.9,
            )
        return _escalate(
            exception_id,
            f"Invoice {inv['invoice_id']} is marked settled but no earlier credit inside the "
            f"{lookback}-day window explains this one.",
        )

    def _classify_shortfall(self, state: AgentState, ev: _Evidence, delta: int) -> PolicyDecision:
        exception_id = state.exception_id
        inv, txn, acct = ev.invoice, ev.transaction, ev.accounting

        if ev.customer is None:
            if state.attempted("get_customer_policy"):
                return _escalate(
                    exception_id,
                    f"Contract terms for {inv['customer_id']} are unavailable, so a {delta} cent "
                    "shortfall cannot be told apart from a bank fee.",
                )
            return _call(
                "get_customer_policy",
                {"customer_id": inv["customer_id"]},
                "A shortfall is only explainable against the customer's fee terms.",
            )

        cust = ev.customer
        fee_gap = abs(delta - cust["wire_fee_cents"])
        if not cust["pays_wire_fees"] and fee_gap <= acct["fee_variance_tolerance_cents"]:
            return _resolve(
                exception_id,
                ResolutionType.BANK_FEE,
                inv["invoice_id"],
                txn["txn_id"],
                delta,
                f"{cust['customer_name']} deducts wire fees under contract and the {delta} cent "
                f"shortfall is within {fee_gap} cents of the agreed {cust['wire_fee_cents']} cent fee.",
                0.9,
            )

        limit = acct["short_payment_writeoff_max_cents"]
        if delta <= limit:
            return _resolve(
                exception_id,
                ResolutionType.SHORT_PAYMENT,
                inv["invoice_id"],
                txn["txn_id"],
                delta,
                f"Shortfall of {delta} cents is not explained by fees or currency and sits under the "
                f"{limit} cent write-off limit.",
                0.8,
            )
        return _escalate(
            exception_id,
            f"Shortfall of {delta} cents on invoice {inv['invoice_id']} exceeds the {limit} cent "
            "write-off limit and needs a human credit decision.",
            severity="high",
            owner="credit_control",
        )


class NaivePolicy(_OfflinePolicy):
    """Match on amount, book it, never escalate — the floor of the eval.

    It is what an agent looks like when the tool surface is treated as a
    lookup rather than as evidence.
    """

    name = "naive"
    model = "naive-v1"

    def next_action(self, state: AgentState) -> PolicyDecision:
        return self._account(state, self._decide(state))

    def _decide(self, state: AgentState) -> PolicyDecision:
        ev = _Evidence(state)
        if ev.exception is None:
            if state.attempted("get_exception"):
                return PolicyDecision(tool_name=None, reasoning="Could not load the exception.")
            return _call("get_exception", {"exception_id": state.exception_id}, "Load the exception.")

        txn = ev.transaction
        if ev.invoice is not None and txn is not None:
            return self._book(state.exception_id, ev.invoice["invoice_id"], txn["txn_id"])
        if txn is None:
            return PolicyDecision(tool_name=None, reasoning="No transaction attached.")

        if not state.attempted("search_invoices"):
            return _call(
                "search_invoices",
                {"amount_cents": txn["amount_cents"], "tolerance_cents": 5_000, "limit": 5},
                "Look for an invoice near this amount.",
            )
        matches = ev.amount_matches or []
        if matches:
            return self._book(state.exception_id, matches[0]["invoice_id"], txn["txn_id"])
        return PolicyDecision(tool_name=None, reasoning="No candidate invoice found.")

    @staticmethod
    def _book(exception_id: str, invoice_id: str, txn_id: str) -> PolicyDecision:
        return _resolve(
            exception_id,
            ResolutionType.MATCH_PAYMENT,
            invoice_id,
            txn_id,
            0,
            "Amount is close enough to the invoice to link them.",
            0.5,
        )


def build_policy(name: str, *, model: str | None = None, api_key: str | None = None) -> Policy:
    if name == "rules":
        return RulesPolicy()
    if name == "naive":
        return NaivePolicy()
    if name in {"anthropic", "llm"}:
        from .llm_policy import AnthropicPolicy

        return AnthropicPolicy(model=model or "claude-sonnet-5", api_key=api_key)
    if name == "pydantic-ai":
        raise ValueError(
            "The pydantic-ai agent runs its own loop; use `run_pydantic_ai_episode` in "
            "reconagent.agent.framework instead of the hand-rolled loop."
        )
    raise ValueError(f"Unknown policy {name!r}. Available: rules, naive, anthropic.")
