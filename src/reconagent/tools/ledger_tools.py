"""Tool handlers: the actual work behind each schema.

Handlers are plain functions of `(store, validated_input) -> output_model`.
They raise `LedgerError` for domain failures and never format a response
string themselves — turning an exception into something a model can read is
the registry's job, so the hand-rolled loop, the framework agent and the MCP
server all see the same error text.
"""

from __future__ import annotations

from ..matching import rank_reference_candidates
from ..models import Escalation, ExceptionStatus, Resolution
from ..store import LedgerStore
from ..world import ACCOUNTING_POLICY
from . import schemas as s


def list_open_exceptions(store: LedgerStore, payload: s.ListOpenExceptionsInput) -> s.ListOpenExceptionsOutput:
    rows = store.list_exceptions(status=ExceptionStatus.OPEN, kind=payload.kind, limit=payload.limit)
    total = len(store.list_exceptions(status=ExceptionStatus.OPEN, limit=10_000))
    return s.ListOpenExceptionsOutput(exceptions=rows, total_open=total)


def get_exception(store: LedgerStore, payload: s.GetExceptionInput) -> s.GetExceptionOutput:
    exc = store.get_exception(payload.exception_id)
    return s.GetExceptionOutput(
        exception=exc,
        invoice=store.get_invoice(exc.invoice_id) if exc.invoice_id else None,
        transaction=store.get_transaction(exc.txn_id) if exc.txn_id else None,
    )


def get_invoice(store: LedgerStore, payload: s.GetInvoiceInput) -> s.GetInvoiceOutput:
    return s.GetInvoiceOutput(invoice=store.get_invoice(payload.invoice_id))


def get_bank_transaction(
    store: LedgerStore, payload: s.GetBankTransactionInput
) -> s.GetBankTransactionOutput:
    return s.GetBankTransactionOutput(transaction=store.get_transaction(payload.txn_id))


def search_invoices(store: LedgerStore, payload: s.SearchInvoicesInput) -> s.SearchInvoicesOutput:
    rows = store.search_invoices(
        customer_id=payload.customer_id,
        amount_cents=payload.amount_cents,
        tolerance_cents=payload.tolerance_cents,
        currency=payload.currency,
        status=payload.status,
        issued_from=payload.issued_from,
        issued_to=payload.issued_to,
        limit=payload.limit + 1,
    )
    return s.SearchInvoicesOutput(invoices=rows[: payload.limit], truncated=len(rows) > payload.limit)


def search_bank_transactions(
    store: LedgerStore, payload: s.SearchBankTransactionsInput
) -> s.SearchBankTransactionsOutput:
    rows = store.search_transactions(
        memo_contains=payload.memo_contains,
        amount_cents=payload.amount_cents,
        tolerance_cents=payload.tolerance_cents,
        currency=payload.currency,
        value_from=payload.value_from,
        value_to=payload.value_to,
        matched_invoice_id=payload.matched_invoice_id,
        limit=payload.limit + 1,
    )
    return s.SearchBankTransactionsOutput(
        transactions=rows[: payload.limit], truncated=len(rows) > payload.limit
    )


def extract_invoice_reference(
    store: LedgerStore, payload: s.ExtractInvoiceReferenceInput
) -> s.ExtractInvoiceReferenceOutput:
    if payload.candidate_invoice_ids is None:
        invoice_ids = list(store.invoices)
    else:
        invoice_ids = [inv_id for inv_id in payload.candidate_invoice_ids if inv_id in store.invoices]
    candidates = rank_reference_candidates(
        payload.memo, invoice_ids, top_k=payload.top_k, min_score=payload.min_score
    )
    return s.ExtractInvoiceReferenceOutput(
        candidates=[
            s.ReferenceCandidate(invoice_id=c.invoice_id, score=c.score, matched_text=c.matched_text)
            for c in candidates
        ]
    )


def get_customer_policy(
    store: LedgerStore, payload: s.GetCustomerPolicyInput
) -> s.GetCustomerPolicyOutput:
    return s.GetCustomerPolicyOutput(policy=store.get_policy(payload.customer_id))


def get_accounting_policy(
    store: LedgerStore, payload: s.GetAccountingPolicyInput
) -> s.GetAccountingPolicyOutput:
    return s.GetAccountingPolicyOutput(policy=ACCOUNTING_POLICY)


def get_fx_rate(store: LedgerStore, payload: s.GetFxRateInput) -> s.GetFxRateOutput:
    rate = store.get_fx_rate(payload.base, payload.quote, payload.on_date)
    return s.GetFxRateOutput(
        base=rate.base,
        quote=rate.quote,
        on_date=rate.on_date,
        rate_ppm=rate.rate_ppm,
        converted_hint="quote_cents = (base_cents * rate_ppm + 500_000) // 1_000_000",
    )


def record_resolution(store: LedgerStore, payload: s.RecordResolutionInput) -> s.RecordResolutionOutput:
    resolution = Resolution(
        exception_id=payload.exception_id,
        resolution_type=payload.resolution_type,
        invoice_id=payload.invoice_id,
        txn_id=payload.txn_id,
        adjustment_cents=payload.adjustment_cents,
        rationale=payload.rationale,
        confidence=payload.confidence,
    )
    exc = store.record_resolution(resolution)
    invoice = store.get_invoice(payload.invoice_id)
    return s.RecordResolutionOutput(
        exception_id=exc.exception_id,
        status=exc.status.value,
        invoice_status=invoice.status,
        message=(
            f"Booked {payload.resolution_type.value} on {exc.exception_id} "
            f"with a {payload.adjustment_cents} cent adjustment. This exception is now closed."
        ),
    )


def escalate_exception(
    store: LedgerStore, payload: s.EscalateExceptionInput
) -> s.EscalateExceptionOutput:
    exc = store.escalate(
        Escalation(
            exception_id=payload.exception_id,
            reason=payload.reason,
            severity=payload.severity,
            suggested_owner=payload.suggested_owner,
        )
    )
    return s.EscalateExceptionOutput(
        exception_id=exc.exception_id,
        status=exc.status.value,
        message=f"Routed {exc.exception_id} to {payload.suggested_owner} at {payload.severity} severity.",
    )
