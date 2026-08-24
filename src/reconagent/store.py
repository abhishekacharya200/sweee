"""The two data sources plus the exception queue, behind one in-process API.

In a real deployment `LedgerStore` is two adapters (an ERP read replica and a
bank statement feed) and a queue table. It is kept in-process here so the eval
harness can run thousands of episodes deterministically, and so every tool has
one obvious place to fail from.
"""

from __future__ import annotations

from datetime import date

from .models import (
    BankTransaction,
    CustomerPolicy,
    Escalation,
    ExceptionKind,
    ExceptionStatus,
    FxRate,
    Invoice,
    InvoiceStatus,
    ReconciliationException,
    Resolution,
)


class LedgerError(Exception):
    """A tool-visible domain failure: bad id, illegal write, unknown rate."""


class LedgerStore:
    def __init__(
        self,
        invoices: dict[str, Invoice],
        transactions: dict[str, BankTransaction],
        policies: dict[str, CustomerPolicy],
        fx_rates: dict[tuple[str, str, date], FxRate],
        exceptions: dict[str, ReconciliationException],
    ) -> None:
        self.invoices = invoices
        self.transactions = transactions
        self.policies = policies
        self.fx_rates = fx_rates
        self.exceptions = exceptions
        self.resolutions: dict[str, Resolution] = {}
        self.escalations: dict[str, Escalation] = {}

    def clone(self) -> LedgerStore:
        """A fresh world for one agent episode — writes must not leak between tasks."""
        return LedgerStore(
            invoices={k: v.model_copy(deep=True) for k, v in self.invoices.items()},
            transactions={k: v.model_copy(deep=True) for k, v in self.transactions.items()},
            policies=dict(self.policies),
            fx_rates=dict(self.fx_rates),
            exceptions={k: v.model_copy(deep=True) for k, v in self.exceptions.items()},
        )

    # ---- reads -----------------------------------------------------------

    def get_invoice(self, invoice_id: str) -> Invoice:
        try:
            return self.invoices[invoice_id]
        except KeyError:
            raise LedgerError(f"No invoice with id {invoice_id!r} exists in the AR ledger.") from None

    def get_transaction(self, txn_id: str) -> BankTransaction:
        try:
            return self.transactions[txn_id]
        except KeyError:
            raise LedgerError(f"No bank transaction with id {txn_id!r} exists.") from None

    def get_exception(self, exception_id: str) -> ReconciliationException:
        try:
            return self.exceptions[exception_id]
        except KeyError:
            raise LedgerError(f"No exception with id {exception_id!r} is in the queue.") from None

    def get_policy(self, customer_id: str) -> CustomerPolicy:
        try:
            return self.policies[customer_id]
        except KeyError:
            raise LedgerError(f"No contract policy on file for customer {customer_id!r}.") from None

    def get_fx_rate(self, base: str, quote: str, on_date: date) -> FxRate:
        rate = self.fx_rates.get((base.upper(), quote.upper(), on_date))
        if rate is None:
            raise LedgerError(
                f"No published {base.upper()}/{quote.upper()} rate for {on_date.isoformat()} "
                "(rates are published on business days only)."
            )
        return rate

    def list_exceptions(
        self,
        status: ExceptionStatus | None = None,
        kind: ExceptionKind | None = None,
        limit: int = 20,
    ) -> list[ReconciliationException]:
        rows = [
            exc
            for exc in self.exceptions.values()
            if (status is None or exc.status == status) and (kind is None or exc.kind == kind)
        ]
        rows.sort(key=lambda e: (e.opened_on, e.exception_id))
        return rows[:limit]

    def search_invoices(
        self,
        customer_id: str | None = None,
        amount_cents: int | None = None,
        tolerance_cents: int = 0,
        currency: str | None = None,
        status: InvoiceStatus | None = None,
        issued_from: date | None = None,
        issued_to: date | None = None,
        limit: int = 10,
    ) -> list[Invoice]:
        rows: list[Invoice] = []
        for inv in self.invoices.values():
            if customer_id is not None and inv.customer_id != customer_id:
                continue
            if currency is not None and inv.currency != currency.upper():
                continue
            if status is not None and inv.status != status:
                continue
            if amount_cents is not None and abs(inv.amount_cents - amount_cents) > tolerance_cents:
                continue
            if issued_from is not None and inv.issued_date < issued_from:
                continue
            if issued_to is not None and inv.issued_date > issued_to:
                continue
            rows.append(inv)
        rows.sort(key=lambda i: i.invoice_id)
        return rows[:limit]

    def search_transactions(
        self,
        memo_contains: str | None = None,
        amount_cents: int | None = None,
        tolerance_cents: int = 0,
        currency: str | None = None,
        value_from: date | None = None,
        value_to: date | None = None,
        matched_invoice_id: str | None = None,
        limit: int = 10,
    ) -> list[BankTransaction]:
        needle = memo_contains.upper() if memo_contains else None
        rows: list[BankTransaction] = []
        for txn in self.transactions.values():
            if needle is not None and needle not in txn.memo.upper():
                continue
            if currency is not None and txn.currency != currency.upper():
                continue
            if amount_cents is not None and abs(txn.amount_cents - amount_cents) > tolerance_cents:
                continue
            if value_from is not None and txn.value_date < value_from:
                continue
            if value_to is not None and txn.value_date > value_to:
                continue
            if matched_invoice_id is not None and txn.matched_invoice_id != matched_invoice_id:
                continue
            rows.append(txn)
        rows.sort(key=lambda t: (t.value_date, t.txn_id))
        return rows[:limit]

    # ---- writes ----------------------------------------------------------

    def record_resolution(self, resolution: Resolution) -> ReconciliationException:
        exc = self.get_exception(resolution.exception_id)
        if exc.status is not ExceptionStatus.OPEN:
            raise LedgerError(
                f"Exception {exc.exception_id} is already {exc.status.value}; it cannot be written twice."
            )
        invoice = self.get_invoice(resolution.invoice_id)
        txn = self.get_transaction(resolution.txn_id)
        if txn.matched_invoice_id not in (None, invoice.invoice_id):
            raise LedgerError(
                f"Transaction {txn.txn_id} is already matched to {txn.matched_invoice_id}; "
                "resolve the existing match before re-linking it."
            )

        txn.matched_invoice_id = invoice.invoice_id
        if txn.txn_id not in invoice.settled_txn_ids:
            invoice.settled_txn_ids.append(txn.txn_id)
        invoice.status = (
            InvoiceStatus.PARTIALLY_PAID if resolution.adjustment_cents > 0 else InvoiceStatus.PAID
        )
        exc.status = ExceptionStatus.RESOLVED
        self.resolutions[exc.exception_id] = resolution
        return exc

    def escalate(self, escalation: Escalation) -> ReconciliationException:
        exc = self.get_exception(escalation.exception_id)
        if exc.status is not ExceptionStatus.OPEN:
            raise LedgerError(
                f"Exception {exc.exception_id} is already {exc.status.value}; it cannot be escalated."
            )
        exc.status = ExceptionStatus.ESCALATED
        self.escalations[exc.exception_id] = escalation
        return exc
