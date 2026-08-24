"""Deterministic generator for the reconciliation world and its golden set.

Every exception in the queue is built from an archetype whose correct
disposition is known by construction, so eval scoring is exact rather than
eyeballed — the same trick Project A uses for its RAG golden set.

The generator is the *only* place ground truth exists. Nothing downstream of
`build_world` can see a `GoldenTask`: the tools, the store and both agent
loops are given the same view an operator would have.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from .models import (
    AccountingPolicy,
    BankTransaction,
    CustomerPolicy,
    ExceptionKind,
    FxRate,
    Invoice,
    InvoiceStatus,
    ReconciliationException,
    ResolutionType,
)
from .store import LedgerStore

ACCOUNTING_POLICY = AccountingPolicy(
    fx_variance_tolerance_bps=75,
    fee_variance_tolerance_cents=200,
    short_payment_writeoff_max_cents=50_000,
    duplicate_lookback_days=30,
)

BOOK_CURRENCY = "USD"

_PERIOD_START = date(2026, 3, 2)
_PERIOD_END = date(2026, 3, 27)
_SETTLE_START = date(2026, 4, 6)
_SETTLE_END = date(2026, 4, 24)

_CUSTOMERS = [
    ("CUST-001", "Northwind Logistics", "USD", True),
    ("CUST-002", "Halberd Instruments", "USD", False),
    ("CUST-003", "Cedar Point Health", "USD", True),
    ("CUST-004", "Meridian Freight", "USD", False),
    ("CUST-005", "Bluecoast Analytics", "USD", True),
    ("CUST-006", "Orrery Systems", "USD", False),
    ("CUST-007", "Vantage Clinical", "USD", True),
    ("CUST-008", "Ashgrove Utilities", "USD", False),
    ("CUST-009", "Ridgeline Foods", "USD", True),
    ("CUST-010", "Kestrel Maritime", "EUR", False),
    ("CUST-011", "Lumen Verlag", "EUR", True),
    ("CUST-012", "Thames & Warden", "GBP", False),
]

_CHANNELS = ["SEPA CREDIT TRANSFER", "ACH CREDIT", "FASTER PAYMENT", "BACS CREDIT"]
_WIRE_CHANNELS = ["WIRE TRANSFER IN", "SWIFT MT103 CREDIT", "RTGS WIRE CREDIT"]

ARCHETYPES = [
    "clean_match",
    "bank_fee",
    "fx_variance",
    "short_payment",
    "duplicate_payment",
    "material_short_payment",
    "ambiguous_multi_match",
    "no_matching_invoice",
]

_ARCHETYPE_COUNTS = {
    "clean_match": 10,
    "bank_fee": 9,
    "fx_variance": 9,
    "short_payment": 9,
    "duplicate_payment": 9,
    "material_short_payment": 8,
    "ambiguous_multi_match": 8,
    "no_matching_invoice": 8,
}


@dataclass(frozen=True)
class GoldenTask:
    """One episode: an exception plus the disposition an auditor would sign off."""

    task_id: str
    exception_id: str
    archetype: str
    expected_escalation: bool
    expected_resolution_type: ResolutionType | None
    expected_invoice_id: str | None
    expected_txn_id: str | None
    expected_adjustment_cents: int | None
    adjustment_tolerance_cents: int
    rationale: str


def _business_days(start: date, end: date) -> list[date]:
    days, cursor = [], start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _mangle_ref(rng: random.Random, invoice_id: str) -> str:
    """Distort a reference the way a payment rail actually would."""
    style = rng.choice(["exact", "compact", "spaced", "slashed", "transposed", "ocr", "lower"])
    if style == "exact":
        return invoice_id
    if style == "compact":
        return invoice_id.replace("-", "")
    if style == "spaced":
        return invoice_id.replace("-", " ")
    if style == "slashed":
        return f"/RFB/{invoice_id}"
    if style == "transposed":
        head, year, tail = invoice_id.split("-")
        swapped = tail[:2] + tail[3] + tail[2] if len(tail) >= 4 else tail
        return f"{head}-{year}-{swapped}"
    if style == "ocr":
        compact = invoice_id.replace("-", "")
        idx = compact.rfind("0")
        return compact if idx < 0 else compact[:idx] + "O" + compact[idx + 1 :]
    return invoice_id.replace("-", "").lower()


class _Sequences:
    def __init__(self) -> None:
        self.invoice = 0
        self.txn = 0
        self.exception = 0

    def next_invoice_id(self) -> str:
        self.invoice += 1
        return f"INV-2026-{self.invoice:04d}"

    def next_txn_id(self) -> str:
        self.txn += 1
        return f"BTX-{self.txn:06d}"

    def next_exception_id(self) -> str:
        self.exception += 1
        return f"EXC-{self.exception:04d}"


class _WorldBuilder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.seq = _Sequences()
        self.invoices: dict[str, Invoice] = {}
        self.transactions: dict[str, BankTransaction] = {}
        self.exceptions: dict[str, ReconciliationException] = {}
        self.tasks: list[GoldenTask] = []
        self.policies = {
            cid: CustomerPolicy(
                customer_id=cid,
                customer_name=name,
                payment_terms_days=self.rng.choice([14, 30, 45, 60]),
                pays_wire_fees=pays_fees,
                wire_fee_cents=self.rng.choice([1500, 2200, 2500, 3500, 4000]),
                settlement_currency=currency,
            )
            for cid, name, currency, pays_fees in _CUSTOMERS
        }
        self.issue_days = _business_days(_PERIOD_START, _PERIOD_END)
        self.settle_days = _business_days(_SETTLE_START, _SETTLE_END)
        # Amounts are spaced $150 apart so an amount search with a $50 tolerance
        # can never collide by accident; collisions only happen where an
        # archetype creates one on purpose.
        self.amount_pool = list(range(50_000, 3_000_000, 15_000))
        self.rng.shuffle(self.amount_pool)
        self.fx_rates = self._build_fx_rates()

    # ---- primitives ------------------------------------------------------

    def _build_fx_rates(self) -> dict[tuple[str, str, date], FxRate]:
        rates: dict[tuple[str, str, date], FxRate] = {}
        for base, start_ppm in (("EUR", 1_082_000), ("GBP", 1_268_000)):
            ppm = start_ppm
            for day in _business_days(_SETTLE_START - timedelta(days=45), _SETTLE_END + timedelta(days=10)):
                ppm += self.rng.randint(-3_500, 3_500)
                rates[(base, BOOK_CURRENCY, day)] = FxRate(
                    base=base, quote=BOOK_CURRENCY, on_date=day, rate_ppm=ppm
                )
        return rates

    def _take_amount(self) -> int:
        return self.amount_pool.pop()

    def _customer(self, *, pays_wire_fees: bool | None = None, currency: str | None = None) -> CustomerPolicy:
        pool = [
            p
            for p in self.policies.values()
            if (pays_wire_fees is None or p.pays_wire_fees == pays_wire_fees)
            and (currency is None or p.settlement_currency == currency)
        ]
        return self.rng.choice(pool)

    def _add_invoice(
        self,
        policy: CustomerPolicy,
        amount_cents: int,
        *,
        currency: str | None = None,
        status: InvoiceStatus = InvoiceStatus.OPEN,
    ) -> Invoice:
        issued = self.rng.choice(self.issue_days)
        invoice = Invoice(
            invoice_id=self.seq.next_invoice_id(),
            customer_id=policy.customer_id,
            customer_name=policy.customer_name,
            amount_cents=amount_cents,
            currency=currency or policy.settlement_currency,
            issued_date=issued,
            due_date=issued + timedelta(days=policy.payment_terms_days),
            status=status,
        )
        self.invoices[invoice.invoice_id] = invoice
        return invoice

    def _add_txn(
        self,
        policy: CustomerPolicy,
        amount_cents: int,
        memo: str,
        *,
        currency: str = BOOK_CURRENCY,
        value_date: date | None = None,
        matched_invoice_id: str | None = None,
    ) -> BankTransaction:
        txn = BankTransaction(
            txn_id=self.seq.next_txn_id(),
            value_date=value_date or self.rng.choice(self.settle_days),
            amount_cents=amount_cents,
            currency=currency,
            memo=memo,
            counterparty=policy.customer_name.upper(),
            matched_invoice_id=matched_invoice_id,
        )
        self.transactions[txn.txn_id] = txn
        return txn

    def _add_exception(
        self,
        kind: ExceptionKind,
        *,
        invoice_id: str | None,
        txn_id: str | None,
        delta_cents: int | None,
        note: str,
        opened_on: date,
    ) -> ReconciliationException:
        exc = ReconciliationException(
            exception_id=self.seq.next_exception_id(),
            kind=kind,
            opened_on=opened_on,
            invoice_id=invoice_id,
            txn_id=txn_id,
            delta_cents=delta_cents,
            note=note,
        )
        self.exceptions[exc.exception_id] = exc
        return exc

    def _memo(self, ref: str | None, policy: CustomerPolicy, *, wire: bool = False) -> str:
        channel = self.rng.choice(_WIRE_CHANNELS if wire else _CHANNELS)
        short_name = policy.customer_name.upper()
        if ref is None:
            return f"{channel} FROM {short_name}"
        return f"{channel} REF {ref} {short_name}"

    def _record_task(self, **kwargs) -> None:
        self.tasks.append(GoldenTask(task_id=f"T{len(self.tasks) + 1:03d}", **kwargs))

    # ---- archetypes ------------------------------------------------------

    def _clean_match(self) -> None:
        policy = self._customer(currency=BOOK_CURRENCY)
        amount = self._take_amount()
        invoice = self._add_invoice(policy, amount)
        txn = self._add_txn(policy, amount, self._memo(_mangle_ref(self.rng, invoice.invoice_id), policy))
        exc = self._add_exception(
            ExceptionKind.UNMATCHED_PAYMENT,
            invoice_id=None,
            txn_id=txn.txn_id,
            delta_cents=None,
            note="Credit received with a reference the matcher could not resolve exactly.",
            opened_on=txn.value_date,
        )
        self._record_task(
            exception_id=exc.exception_id,
            archetype="clean_match",
            expected_escalation=False,
            expected_resolution_type=ResolutionType.MATCH_PAYMENT,
            expected_invoice_id=invoice.invoice_id,
            expected_txn_id=txn.txn_id,
            expected_adjustment_cents=0,
            adjustment_tolerance_cents=0,
            rationale="Memo reference resolves to exactly one open invoice for the same amount.",
        )

    def _bank_fee(self) -> None:
        policy = self._customer(pays_wire_fees=False, currency=BOOK_CURRENCY)
        amount = self._take_amount()
        invoice = self._add_invoice(policy, amount)
        received = amount - policy.wire_fee_cents
        txn = self._add_txn(
            policy,
            received,
            self._memo(_mangle_ref(self.rng, invoice.invoice_id), policy, wire=True),
        )
        exc = self._add_exception(
            ExceptionKind.AMOUNT_MISMATCH,
            invoice_id=invoice.invoice_id,
            txn_id=txn.txn_id,
            delta_cents=amount - received,
            note="Credit references a known invoice but settles short.",
            opened_on=txn.value_date,
        )
        self._record_task(
            exception_id=exc.exception_id,
            archetype="bank_fee",
            expected_escalation=False,
            expected_resolution_type=ResolutionType.BANK_FEE,
            expected_invoice_id=invoice.invoice_id,
            expected_txn_id=txn.txn_id,
            expected_adjustment_cents=policy.wire_fee_cents,
            adjustment_tolerance_cents=0,
            rationale="Shortfall equals the contracted wire fee and the customer does not bear fees.",
        )

    def _fx_variance(self) -> None:
        policy = self._customer(currency="EUR") if self.rng.random() < 0.6 else self._customer(currency="GBP")
        amount = self._take_amount()
        invoice = self._add_invoice(policy, amount)
        value_date = self.rng.choice(self.settle_days)
        rate = self.fx_rates[(invoice.currency, BOOK_CURRENCY, value_date)]
        converted = rate.convert_cents(amount)
        drift_cap = max(1, (converted * ACCOUNTING_POLICY.fx_variance_tolerance_bps) // 10_000)
        received = converted - self.rng.randint(-drift_cap, drift_cap)
        txn = self._add_txn(
            policy,
            received,
            self._memo(_mangle_ref(self.rng, invoice.invoice_id), policy, wire=True),
            value_date=value_date,
        )
        exc = self._add_exception(
            ExceptionKind.AMOUNT_MISMATCH,
            invoice_id=invoice.invoice_id,
            txn_id=txn.txn_id,
            delta_cents=None,
            note=(
                f"Invoice is denominated in {invoice.currency} but settled in {BOOK_CURRENCY}; "
                "amounts are not directly comparable."
            ),
            opened_on=value_date,
        )
        self._record_task(
            exception_id=exc.exception_id,
            archetype="fx_variance",
            expected_escalation=False,
            expected_resolution_type=ResolutionType.FX_VARIANCE,
            expected_invoice_id=invoice.invoice_id,
            expected_txn_id=txn.txn_id,
            expected_adjustment_cents=converted - received,
            adjustment_tolerance_cents=drift_cap,
            rationale="Residual after converting at the value-date rate sits inside the FX tolerance.",
        )

    def _short_payment(self, *, material: bool) -> None:
        policy = self._customer(pays_wire_fees=True, currency=BOOK_CURRENCY)
        amount = self._take_amount()
        invoice = self._add_invoice(policy, amount)
        limit = ACCOUNTING_POLICY.short_payment_writeoff_max_cents
        if material:
            shortfall = self.rng.randint(limit + 5_000, limit + 90_000)
        else:
            shortfall = self.rng.randint(policy.wire_fee_cents + 5_000, limit - 5_000)
        received = amount - shortfall
        txn = self._add_txn(policy, received, self._memo(_mangle_ref(self.rng, invoice.invoice_id), policy))
        exc = self._add_exception(
            ExceptionKind.AMOUNT_MISMATCH,
            invoice_id=invoice.invoice_id,
            txn_id=txn.txn_id,
            delta_cents=shortfall,
            note="Credit references a known invoice but settles short.",
            opened_on=txn.value_date,
        )
        self._record_task(
            exception_id=exc.exception_id,
            archetype="material_short_payment" if material else "short_payment",
            expected_escalation=material,
            expected_resolution_type=None if material else ResolutionType.SHORT_PAYMENT,
            expected_invoice_id=invoice.invoice_id,
            expected_txn_id=txn.txn_id,
            expected_adjustment_cents=None if material else shortfall,
            adjustment_tolerance_cents=0,
            rationale=(
                "Shortfall exceeds the write-off limit, so it needs a human credit decision."
                if material
                else "Shortfall is unexplained by fees or FX but sits under the write-off limit."
            ),
        )

    def _duplicate_payment(self) -> None:
        policy = self._customer(currency=BOOK_CURRENCY)
        amount = self._take_amount()
        invoice = self._add_invoice(policy, amount, status=InvoiceStatus.PAID)
        first_date = self.rng.choice(self.settle_days[: len(self.settle_days) // 2])
        ref = _mangle_ref(self.rng, invoice.invoice_id)
        first = self._add_txn(
            policy,
            amount,
            self._memo(ref, policy),
            value_date=first_date,
            matched_invoice_id=invoice.invoice_id,
        )
        invoice.settled_txn_ids.append(first.txn_id)
        second_date = min(first_date + timedelta(days=self.rng.choice([3, 5, 7, 9])), self.settle_days[-1])
        second = self._add_txn(policy, amount, self._memo(ref, policy), value_date=second_date)
        exc = self._add_exception(
            ExceptionKind.POSSIBLE_DUPLICATE,
            invoice_id=None,
            txn_id=second.txn_id,
            delta_cents=None,
            note="A second credit of the same amount arrived from the same counterparty.",
            opened_on=second_date,
        )
        self._record_task(
            exception_id=exc.exception_id,
            archetype="duplicate_payment",
            expected_escalation=False,
            expected_resolution_type=ResolutionType.DUPLICATE_PAYMENT,
            expected_invoice_id=invoice.invoice_id,
            expected_txn_id=second.txn_id,
            expected_adjustment_cents=-amount,
            adjustment_tolerance_cents=0,
            rationale="The invoice was already settled in full by an earlier credit inside the lookback window.",
        )

    def _ambiguous_multi_match(self) -> None:
        first_policy = self._customer(currency=BOOK_CURRENCY)
        second_policy = self._customer(currency=BOOK_CURRENCY)
        while second_policy.customer_id == first_policy.customer_id:
            second_policy = self._customer(currency=BOOK_CURRENCY)
        amount = self._take_amount()
        self._add_invoice(first_policy, amount)
        self._add_invoice(second_policy, amount)
        payer = self.rng.choice([first_policy, second_policy])
        txn = self._add_txn(payer, amount, f"{self.rng.choice(_CHANNELS)} FROM CUSTOMER ACCOUNT")
        exc = self._add_exception(
            ExceptionKind.UNMATCHED_PAYMENT,
            invoice_id=None,
            txn_id=txn.txn_id,
            delta_cents=None,
            note="Credit carries no usable remittance reference.",
            opened_on=txn.value_date,
        )
        self._record_task(
            exception_id=exc.exception_id,
            archetype="ambiguous_multi_match",
            expected_escalation=True,
            expected_resolution_type=None,
            expected_invoice_id=None,
            expected_txn_id=txn.txn_id,
            expected_adjustment_cents=None,
            adjustment_tolerance_cents=0,
            rationale="Two open invoices from different customers match the amount and nothing breaks the tie.",
        )

    def _no_matching_invoice(self) -> None:
        policy = self._customer(currency=BOOK_CURRENCY)
        amount = self._take_amount()
        phantom_ref = f"INV-2019-{self.rng.randint(1000, 9999)}"
        txn = self._add_txn(policy, amount, self._memo(phantom_ref, policy))
        exc = self._add_exception(
            ExceptionKind.UNMATCHED_PAYMENT,
            invoice_id=None,
            txn_id=txn.txn_id,
            delta_cents=None,
            note="Credit references an invoice number that is not in the AR ledger.",
            opened_on=txn.value_date,
        )
        self._record_task(
            exception_id=exc.exception_id,
            archetype="no_matching_invoice",
            expected_escalation=True,
            expected_resolution_type=None,
            expected_invoice_id=None,
            expected_txn_id=txn.txn_id,
            expected_adjustment_cents=None,
            adjustment_tolerance_cents=0,
            rationale="Neither the reference nor the amount resolves to any invoice on file.",
        )

    # ---- noise -----------------------------------------------------------

    def _add_noise(self, invoices: int, transactions: int) -> None:
        """Clean, already-reconciled records so searches return realistic noise."""
        for _ in range(invoices):
            policy = self._customer()
            self._add_invoice(policy, self._take_amount(), status=InvoiceStatus.PAID)
        for _ in range(transactions):
            policy = self._customer(currency=BOOK_CURRENCY)
            paid = [
                inv
                for inv in self.invoices.values()
                if inv.customer_id == policy.customer_id
                and inv.status is InvoiceStatus.PAID
                and not inv.settled_txn_ids
            ]
            if not paid:
                continue
            invoice = self.rng.choice(paid)
            txn = self._add_txn(
                policy,
                invoice.amount_cents,
                self._memo(invoice.invoice_id, policy),
                matched_invoice_id=invoice.invoice_id,
            )
            invoice.settled_txn_ids.append(txn.txn_id)

    def build(self) -> tuple[LedgerStore, list[GoldenTask]]:
        makers = {
            "clean_match": self._clean_match,
            "bank_fee": self._bank_fee,
            "fx_variance": self._fx_variance,
            "short_payment": lambda: self._short_payment(material=False),
            "duplicate_payment": self._duplicate_payment,
            "material_short_payment": lambda: self._short_payment(material=True),
            "ambiguous_multi_match": self._ambiguous_multi_match,
            "no_matching_invoice": self._no_matching_invoice,
        }
        plan = [name for name in ARCHETYPES for _ in range(_ARCHETYPE_COUNTS[name])]
        self.rng.shuffle(plan)
        for name in plan:
            makers[name]()
        self._add_noise(invoices=40, transactions=25)
        store = LedgerStore(
            invoices=self.invoices,
            transactions=self.transactions,
            policies=self.policies,
            fx_rates=self.fx_rates,
            exceptions=self.exceptions,
        )
        return store, self.tasks


def build_world(seed: int = 20260301) -> tuple[LedgerStore, list[GoldenTask]]:
    return _WorldBuilder(seed).build()
