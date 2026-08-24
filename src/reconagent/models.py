"""Domain model for the reconciliation queue.

Money is integer cents everywhere. There is no float arithmetic on money in
this codebase, deliberately: a reconciliation agent that reports a 1-cent
variance because of binary floating point is worse than useless.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class InvoiceStatus(str, Enum):
    OPEN = "open"
    PAID = "paid"
    PARTIALLY_PAID = "partially_paid"
    VOID = "void"


class ExceptionKind(str, Enum):
    """The *symptom* the nightly reconciliation job observed.

    This is deliberately not the answer: the job can see that a payment did
    not match, but not why. Deciding the why is the agent's job.
    """

    UNMATCHED_PAYMENT = "unmatched_payment"
    UNMATCHED_INVOICE = "unmatched_invoice"
    AMOUNT_MISMATCH = "amount_mismatch"
    POSSIBLE_DUPLICATE = "possible_duplicate"


class ExceptionStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ResolutionType(str, Enum):
    """The dispositions the agent is allowed to write back."""

    MATCH_PAYMENT = "match_payment"
    BANK_FEE = "bank_fee"
    FX_VARIANCE = "fx_variance"
    SHORT_PAYMENT = "short_payment"
    DUPLICATE_PAYMENT = "duplicate_payment"


class Invoice(BaseModel):
    invoice_id: str
    customer_id: str
    customer_name: str
    amount_cents: int
    currency: str
    issued_date: date
    due_date: date
    status: InvoiceStatus = InvoiceStatus.OPEN
    settled_txn_ids: list[str] = Field(default_factory=list)


class BankTransaction(BaseModel):
    txn_id: str
    value_date: date
    amount_cents: int = Field(description="Positive for credits received.")
    currency: str
    memo: str = Field(description="Free-text bank memo, as received. Often mangled.")
    counterparty: str
    matched_invoice_id: str | None = None


class CustomerPolicy(BaseModel):
    """Contract terms that explain otherwise-suspicious variances."""

    customer_id: str
    customer_name: str
    payment_terms_days: int
    pays_wire_fees: bool = Field(
        description="If false, the customer deducts the wire fee from the remittance."
    )
    wire_fee_cents: int
    settlement_currency: str


class AccountingPolicy(BaseModel):
    """Company-wide thresholds that decide a disposition.

    These live in a tool rather than as constants in the agent so the same
    agent can be dropped into an entity with different materiality limits.
    """

    fx_variance_tolerance_bps: int = Field(
        description="Residual after FX conversion, in basis points of invoice value, still bookable as FX."
    )
    fee_variance_tolerance_cents: int = Field(
        description="How far a shortfall may sit from the contracted wire fee and still be a fee."
    )
    short_payment_writeoff_max_cents: int = Field(
        description="Shortfalls above this are material and must be escalated, not written off."
    )
    duplicate_lookback_days: int = Field(
        description="Window in which a second identical credit counts as a duplicate."
    )


class ReconciliationException(BaseModel):
    exception_id: str
    kind: ExceptionKind
    opened_on: date
    invoice_id: str | None = None
    txn_id: str | None = None
    delta_cents: int | None = Field(
        default=None,
        description="invoice.amount_cents - txn.amount_cents when both sides are known.",
    )
    note: str = ""
    status: ExceptionStatus = ExceptionStatus.OPEN


class Resolution(BaseModel):
    exception_id: str
    resolution_type: ResolutionType
    invoice_id: str
    txn_id: str
    adjustment_cents: int = Field(
        default=0,
        description="Signed write-off/adjustment posted to the ledger, in cents.",
    )
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class Escalation(BaseModel):
    exception_id: str
    reason: str
    severity: str
    suggested_owner: str


class FxRate(BaseModel):
    base: str
    quote: str
    on_date: date
    rate_ppm: int = Field(
        description="Rate expressed in parts-per-million so conversion stays integer-exact."
    )

    def convert_cents(self, amount_cents: int) -> int:
        """base -> quote, half-up, no floats."""
        return (amount_cents * self.rate_ppm + 500_000) // 1_000_000
