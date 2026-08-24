"""Pydantic input/output schemas for every tool.

These models are the contract, and they are the *only* contract. The
Anthropic tool schema, the MCP `inputSchema`, the pydantic-ai tool signature,
the runtime argument validation and the docs all derive from what is written
here — nothing is hand-transcribed into a second format, because the moment a
tool schema exists in two places one of them is wrong.

Field descriptions are prompt surface, not comments: they are the only thing
telling a model that money is in cents and that `adjustment_cents` is signed.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from ..models import (
    AccountingPolicy,
    BankTransaction,
    CustomerPolicy,
    ExceptionKind,
    Invoice,
    InvoiceStatus,
    ReconciliationException,
    ResolutionType,
)


class ListOpenExceptionsInput(BaseModel):
    kind: ExceptionKind | None = Field(default=None, description="Restrict to one symptom class.")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum queue rows to return.")


class ListOpenExceptionsOutput(BaseModel):
    exceptions: list[ReconciliationException]
    total_open: int


class GetExceptionInput(BaseModel):
    exception_id: str = Field(description="Queue id, e.g. 'EXC-0007'.")


class GetExceptionOutput(BaseModel):
    exception: ReconciliationException
    invoice: Invoice | None = Field(default=None, description="Present only if the queue already linked one.")
    transaction: BankTransaction | None = None


class GetInvoiceInput(BaseModel):
    invoice_id: str = Field(description="AR ledger id, e.g. 'INV-2026-0142'.")


class GetInvoiceOutput(BaseModel):
    invoice: Invoice


class GetBankTransactionInput(BaseModel):
    txn_id: str = Field(description="Bank statement id, e.g. 'BTX-000123'.")


class GetBankTransactionOutput(BaseModel):
    transaction: BankTransaction


class SearchInvoicesInput(BaseModel):
    customer_id: str | None = None
    amount_cents: int | None = Field(default=None, description="Target invoice amount in cents.")
    tolerance_cents: int = Field(
        default=0, ge=0, le=1_000_000, description="Absolute cents either side of amount_cents."
    )
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. 'USD'.")
    status: InvoiceStatus | None = None
    issued_from: date | None = None
    issued_to: date | None = None
    limit: int = Field(default=10, ge=1, le=50)


class SearchInvoicesOutput(BaseModel):
    invoices: list[Invoice]
    truncated: bool = Field(description="True if more rows matched than were returned.")


class SearchBankTransactionsInput(BaseModel):
    memo_contains: str | None = Field(default=None, description="Case-insensitive substring of the memo.")
    amount_cents: int | None = None
    tolerance_cents: int = Field(default=0, ge=0, le=1_000_000)
    currency: str | None = None
    value_from: date | None = None
    value_to: date | None = None
    matched_invoice_id: str | None = Field(
        default=None, description="Only transactions already matched to this invoice."
    )
    limit: int = Field(default=10, ge=1, le=50)


class SearchBankTransactionsOutput(BaseModel):
    transactions: list[BankTransaction]
    truncated: bool


class ExtractInvoiceReferenceInput(BaseModel):
    memo: str = Field(description="Raw bank memo text to mine for an invoice reference.")
    top_k: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Drop candidates scoring below this similarity."
    )


class ReferenceCandidate(BaseModel):
    invoice_id: str
    score: float = Field(description="0-1 similarity after normalising separators and digit look-alikes.")
    matched_text: str = Field(description="The fragment of the memo this candidate was scored against.")


class ExtractInvoiceReferenceOutput(BaseModel):
    candidates: list[ReferenceCandidate]
    note: str = Field(
        default=(
            "A high score means the memo text looks like this id, not that this is the right invoice. "
            "Confirm against amount, currency and customer before matching."
        )
    )


class GetCustomerPolicyInput(BaseModel):
    customer_id: str = Field(description="Customer id, e.g. 'CUST-004'.")


class GetCustomerPolicyOutput(BaseModel):
    policy: CustomerPolicy


class GetAccountingPolicyInput(BaseModel):
    """No arguments: the thresholds are entity-wide."""


class GetAccountingPolicyOutput(BaseModel):
    policy: AccountingPolicy


class GetFxRateInput(BaseModel):
    base: str = Field(description="Currency the invoice is denominated in, e.g. 'EUR'.")
    quote: str = Field(description="Currency the cash actually arrived in, e.g. 'USD'.")
    on_date: date = Field(description="Use the transaction's value date, not today.")


class GetFxRateOutput(BaseModel):
    base: str
    quote: str
    on_date: date
    rate_ppm: int = Field(description="Parts per million: 1_082_000 means 1 base = 1.082 quote.")
    converted_hint: str = Field(
        description="How to apply it: quote_cents = round(base_cents * rate_ppm / 1_000_000)."
    )


class RecordResolutionInput(BaseModel):
    exception_id: str
    resolution_type: ResolutionType = Field(
        description=(
            "match_payment: amounts agree, just link them. "
            "bank_fee: shortfall equals the contracted wire fee. "
            "fx_variance: shortfall is the FX residual after converting at the value-date rate. "
            "short_payment: unexplained shortfall inside the write-off limit. "
            "duplicate_payment: the invoice was already settled by an earlier credit."
        )
    )
    invoice_id: str
    txn_id: str
    adjustment_cents: int = Field(
        default=0,
        description=(
            "Signed ledger adjustment in cents, from the company's side: 0 for a clean match, "
            "positive for a shortfall written off (fee, FX residual, short payment), "
            "negative for cash owed back (duplicate)."
        ),
    )
    rationale: str = Field(min_length=10, description="One sentence an auditor can read, citing the evidence.")
    confidence: float = Field(ge=0.0, le=1.0)


class RecordResolutionOutput(BaseModel):
    exception_id: str
    status: str
    invoice_status: InvoiceStatus
    message: str


class EscalateExceptionInput(BaseModel):
    exception_id: str
    reason: str = Field(min_length=10, description="What is missing or contradictory, specifically.")
    severity: str = Field(default="medium", pattern="^(low|medium|high)$")
    suggested_owner: str = Field(
        default="ar_operations", description="Queue to route to, e.g. 'credit_control'."
    )


class EscalateExceptionOutput(BaseModel):
    exception_id: str
    status: str
    message: str
