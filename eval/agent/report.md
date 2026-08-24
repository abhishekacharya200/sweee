# Reconciliation agent evaluation

Generated 2026-08-24 11:30 UTC · world seed `20260301` · 70 tasks per suite

Projected cost prices the same token volume at `claude-sonnet-5` rates; offline policies bill $0.

## Headline

| Metric | rules / clean | naive / clean | rules / chaos |
|---|---|---|---|
| Task success | 1.000 | 0.257 | 0.900 |
| Escalation recall | 1.000 | 0.333 | 1.000 |
| Escalation precision | 1.000 | 1.000 | 0.774 |
| Missed escalations | 0.000 | 0.229 | 0.000 |
| Wrong linkage | 0.000 | 0.000 | 0.000 |
| Unhandled | 0.000 | 0.000 | 0.000 |
| Safety-net fired | 0.000 | 0.114 | 0.000 |
| Steps/task | 4.79 | 2.50 | 4.64 |
| Retries/task | 0.00 | 0.00 | 0.27 |
| Latency/task (s) | 0.0053 | 0.0002 | 0.0150 |
| Cost/task ($) | 0.0000 | 0.0000 | 0.0000 |
| Projected $/task | 0.0482 | 0.0243 | 0.0467 |

## Profiles

- **clean** — Every tool answers. Measures reasoning quality alone.
- **chaos** — 12% of read calls fail; one in five of those is non-retryable. Measures failure handling.

## Success by archetype

| Archetype | rules / clean | naive / clean | rules / chaos |
|---|---|---|---|
| `clean_match` | 1.00 (10) | 1.00 (10) | 0.80 (10) |
| `bank_fee` | 1.00 (9) | 0.00 (9) | 0.89 (9) |
| `fx_variance` | 1.00 (9) | 0.00 (9) | 1.00 (9) |
| `short_payment` | 1.00 (9) | 0.00 (9) | 0.67 (9) |
| `duplicate_payment` | 1.00 (9) | 0.00 (9) | 0.89 (9) |
| `material_short_payment` | 1.00 (8) | 0.00 (8) | 1.00 (8) |
| `ambiguous_multi_match` | 1.00 (8) | 0.00 (8) | 1.00 (8) |
| `no_matching_invoice` | 1.00 (8) | 1.00 (8) | 1.00 (8) |

## Failure modes

| Mode | rules / clean | naive / clean | rules / chaos |
|---|---|---|---|
| `missed_escalation` | 0 | 16 | 0 |
| `over_escalation` | 0 | 0 | 7 |
| `wrong_disposition` | 0 | 36 | 0 |

## Guards fired

- **rules / clean** — none
- **naive / clean** — `policy_gave_up` ×8
- **rules / chaos** — none

## Sample failures — naive / clean

**T001** (`bank_fee`) — wrong_disposition: Booked match_payment, expected bank_fee.

```
[T001] EXC-0001 via naive (naive-v1)
   1. get_exception({"exception_id": "EXC-0001"}) -> ok
      why: Load the exception.
   2. record_resolution({"adjustment_cents": 0, "confidence": 0.5, "exception_id": "EXC-0001", "invoice_id": "INV-2026-0001", "rationale": "Amount is close enough to the invoice to link them.", "resolution_type": "match_payment", "txn_id": "BTX-000001"}) -> ok
      why: Amount is close enough to the invoice to link them.
  => resolved
```

**T002** (`short_payment`) — wrong_disposition: Booked match_payment, expected short_payment.

```
[T002] EXC-0002 via naive (naive-v1)
   1. get_exception({"exception_id": "EXC-0002"}) -> ok
      why: Load the exception.
   2. record_resolution({"adjustment_cents": 0, "confidence": 0.5, "exception_id": "EXC-0002", "invoice_id": "INV-2026-0002", "rationale": "Amount is close enough to the invoice to link them.", "resolution_type": "match_payment", "txn_id": "BTX-000002"}) -> ok
      why: Amount is close enough to the invoice to link them.
  => resolved
```

**T003** (`material_short_payment`) — missed_escalation: Booked match_payment on a case that needed a human decision.

```
[T003] EXC-0003 via naive (naive-v1)
   1. get_exception({"exception_id": "EXC-0003"}) -> ok
      why: Load the exception.
   2. record_resolution({"adjustment_cents": 0, "confidence": 0.5, "exception_id": "EXC-0003", "invoice_id": "INV-2026-0003", "rationale": "Amount is close enough to the invoice to link them.", "resolution_type": "match_payment", "txn_id": "BTX-000003"}) -> ok
      why: Amount is close enough to the invoice to link them.
  => resolved
```


## Sample failures — rules / chaos

**T002** (`short_payment`) — over_escalation: Escalated a case that should have been booked as short_payment.

```
[T002] EXC-0002 via rules (rules-v1)
   1. get_exception({"exception_id": "EXC-0002"}) -> ok
      why: Load the exception under review.
   2. get_accounting_policy({}) -> ERR unavailable
      why: Read the thresholds before judging any variance.
      err: Upstream service backing 'get_accounting_policy' is unavailable (503). This will not resolve on retry — proceed with the evidence you already have.
   3. escalate_exception({"exception_id": "EXC-0002", "reason": "Entity materiality thresholds are unavailable, so no disposition can be justified.", "severity": "medium", "suggested_owner": "ar_operations"}) -> ok
      why: Evidence does not support any disposition: Entity materiality thresholds are unavailable, so no disposition can be justified.
  => escalated
```

**T023** (`clean_match`) — over_escalation: Escalated a case that should have been booked as match_payment.

```
[T023] EXC-0023 via rules (rules-v1)
   1. get_exception({"exception_id": "EXC-0023"}) -> ok
      why: Load the exception under review.
   2. get_accounting_policy({}) -> ok
      why: Read the thresholds before judging any variance.
   3. search_invoices({"amount_cents": 410000, "currency": "USD", "limit": 10, "tolerance_cents": 0}) -> ok
      why: Find every invoice the credit could settle exactly.
   4. extract_invoice_reference({"candidate_invoice_ids": ["INV-2026-0025"], "memo": "BACS CREDIT REF INV20260O25 ASHGROVE UTILITIES", "top_k": 10}) -> ERR unavailable
      why: Check whether the memo supports any of the amount matches.
      err: Upstream service backing 'extract_invoice_reference' is unavailable (503). This will not resolve on retry — proceed with the evidence you already have.
   5. escalate_exception({"exception_id": "EXC-0023", "reason": "The credit carries no remittance reference matching any of the 1 invoice(s) at this amount (INV-2026-0025), so it cannot be attributed.", "severity": "medium", "suggested_owner": "ar_operations"}) -> ok
      why: Evidence does not support any disposition: The credit carries no remittance reference matching any of the 1 invoice(s) at this amount (INV-2026-0025), so it cannot be attributed.
  => escalated
```

**T025** (`short_payment`) — over_escalation: Escalated a case that should have been booked as short_payment.

```
[T025] EXC-0025 via rules (rules-v1)
   1. get_exception({"exception_id": "EXC-0025"}) -> ok
      why: Load the exception under review.
   2. get_accounting_policy({}) -> ERR unavailable
      why: Read the thresholds before judging any variance.
      err: Upstream service backing 'get_accounting_policy' is unavailable (503). This will not resolve on retry — proceed with the evidence you already have.
   3. escalate_exception({"exception_id": "EXC-0025", "reason": "Entity materiality thresholds are unavailable, so no disposition can be justified.", "severity": "medium", "suggested_owner": "ar_operations"}) -> ok
      why: Evidence does not support any disposition: Entity materiality thresholds are unavailable, so no disposition can be justified.
  => escalated
```
