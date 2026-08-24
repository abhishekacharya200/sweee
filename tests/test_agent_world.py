"""The generated world has to be reproducible and internally honest.

If ground truth and the corpus can drift apart, every eval number downstream
is decoration.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import pytest

from reconagent.matching import normalize_ref, rank_reference_candidates
from reconagent.models import ExceptionStatus, InvoiceStatus
from reconagent.world import ACCOUNTING_POLICY, ARCHETYPES, build_world


def test_the_same_seed_builds_the_same_world():
    first_store, first_tasks = build_world(4242)
    second_store, second_tasks = build_world(4242)
    assert [t.task_id for t in first_tasks] == [t.task_id for t in second_tasks]
    assert [t.archetype for t in first_tasks] == [t.archetype for t in second_tasks]
    assert first_store.invoices.keys() == second_store.invoices.keys()
    assert all(
        first_store.invoices[k].amount_cents == second_store.invoices[k].amount_cents
        for k in first_store.invoices
    )


def test_different_seeds_build_different_worlds():
    _, a = build_world(1)
    _, b = build_world(2)
    assert [t.archetype for t in a] != [t.archetype for t in b]


def test_every_archetype_is_represented(world):
    _, tasks = world
    counts = Counter(t.archetype for t in tasks)
    assert set(counts) == set(ARCHETYPES)
    assert all(count >= 5 for count in counts.values())


def test_the_queue_matches_the_golden_set(world):
    store, tasks = world
    assert len(store.exceptions) == len(tasks)
    assert {t.exception_id for t in tasks} == set(store.exceptions)
    assert all(exc.status is ExceptionStatus.OPEN for exc in store.exceptions.values())


def test_ground_truth_points_at_records_that_exist(world):
    store, tasks = world
    for task in tasks:
        if task.expected_invoice_id:
            assert task.expected_invoice_id in store.invoices, task.task_id
        if task.expected_txn_id:
            assert task.expected_txn_id in store.transactions, task.task_id
        assert task.expected_escalation == (task.expected_resolution_type is None)


def test_clones_do_not_share_state(world):
    store, tasks = world
    first, second = store.clone(), store.clone()
    exception_id = tasks[0].exception_id
    first.get_exception(exception_id).status = ExceptionStatus.RESOLVED
    assert second.get_exception(exception_id).status is ExceptionStatus.OPEN


@pytest.mark.parametrize("archetype", ["bank_fee", "short_payment", "material_short_payment"])
def test_shortfall_archetypes_land_on_the_right_side_of_the_limit(world, archetype):
    store, tasks = world
    limit = ACCOUNTING_POLICY.short_payment_writeoff_max_cents
    for task in (t for t in tasks if t.archetype == archetype):
        invoice = store.invoices[task.expected_invoice_id]
        txn = store.transactions[task.expected_txn_id]
        shortfall = invoice.amount_cents - txn.amount_cents
        assert shortfall > 0
        if archetype == "material_short_payment":
            assert shortfall > limit
        else:
            assert shortfall <= limit


def test_fx_archetypes_are_within_the_published_tolerance(world):
    store, tasks = world
    for task in (t for t in tasks if t.archetype == "fx_variance"):
        invoice = store.invoices[task.expected_invoice_id]
        txn = store.transactions[task.expected_txn_id]
        assert invoice.currency != txn.currency
        rate = store.get_fx_rate(invoice.currency, txn.currency, txn.value_date)
        converted = rate.convert_cents(invoice.amount_cents)
        tolerance = (converted * ACCOUNTING_POLICY.fx_variance_tolerance_bps) // 10_000
        assert abs(converted - txn.amount_cents) <= tolerance


def test_duplicate_archetypes_have_a_prior_settlement(world):
    store, tasks = world
    for task in (t for t in tasks if t.archetype == "duplicate_payment"):
        invoice = store.invoices[task.expected_invoice_id]
        assert invoice.status is InvoiceStatus.PAID
        prior = [t for t in invoice.settled_txn_ids if t != task.expected_txn_id]
        assert prior, task.task_id
        gap = store.transactions[task.expected_txn_id].value_date - store.transactions[prior[0]].value_date
        assert 0 < gap.days <= ACCOUNTING_POLICY.duplicate_lookback_days


def test_unmatchable_credits_really_have_no_invoice_at_that_amount(world):
    store, tasks = world
    for task in (t for t in tasks if t.archetype == "no_matching_invoice"):
        txn = store.transactions[task.expected_txn_id]
        assert not store.search_invoices(amount_cents=txn.amount_cents, tolerance_cents=0)


def test_ambiguous_credits_match_more_than_one_customer(world):
    store, tasks = world
    for task in (t for t in tasks if t.archetype == "ambiguous_multi_match"):
        txn = store.transactions[task.expected_txn_id]
        matches = store.search_invoices(amount_cents=txn.amount_cents, tolerance_cents=0)
        assert len({inv.customer_id for inv in matches}) > 1


def test_fx_conversion_is_integer_exact(world):
    store, _ = world
    rate = store.get_fx_rate("EUR", "USD", next(d for (b, q, d) in store.fx_rates if b == "EUR"))
    assert isinstance(rate.convert_cents(1_000_000), int)


def test_reference_normalisation_repairs_only_the_numeric_tail():
    assert normalize_ref("INV-2026-0142") == "INV20260142"
    assert normalize_ref("inv 2026 0142") == "INV20260142"
    assert normalize_ref("/RFB/INV2O26O142") == "RFBINV20260142"


def test_mangled_references_still_rank_their_own_invoice_first():
    invoice_ids = [f"INV-2026-{n:04d}" for n in range(100, 140)]
    target = "INV-2026-0117"
    for memo in (
        "ACH CREDIT REF INV-2026-0117 ACME",
        "SEPA CREDIT TRANSFER REF INV20260117 ACME",
        "WIRE TRANSFER IN REF INV 2026 0117 ACME",
        "BACS CREDIT REF /RFB/INV-2026-0117 ACME",
        "SWIFT MT103 CREDIT REF INV2026O117 ACME",
    ):
        candidates = rank_reference_candidates(memo, invoice_ids, top_k=3)
        assert candidates and candidates[0].invoice_id == target, memo
        assert candidates[0].score >= 0.99


def test_references_from_another_year_do_not_clear_the_acceptance_bar():
    from reconagent.agent.policies import REF_ACCEPT_SCORE

    invoice_ids = [f"INV-2026-{n:04d}" for n in range(100, 140)]
    candidates = rank_reference_candidates("ACH CREDIT REF INV-2019-7155 ORRERY", invoice_ids)
    assert all(c.score < REF_ACCEPT_SCORE for c in candidates)


def test_memos_without_a_reference_produce_no_candidates():
    invoice_ids = [f"INV-2026-{n:04d}" for n in range(100, 140)]
    assert rank_reference_candidates("SEPA CREDIT TRANSFER FROM CUSTOMER ACCOUNT", invoice_ids) == []


def test_business_dates_only_carry_published_rates(world):
    store, _ = world
    weekend = next(
        (d for (b, q, d) in store.fx_rates if d.weekday() >= 5),
        None,
    )
    assert weekend is None
    assert all(isinstance(d, date) for (_, _, d) in store.fx_rates)
