"""The harness has to be able to tell a good agent from a careless one."""

from __future__ import annotations

import json

import pytest

from reconagent.agent.trace import AgentRun
from reconagent.evaluation import gate, render_json, render_markdown, run_suite, score_run
from reconagent.evaluation.harness import DEFAULT_GATE_THRESHOLDS, PROFILES
from reconagent.models import ResolutionType
from reconagent.world import GoldenTask


@pytest.fixture(scope="module")
def rules_suite():
    return run_suite("rules", profile="clean")


@pytest.fixture(scope="module")
def naive_suite():
    return run_suite("naive", profile="clean")


def _task(**overrides) -> GoldenTask:
    base = {
        "task_id": "T001",
        "exception_id": "EXC-0001",
        "archetype": "bank_fee",
        "expected_escalation": False,
        "expected_resolution_type": ResolutionType.BANK_FEE,
        "expected_invoice_id": "INV-2026-0001",
        "expected_txn_id": "BTX-000001",
        "expected_adjustment_cents": 2500,
        "adjustment_tolerance_cents": 0,
        "rationale": "",
    }
    base.update(overrides)
    return GoldenTask(**base)


def _run(**overrides) -> AgentRun:
    run = AgentRun(
        task_id="T001",
        exception_id="EXC-0001",
        policy="test",
        model="rules-v1",
        stop_reason=overrides.pop("stop_reason", "resolved"),
        terminal_tool=overrides.pop("terminal_tool", "record_resolution"),
    )
    run.terminal_arguments = overrides.pop(
        "terminal_arguments",
        {
            "resolution_type": "bank_fee",
            "invoice_id": "INV-2026-0001",
            "txn_id": "BTX-000001",
            "adjustment_cents": 2500,
        },
    )
    return run


def test_a_correct_booking_scores_as_success():
    assert score_run(_task(), _run()).success


@pytest.mark.parametrize(
    ("overrides", "expected_mode"),
    [
        ({"terminal_arguments": {"resolution_type": "short_payment", "invoice_id": "INV-2026-0001",
                                 "txn_id": "BTX-000001", "adjustment_cents": 2500}}, "wrong_disposition"),
        ({"terminal_arguments": {"resolution_type": "bank_fee", "invoice_id": "INV-2026-0009",
                                 "txn_id": "BTX-000001", "adjustment_cents": 2500}}, "wrong_linkage"),
        ({"terminal_arguments": {"resolution_type": "bank_fee", "invoice_id": "INV-2026-0001",
                                 "txn_id": "BTX-000001", "adjustment_cents": 9999}}, "wrong_adjustment"),
        ({"terminal_tool": "escalate_exception", "stop_reason": "escalated",
          "terminal_arguments": {}}, "over_escalation"),
        ({"terminal_tool": None, "stop_reason": "abandoned", "terminal_arguments": {}}, "unhandled"),
    ],
    ids=["disposition", "linkage", "adjustment", "over-escalation", "unhandled"],
)
def test_the_failure_taxonomy_separates_the_ways_of_being_wrong(overrides, expected_mode):
    verdict = score_run(_task(), _run(**overrides))
    assert not verdict.success
    assert verdict.failure_mode == expected_mode


def test_booking_a_case_that_needed_a_human_is_its_own_failure_mode():
    task = _task(expected_escalation=True, expected_resolution_type=None,
                 expected_adjustment_cents=None, archetype="material_short_payment")
    assert score_run(task, _run()).failure_mode == "missed_escalation"


def test_partial_credit_is_not_available():
    """Right invoice, wrong number is a wrong journal entry, not a near miss."""
    verdict = score_run(
        _task(),
        _run(terminal_arguments={"resolution_type": "bank_fee", "invoice_id": "INV-2026-0001",
                                 "txn_id": "BTX-000001", "adjustment_cents": 2501}),
    )
    assert verdict.success is False


def test_the_rules_policy_clears_the_gate(rules_suite):
    passed, reasons = gate(rules_suite, DEFAULT_GATE_THRESHOLDS)
    assert passed, reasons


def test_the_naive_baseline_does_not_clear_the_gate(naive_suite):
    passed, reasons = gate(naive_suite, DEFAULT_GATE_THRESHOLDS)
    assert not passed
    assert any("task_success" in reason for reason in reasons)


def test_the_gap_between_the_two_is_large_enough_to_measure(rules_suite, naive_suite):
    assert rules_suite.metrics["task_success"] - naive_suite.metrics["task_success"] > 0.5


def test_no_episode_is_ever_left_unhandled(rules_suite, naive_suite):
    for suite in (rules_suite, naive_suite):
        assert suite.metrics["unhandled_rate"] == 0.0


def test_the_naive_baseline_books_guesses_that_needed_a_human(naive_suite):
    assert naive_suite.metrics["missed_escalation_rate"] > 0.1


def test_cost_and_effort_are_reported_per_task(rules_suite):
    metrics = rules_suite.metrics
    assert metrics["mean_steps"] > 1
    assert metrics["cost_usd_per_task"] == 0.0
    assert metrics["projected_cost_usd_per_task"] > 0.0
    assert metrics["mean_input_tokens"] > 0


def test_every_archetype_is_scored(rules_suite):
    breakdown = rules_suite.by_archetype()
    assert len(breakdown) == 8
    assert all(entry["tasks"] > 0 for entry in breakdown.values())


def test_injected_faults_cost_accuracy_but_never_safety():
    """Under chaos the agent should escalate more, and misbook no more."""
    chaos = run_suite("rules", profile="chaos", limit=40)
    assert chaos.metrics["missed_escalation_rate"] == 0.0
    assert chaos.metrics["wrong_linkage_rate"] == 0.0
    assert chaos.metrics["unhandled_rate"] == 0.0
    assert chaos.metrics["mean_retries"] > 0


def test_suites_are_reproducible():
    first = run_suite("rules", profile="chaos", limit=25)
    second = run_suite("rules", profile="chaos", limit=25)
    assert first.metrics["task_success"] == second.metrics["task_success"]
    assert first.metrics["mean_retries"] == second.metrics["mean_retries"]


def test_reports_render(rules_suite, naive_suite):
    markdown = render_markdown([rules_suite, naive_suite])
    assert "## Success by archetype" in markdown
    assert "rules / clean" in markdown
    payload = json.loads(render_json([rules_suite]))
    assert payload["suites"][0]["policy"] == "rules"
    assert len(payload["suites"][0]["tasks"]) == rules_suite.metrics["tasks"]


def test_unknown_profiles_are_rejected():
    with pytest.raises(ValueError):
        run_suite("rules", profile="hurricane")
    assert set(PROFILES) == {"clean", "chaos"}


def test_the_cost_basis_names_its_counter_and_adds_up(rules_suite):
    """A cost number without its estimator named is a number nobody can check."""
    from reconagent.budget import BPE_ENCODING, CHAR_ESTIMATE
    from reconagent.evaluation import cost_basis

    basis = cost_basis(rules_suite)
    assert basis["tokenizer"] in {BPE_ENCODING, CHAR_ESTIMATE}
    assert 0 < basis["fixed_prefix_tokens"] < basis["mean_input_tokens"]
    assert basis["resent_prefix_tokens"] <= basis["mean_input_tokens"]
    assert 0.0 < basis["resent_prefix_share"] <= 1.0
    assert basis["projected_cost_usd_per_task_cached"] < basis["projected_cost_usd_per_task"]
    assert 0.0 < basis["cache_saving"] < 1.0


def test_the_report_states_which_counter_produced_its_numbers(rules_suite):
    from reconagent.evaluation import render_markdown

    markdown = render_markdown([rules_suite])
    assert "## Cost basis" in markdown
    assert "Token counter:" in markdown
