"""The loop's guarantees, exercised with policies that misbehave on purpose."""

from __future__ import annotations

import pytest

from reconagent.agent.loop import run_episode
from reconagent.agent.policies import RulesPolicy
from reconagent.agent.state import AgentState, PolicyDecision
from reconagent.agent.trace import (
    GUARD_MAX_STEPS,
    GUARD_POLICY_ERROR,
    GUARD_POLICY_GAVE_UP,
    GUARD_REPEAT_LOOP,
    STOP_ESCALATED,
    STOP_RESOLVED,
)
from reconagent.budget import Budget
from reconagent.models import ExceptionStatus
from reconagent.tools.faults import FaultInjector
from reconagent.tools.registry import build_registry

FAST = Budget(tool_backoff_base_s=0.0)


class _Spinner:
    """Calls a harmless read tool forever, with a different argument each time."""

    name = "spinner"
    model = "rules-v1"

    def next_action(self, state: AgentState) -> PolicyDecision:
        return PolicyDecision(
            tool_name="list_open_exceptions",
            arguments={"limit": (len(state.turns) % 40) + 1},
            reasoning="spinning",
        )


class _Repeater:
    name = "repeater"
    model = "rules-v1"

    def next_action(self, state: AgentState) -> PolicyDecision:
        return PolicyDecision(tool_name="get_accounting_policy", arguments={}, reasoning="again")


class _Quitter:
    name = "quitter"
    model = "rules-v1"

    def next_action(self, state: AgentState) -> PolicyDecision:
        return PolicyDecision(tool_name=None, reasoning="I do not know what to do.")


class _Exploder:
    name = "exploder"
    model = "rules-v1"

    def next_action(self, state: AgentState) -> PolicyDecision:
        raise RuntimeError("policy blew up")


@pytest.fixture
def first_exception(ledger):
    return ledger.list_exceptions(status=ExceptionStatus.OPEN, limit=1)[0].exception_id


@pytest.mark.parametrize(
    ("policy", "expected_guard"),
    [
        (_Spinner(), GUARD_MAX_STEPS),
        (_Repeater(), GUARD_REPEAT_LOOP),
        (_Quitter(), GUARD_POLICY_GAVE_UP),
        (_Exploder(), GUARD_POLICY_ERROR),
    ],
    ids=["spins", "repeats", "quits", "explodes"],
)
def test_a_misbehaving_policy_still_ends_in_an_escalation(
    registry, ledger, first_exception, policy, expected_guard
):
    run = run_episode(first_exception, registry, policy, Budget(max_steps=6, tool_backoff_base_s=0.0))
    assert run.guard_trigger == expected_guard
    assert run.forced_escalation is True
    assert run.stop_reason == STOP_ESCALATED
    assert ledger.get_exception(first_exception).status is ExceptionStatus.ESCALATED


def test_the_safety_net_is_not_blocked_by_the_budget_that_fired_it(registry, ledger, first_exception):
    run = run_episode(first_exception, registry, _Spinner(), Budget(max_steps=2, max_tool_calls=2))
    assert run.stop_reason == STOP_ESCALATED
    assert run.steps[-1].tool_name == "escalate_exception"
    assert run.steps[-1].ok


def test_every_episode_ends_with_exactly_one_terminal_write(world):
    store, tasks = world
    policy = RulesPolicy()
    for task in tasks[:20]:
        registry = build_registry(store.clone())
        run = run_episode(task.exception_id, registry, policy, FAST)
        terminal = [s for s in run.steps if s.tool_name in {"record_resolution", "escalate_exception"}]
        assert len(terminal) == 1, run.transcript()
        assert run.stop_reason in {STOP_RESOLVED, STOP_ESCALATED}


def test_retryable_failures_are_retried_and_the_episode_survives(world):
    store, tasks = world
    registry = build_registry(store.clone(), faults=FaultInjector(rate=0.9, seed=3, hard_share=0.0))
    run = run_episode(tasks[0].exception_id, registry, RulesPolicy(), FAST)
    assert run.retries > 0
    assert run.stop_reason in {STOP_RESOLVED, STOP_ESCALATED}


def test_non_retryable_failures_are_not_retried(world):
    store, tasks = world
    registry = build_registry(store.clone(), faults=FaultInjector(rate=1.0, seed=5, hard_share=1.0))
    run = run_episode(tasks[0].exception_id, registry, RulesPolicy(), FAST)
    assert run.retries == 0
    assert all(step.attempts == 1 for step in run.steps)
    assert run.stop_reason == STOP_ESCALATED


def test_tool_error_budget_stops_the_loop(world):
    store, tasks = world
    registry = build_registry(store.clone(), faults=FaultInjector(rate=1.0, seed=9, hard_share=1.0))
    run = run_episode(
        tasks[0].exception_id,
        registry,
        _Spinner(),
        Budget(max_steps=50, max_tool_errors=2, tool_backoff_base_s=0.0),
    )
    assert run.guard_trigger == "tool_error_budget"
    assert run.forced_escalation


def test_cost_and_token_totals_are_reported(world):
    store, tasks = world
    run = run_episode(tasks[0].exception_id, build_registry(store.clone()), RulesPolicy(), FAST)
    assert run.input_tokens > 0
    assert run.cost_usd == 0.0, "the offline policy must bill nothing"
    assert run.projected_cost_usd > 0.0, "but it should still say what it would have cost"


def test_the_rules_policy_reaches_a_disposition_without_the_safety_net(world):
    store, tasks = world
    task = next(t for t in tasks if t.archetype == "bank_fee")
    run = run_episode(task.exception_id, build_registry(store.clone()), RulesPolicy(), FAST)
    assert run.guard_trigger is None
    assert run.forced_escalation is False
    assert run.terminal_arguments["resolution_type"] == "bank_fee"
