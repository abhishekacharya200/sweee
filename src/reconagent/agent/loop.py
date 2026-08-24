"""The hand-rolled agent loop.

Written before reaching for a framework, and kept, because the interesting
behaviour of this system lives in about forty lines of it: what stops the
loop, what happens to the work when it stops, and what a failing tool does to
the next decision.

Two invariants hold no matter how badly a policy or a tool behaves:

1. Every episode ends in exactly one terminal write. If a bound fires, the
   loop escalates on the agent's behalf rather than returning empty-handed —
   an exception that silently falls out of the queue is the one outcome an AR
   team cannot detect.
2. The safety-net escalation is not itself subject to the budget it is
   reacting to. A guard that could be blocked by the guard it fired for is
   not a guard.
"""

from __future__ import annotations

import json
import time

from ..budget import Budget, CostLedger
from ..tools.registry import TERMINAL_TOOLS, ToolOutcome, ToolRegistry
from .policies import Policy
from .prompts import SYSTEM_PROMPT, build_task_prompt
from .state import AgentState, Turn
from .trace import (
    GUARD_COST,
    GUARD_MAX_STEPS,
    GUARD_MAX_TOOL_CALLS,
    GUARD_POLICY_ERROR,
    GUARD_POLICY_GAVE_UP,
    GUARD_REPEAT_LOOP,
    GUARD_TOOL_ERRORS,
    GUARD_WALL_CLOCK,
    STOP_ABANDONED,
    STOP_ESCALATED,
    STOP_RESOLVED,
    AgentRun,
    AgentStep,
)

_GUARD_EXPLANATIONS = {
    GUARD_MAX_STEPS: "the agent used its full step budget without reaching a disposition",
    GUARD_MAX_TOOL_CALLS: "the agent used its full tool-call budget without reaching a disposition",
    GUARD_TOOL_ERRORS: "too many tool calls failed for the agent to gather usable evidence",
    GUARD_COST: "the agent hit its cost ceiling before reaching a disposition",
    GUARD_WALL_CLOCK: "the agent ran out of time before reaching a disposition",
    GUARD_REPEAT_LOOP: "the agent repeated an identical tool call and stopped making progress",
    GUARD_POLICY_ERROR: "the agent policy raised an error mid-task",
    GUARD_POLICY_GAVE_UP: "the agent stopped without choosing a disposition",
}


def _digest(outcome: ToolOutcome, limit: int = 160) -> str:
    text = outcome.to_model_text()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def execute_with_retries(
    registry: ToolRegistry, tool_name: str, arguments: dict, budget: Budget
) -> ToolOutcome:
    """Retry only what is worth retrying, with exponential backoff.

    A 503 gets another go; a validation error or a missing invoice does not,
    because the second call has exactly as much chance as the first.
    """
    delay = budget.tool_backoff_base_s
    outcome = registry.call(tool_name, arguments)
    for attempt in range(2, budget.tool_max_attempts + 1):
        if outcome.ok or not outcome.retryable:
            break
        time.sleep(delay)
        delay *= 2
        outcome = registry.call(tool_name, arguments)
        outcome.attempts = attempt
    return outcome


def run_episode(
    exception_id: str,
    registry: ToolRegistry,
    policy: Policy,
    budget: Budget | None = None,
    *,
    task_id: str = "",
    projection_model: str = "claude-sonnet-5",
) -> AgentRun:
    budget = budget or Budget()
    ledger = CostLedger(model=policy.model, projection_model=projection_model)
    state = AgentState(
        exception_id=exception_id,
        system_prompt=SYSTEM_PROMPT,
        task_prompt=build_task_prompt(exception_id),
    )
    run = AgentRun(
        task_id=task_id or exception_id,
        exception_id=exception_id,
        policy=policy.name,
        model=policy.model,
        stop_reason=STOP_ABANDONED,
    )

    tool_calls = 0
    tool_errors = 0

    while True:
        guard = _check_budget(budget, run, ledger, tool_calls, tool_errors)
        if guard:
            escalate_as_safety_net(registry, exception_id, guard, run, ledger)
            break

        try:
            decision = policy.next_action(state)
        except Exception as exc:  # noqa: BLE001 - a broken policy must not lose the exception
            escalate_as_safety_net(
                registry, exception_id, GUARD_POLICY_ERROR, run, ledger, detail=str(exc)
            )
            break

        ledger.add(decision.input_tokens, decision.output_tokens)
        if decision.gave_up:
            escalate_as_safety_net(
                registry, exception_id, GUARD_POLICY_GAVE_UP, run, ledger, detail=decision.reasoning
            )
            break

        outcome = execute_with_retries(registry, decision.tool_name, decision.arguments, budget)
        tool_calls += outcome.attempts
        if not outcome.ok:
            tool_errors += 1

        run.steps.append(
            AgentStep(
                index=len(run.steps) + 1,
                tool_name=decision.tool_name,
                arguments=decision.arguments,
                reasoning=decision.reasoning,
                ok=outcome.ok,
                attempts=outcome.attempts,
                latency_s=outcome.latency_s,
                input_tokens=decision.input_tokens,
                output_tokens=decision.output_tokens,
                error=outcome.error,
                error_kind=outcome.error_kind,
                result_digest=_digest(outcome),
            )
        )
        state.turns.append(
            Turn(
                tool_name=decision.tool_name,
                arguments=decision.arguments,
                reasoning=decision.reasoning,
                outcome=outcome,
                tool_use_id=decision.tool_use_id,
            )
        )

        if outcome.ok and decision.tool_name in TERMINAL_TOOLS:
            run.stop_reason = (
                STOP_RESOLVED if decision.tool_name == "record_resolution" else STOP_ESCALATED
            )
            run.terminal_tool = decision.tool_name
            run.terminal_arguments = decision.arguments
            break

        signature = (decision.tool_name, json.dumps(decision.arguments, sort_keys=True, default=str))
        if state.call_signature_counts().get(signature, 0) > budget.max_identical_calls:
            escalate_as_safety_net(registry, exception_id, GUARD_REPEAT_LOOP, run, ledger)
            break

    run.input_tokens = ledger.input_tokens
    run.output_tokens = ledger.output_tokens
    run.cost_usd = ledger.cost_usd
    run.projected_cost_usd = ledger.projected_cost_usd
    run.wall_clock_s = ledger.elapsed_s
    return run


def _check_budget(
    budget: Budget, run: AgentRun, ledger: CostLedger, tool_calls: int, tool_errors: int
) -> str | None:
    if len(run.steps) >= budget.max_steps:
        return GUARD_MAX_STEPS
    if tool_calls >= budget.max_tool_calls:
        return GUARD_MAX_TOOL_CALLS
    if tool_errors > budget.max_tool_errors:
        return GUARD_TOOL_ERRORS
    if ledger.cost_usd > budget.max_cost_usd:
        return GUARD_COST
    if ledger.elapsed_s > budget.max_wall_clock_s:
        return GUARD_WALL_CLOCK
    return None


def escalate_as_safety_net(
    registry: ToolRegistry,
    exception_id: str,
    guard: str,
    run: AgentRun,
    ledger: CostLedger,
    *,
    detail: str = "",
) -> None:
    """Escalate on the agent's behalf so the exception stays owned by someone."""
    run.guard_trigger = guard
    explanation = _GUARD_EXPLANATIONS.get(guard, guard)
    reason = f"Automatically escalated: {explanation}."
    if detail:
        reason = f"{reason} Detail: {detail[:300]}"

    outcome = registry.call(
        "escalate_exception",
        {
            "exception_id": exception_id,
            "reason": reason,
            "severity": "high",
            "suggested_owner": "ar_operations",
        },
    )
    run.steps.append(
        AgentStep(
            index=len(run.steps) + 1,
            tool_name="escalate_exception",
            arguments={"exception_id": exception_id, "reason": reason},
            reasoning=f"Safety net fired on {guard}.",
            ok=outcome.ok,
            attempts=outcome.attempts,
            latency_s=outcome.latency_s,
            input_tokens=0,
            output_tokens=0,
            error=outcome.error,
            error_kind=outcome.error_kind,
            result_digest=_digest(outcome),
        )
    )
    run.forced_escalation = True
    if outcome.ok:
        run.stop_reason = STOP_ESCALATED
        run.terminal_tool = "escalate_exception"
        run.terminal_arguments = {"exception_id": exception_id, "reason": reason}
    else:
        run.stop_reason = STOP_ABANDONED


def run_queue(
    registry: ToolRegistry,
    policy: Policy,
    budget: Budget | None = None,
    *,
    limit: int = 10,
    projection_model: str = "claude-sonnet-5",
) -> list[AgentRun]:
    """Drain the open queue, one episode per exception.

    Episodes deliberately do not share context: an exception is a unit of work
    with its own budget, and letting turn history accumulate across a shift is
    how a queue-draining agent's cost goes quadratic.
    """
    listing = registry.call("list_open_exceptions", {"limit": limit})
    if not listing.ok:
        raise RuntimeError(f"Could not read the exception queue: {listing.error}")
    exception_ids = [row["exception_id"] for row in listing.payload["exceptions"]]
    return [
        run_episode(
            exception_id,
            registry,
            policy,
            budget,
            task_id=exception_id,
            projection_model=projection_model,
        )
        for exception_id in exception_ids
    ]
