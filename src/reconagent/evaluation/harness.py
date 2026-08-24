"""The eval harness: run a policy over the whole queue and measure it.

Three things are measured because three things can regress independently:
whether the agent gets the answer right, how much work it does to get there,
and whether it stays safe when it does not know. A harness that reports only
the first will pass a change that doubles the token bill or starts booking
guesses.

Every suite is deterministic. The world seed fixes the queue, the fault seed
fixes which tool calls fail, and both are recorded in the result, so a number
in the report can always be reproduced.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..agent.loop import run_episode
from ..agent.policies import build_policy
from ..agent.trace import STOP_ABANDONED, AgentRun
from ..budget import Budget, price, tokenizer_name
from ..config import settings
from ..store import LedgerStore
from ..tools.faults import FaultInjector
from ..tools.registry import build_registry
from ..world import GoldenTask, build_world
from .scoring import FAILURE_MODES, Verdict, score_run


@dataclass(frozen=True)
class Profile:
    """A named operating condition to measure the same policy under."""

    name: str
    fault_rate: float
    description: str


PROFILES: dict[str, Profile] = {
    "clean": Profile("clean", 0.0, "Every tool answers. Measures reasoning quality alone."),
    "chaos": Profile(
        "chaos",
        settings.fault_rate,
        f"{settings.fault_rate:.0%} of read calls fail; one in five of those is non-retryable. "
        "Measures failure handling.",
    ),
}

DEFAULT_GATE_THRESHOLDS = {
    "task_success_min": 0.90,
    "escalation_recall_min": 0.85,
    "missed_escalation_rate_max": 0.05,
    "wrong_linkage_rate_max": 0.02,
    "unhandled_rate_max": 0.0,
}


@dataclass
class TaskResult:
    task: GoldenTask
    run: AgentRun
    verdict: Verdict

    def to_dict(self) -> dict:
        return {
            "task_id": self.task.task_id,
            "exception_id": self.task.exception_id,
            "archetype": self.task.archetype,
            "expected": (
                "escalate"
                if self.task.expected_escalation
                else self.task.expected_resolution_type.value
            ),
            "actual": self.verdict.actual_disposition,
            "success": self.verdict.success,
            "failure_mode": self.verdict.failure_mode,
            "detail": self.verdict.detail,
            "steps": self.run.tool_calls,
            "retries": self.run.retries,
            "tool_errors": self.run.tool_errors,
            "guard_trigger": self.run.guard_trigger,
            "forced_escalation": self.run.forced_escalation,
            "input_tokens": self.run.input_tokens,
            "output_tokens": self.run.output_tokens,
            "cost_usd": round(self.run.cost_usd, 6),
            "projected_cost_usd": round(self.run.projected_cost_usd, 6),
            "wall_clock_s": round(self.run.wall_clock_s, 4),
        }


@dataclass
class SuiteResult:
    policy: str
    model: str
    profile: str
    world_seed: int
    results: list[TaskResult] = field(default_factory=list)

    @property
    def metrics(self) -> dict:
        total = len(self.results)
        if total == 0:
            return {}
        modes = {mode: 0 for mode in FAILURE_MODES}
        for result in self.results:
            if result.verdict.failure_mode:
                modes[result.verdict.failure_mode] += 1

        should_escalate = [r for r in self.results if r.task.expected_escalation]
        did_escalate = [r for r in self.results if r.verdict.actual_disposition == "escalate"]
        correct_escalations = [r for r in did_escalate if r.task.expected_escalation]

        return {
            "tasks": total,
            "task_success": _ratio(sum(r.verdict.success for r in self.results), total),
            "escalation_recall": _ratio(len(correct_escalations), len(should_escalate)),
            "escalation_precision": _ratio(len(correct_escalations), len(did_escalate)),
            "missed_escalation_rate": _ratio(modes["missed_escalation"], total),
            "wrong_linkage_rate": _ratio(modes["wrong_linkage"], total),
            "unhandled_rate": _ratio(
                sum(1 for r in self.results if r.run.stop_reason == STOP_ABANDONED), total
            ),
            "forced_escalation_rate": _ratio(
                sum(1 for r in self.results if r.run.forced_escalation), total
            ),
            "mean_steps": round(statistics.mean(r.run.tool_calls for r in self.results), 2),
            "mean_tool_errors": round(statistics.mean(r.run.tool_errors for r in self.results), 2),
            "mean_retries": round(statistics.mean(r.run.retries for r in self.results), 2),
            "mean_latency_s": round(statistics.mean(r.run.wall_clock_s for r in self.results), 4),
            "cost_usd_per_task": round(
                statistics.mean(r.run.cost_usd for r in self.results), 6
            ),
            "projected_cost_usd_per_task": round(
                statistics.mean(r.run.projected_cost_usd for r in self.results), 6
            ),
            "mean_input_tokens": round(statistics.mean(r.run.input_tokens for r in self.results)),
            "mean_output_tokens": round(statistics.mean(r.run.output_tokens for r in self.results)),
            "failure_modes": {mode: count for mode, count in modes.items() if count},
        }

    def by_archetype(self) -> dict[str, dict]:
        buckets: dict[str, list[TaskResult]] = {}
        for result in self.results:
            buckets.setdefault(result.task.archetype, []).append(result)
        return {
            archetype: {
                "tasks": len(rows),
                "success": _ratio(sum(r.verdict.success for r in rows), len(rows)),
                "mean_steps": round(statistics.mean(r.run.tool_calls for r in rows), 2),
            }
            for archetype, rows in sorted(buckets.items())
        }

    def guard_histogram(self) -> dict[str, int]:
        histogram: dict[str, int] = {}
        for result in self.results:
            if result.run.guard_trigger:
                histogram[result.run.guard_trigger] = histogram.get(result.run.guard_trigger, 0) + 1
        return dict(sorted(histogram.items()))

    def failures(self) -> list[TaskResult]:
        return [r for r in self.results if not r.verdict.success]

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "model": self.model,
            "profile": self.profile,
            "world_seed": self.world_seed,
            "metrics": self.metrics,
            "by_archetype": self.by_archetype(),
            "guard_triggers": self.guard_histogram(),
            "tasks": [r.to_dict() for r in self.results],
        }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


# Anthropic prompt caching bills a cache write at 1.25x the input rate and a
# cache read at 0.1x. Kept here rather than worked out in prose so the saving
# claimed in the docs is recomputed on every run instead of ageing.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def cost_basis(suite: SuiteResult, projection_model: str = "claude-sonnet-5") -> dict:
    """Where the token bill actually goes, and what caching would do to it.

    The system prompt and the tool schemas are re-sent on every turn of every
    episode. On a 12-tool surface and a ~5-turn episode that fixed prefix is
    most of the input bill, which makes it a bigger lever than anything about
    the agent loop.
    """
    from ..agent.policies import fixed_prefix_tokens

    metrics = suite.metrics
    if not metrics:
        return {}
    prefix = fixed_prefix_tokens()
    steps = metrics["mean_steps"]
    mean_input = metrics["mean_input_tokens"]
    resent = prefix * steps

    cached_input = mean_input - resent + prefix * (
        CACHE_WRITE_MULTIPLIER + CACHE_READ_MULTIPLIER * max(0.0, steps - 1)
    )
    baseline = price(projection_model, mean_input, metrics["mean_output_tokens"])
    cached = price(projection_model, round(cached_input), metrics["mean_output_tokens"])
    return {
        "tokenizer": tokenizer_name(),
        "projection_model": projection_model,
        "fixed_prefix_tokens": prefix,
        "mean_input_tokens": mean_input,
        "resent_prefix_tokens": round(resent),
        "resent_prefix_share": _ratio(round(resent), mean_input),
        "projected_cost_usd_per_task": round(baseline, 6),
        "projected_cost_usd_per_task_cached": round(cached, 6),
        "cache_saving": round(1 - cached / baseline, 4) if baseline else 0.0,
    }


def run_suite(
    policy_name: str = "rules",
    *,
    profile: str = "clean",
    world_seed: int = 20260301,
    budget: Budget | None = None,
    limit: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
    runner: str = "loop",
    projection_model: str = "claude-sonnet-5",
) -> SuiteResult:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Available: {', '.join(PROFILES)}.")
    faults = PROFILES[profile].fault_rate
    world, tasks = build_world(world_seed)
    budget = budget or settings.budget()
    if limit is not None:
        tasks = tasks[:limit]

    if runner == "framework":
        return _run_framework_suite(
            world, tasks, profile, world_seed, faults, budget, model, api_key, projection_model
        )

    policy = build_policy(policy_name, model=model, api_key=api_key)
    suite = SuiteResult(
        policy=policy.name, model=policy.model, profile=profile, world_seed=world_seed
    )
    for index, task in enumerate(tasks):
        registry = build_registry(
            world.clone(),
            faults=FaultInjector(rate=faults, seed=world_seed + index) if faults else None,
        )
        run = run_episode(
            task.exception_id,
            registry,
            policy,
            budget,
            task_id=task.task_id,
            projection_model=projection_model,
        )
        suite.results.append(TaskResult(task=task, run=run, verdict=score_run(task, run)))
    return suite


def _run_framework_suite(
    world: LedgerStore,
    tasks: list[GoldenTask],
    profile: str,
    world_seed: int,
    faults: float,
    budget: Budget | None,
    model: str | None,
    api_key: str | None,
    projection_model: str,
) -> SuiteResult:
    from ..agent.framework import FRAMEWORK_NAME, run_framework_episode

    model = model or "claude-sonnet-5"
    suite = SuiteResult(policy=FRAMEWORK_NAME, model=model, profile=profile, world_seed=world_seed)
    for index, task in enumerate(tasks):
        registry = build_registry(
            world.clone(),
            faults=FaultInjector(rate=faults, seed=world_seed + index) if faults else None,
        )
        run = run_framework_episode(
            task.exception_id,
            registry,
            model=model,
            budget=budget,
            api_key=api_key,
            task_id=task.task_id,
            projection_model=projection_model,
        )
        suite.results.append(TaskResult(task=task, run=run, verdict=score_run(task, run)))
    return suite


def gate(suite: SuiteResult, thresholds: dict | None = None) -> tuple[bool, list[str]]:
    """Pass/fail with the reasons, so CI prints why rather than just red."""
    thresholds = thresholds or DEFAULT_GATE_THRESHOLDS
    metrics = suite.metrics
    failures: list[str] = []
    for key, threshold in thresholds.items():
        if key.endswith("_min"):
            metric = key[: -len("_min")]
            if metrics.get(metric, 0.0) < threshold:
                failures.append(f"{metric} {metrics.get(metric, 0.0):.3f} < {threshold:.3f}")
        elif key.endswith("_max"):
            metric = key[: -len("_max")]
            if metrics.get(metric, 0.0) > threshold:
                failures.append(f"{metric} {metrics.get(metric, 0.0):.3f} > {threshold:.3f}")
    return not failures, failures
