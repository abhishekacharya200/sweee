"""The execution trace.

A step trace is not logging garnish here: `stop_reason` and `guard_trigger`
are the eval harness's primary failure taxonomy, and `forced_escalation` is
how the system proves it never dropped an exception on the floor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Why an episode stopped taking actions.
STOP_RESOLVED = "resolved"
STOP_ESCALATED = "escalated"
STOP_ABANDONED = "abandoned"

# Which bound fired, if any. `None` means the agent stopped on its own terms.
GUARD_MAX_STEPS = "max_steps"
GUARD_MAX_TOOL_CALLS = "max_tool_calls"
GUARD_TOOL_ERRORS = "tool_error_budget"
GUARD_COST = "cost_budget"
GUARD_WALL_CLOCK = "wall_clock"
GUARD_REPEAT_LOOP = "repeat_loop"
GUARD_POLICY_ERROR = "policy_error"
GUARD_POLICY_GAVE_UP = "policy_gave_up"


@dataclass
class AgentStep:
    index: int
    tool_name: str
    arguments: dict
    reasoning: str
    ok: bool
    attempts: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    error: str | None = None
    error_kind: str | None = None
    result_digest: str = ""


@dataclass
class AgentRun:
    task_id: str
    exception_id: str
    policy: str
    model: str
    stop_reason: str
    guard_trigger: str | None = None
    forced_escalation: bool = False
    terminal_tool: str | None = None
    terminal_arguments: dict = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    projected_cost_usd: float = 0.0
    wall_clock_s: float = 0.0

    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    @property
    def tool_errors(self) -> int:
        return sum(1 for step in self.steps if not step.ok)

    @property
    def retries(self) -> int:
        return sum(step.attempts - 1 for step in self.steps)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tool_calls"] = self.tool_calls
        data["tool_errors"] = self.tool_errors
        data["retries"] = self.retries
        return data

    def transcript(self) -> str:
        """Human-readable replay of one episode, for debugging a bad decision."""
        lines = [f"[{self.task_id}] {self.exception_id} via {self.policy} ({self.model})"]
        for step in self.steps:
            status = "ok" if step.ok else f"ERR {step.error_kind}"
            args = json.dumps(step.arguments, sort_keys=True)
            lines.append(f"  {step.index:>2}. {step.tool_name}({args}) -> {status}")
            if step.reasoning:
                lines.append(f"      why: {step.reasoning}")
            if not step.ok:
                lines.append(f"      err: {step.error}")
        guard = f" [guard: {self.guard_trigger}]" if self.guard_trigger else ""
        lines.append(f"  => {self.stop_reason}{guard}")
        return "\n".join(lines)
