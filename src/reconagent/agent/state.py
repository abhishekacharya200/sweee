"""What a policy is allowed to see.

Both policies get the same object. The rules policy reads structured tool
payloads out of it; the LLM policy renders it into a messages array. Neither
can see a `GoldenTask` — the eval harness holds ground truth and the agent
never touches it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..tools.registry import ToolOutcome


@dataclass
class Turn:
    tool_name: str
    arguments: dict
    reasoning: str
    outcome: ToolOutcome
    tool_use_id: str | None = None


@dataclass
class AgentState:
    exception_id: str
    system_prompt: str
    task_prompt: str
    turns: list[Turn] = field(default_factory=list)

    def attempted(self, tool_name: str) -> bool:
        """True once a tool has been tried, successfully or not.

        Policies branch on this rather than on success so a tool that is hard
        down does not get asked forever.
        """
        return any(turn.tool_name == tool_name for turn in self.turns)

    def payloads(self, tool_name: str) -> list[dict]:
        return [
            turn.outcome.payload
            for turn in self.turns
            if turn.tool_name == tool_name and turn.outcome.ok and turn.outcome.payload is not None
        ]

    def last_payload(self, tool_name: str) -> dict | None:
        found = self.payloads(tool_name)
        return found[-1] if found else None

    def last_failure(self, tool_name: str) -> ToolOutcome | None:
        failures = [t.outcome for t in self.turns if t.tool_name == tool_name and not t.outcome.ok]
        return failures[-1] if failures else None

    def call_signature_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for turn in self.turns:
            key = (turn.tool_name, json.dumps(turn.arguments, sort_keys=True, default=str))
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass
class PolicyDecision:
    """A policy's next move: always a tool call, or an explicit surrender.

    There is no third option on purpose — a policy that "just responds" mid
    task is how exceptions get silently dropped.
    """

    tool_name: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_use_id: str | None = None

    @property
    def gave_up(self) -> bool:
        return self.tool_name is None
