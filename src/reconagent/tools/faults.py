"""Deterministic fault injection for the tool surface.

Agent loops are easy to make look good against tools that always answer. The
interesting behaviour — retry with backoff, give up and take the safe route,
never leave an exception silently unhandled — only shows up when the tools
misbehave, so misbehaviour is a first-class, seeded feature here rather than
something to be discovered in production.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


class ToolError(Exception):
    """Base class for failures raised from inside a tool handler."""

    retryable = False


class TransientToolError(ToolError):
    """Upstream blipped. Worth another attempt."""

    retryable = True


class HardToolError(ToolError):
    """Upstream is down for this call. Retrying will not help; route around it."""

    retryable = False


@dataclass
class FaultInjector:
    """Fails a seeded fraction of tool calls.

    `hard_share` of those failures are non-retryable, which is what forces the
    agent to have a plan for "this evidence is simply unavailable" instead of
    only a retry loop.
    """

    rate: float = 0.0
    seed: int = 0
    hard_share: float = 0.2
    exempt_tools: frozenset[str] = field(default_factory=lambda: frozenset({"record_resolution", "escalate_exception"}))

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.injected_transient = 0
        self.injected_hard = 0

    def maybe_fail(self, tool_name: str) -> None:
        # Terminal writes are exempt: injecting a write failure would test the
        # store's idempotency, not the agent's reasoning, and would let an
        # exception fall off the queue entirely.
        if self.rate <= 0 or tool_name in self.exempt_tools:
            return
        if self._rng.random() >= self.rate:
            return
        if self._rng.random() < self.hard_share:
            self.injected_hard += 1
            raise HardToolError(
                f"Upstream service backing {tool_name!r} is unavailable (503). "
                "This will not resolve on retry — proceed with the evidence you already have."
            )
        self.injected_transient += 1
        raise TransientToolError(f"Timed out calling {tool_name!r} after 5000ms.")
