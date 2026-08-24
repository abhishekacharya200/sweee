"""Budgets and cost accounting.

An agent loop without a budget is an outage waiting for a bad prompt. Every
limit that can stop this loop is declared in one dataclass so the answer to
"what stops it?" is a single object, not a scatter of `if step > 10` checks.

Cost is priced from an explicit table rather than read back from a provider
response, so the offline run can still report what the same token counts
*would* have cost on a real model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    # model -> (input $/1M, output $/1M)
    "rules-v1": (0.0, 0.0),
    "naive-v1": (0.0, 0.0),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
}

_encoding = None
_encoding_unavailable = False


def count_tokens(text: str) -> int:
    global _encoding, _encoding_unavailable
    if not _encoding_unavailable and _encoding is None:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - any failure means "use the char estimate"
            _encoding_unavailable = True
    if _encoding is not None:
        return len(_encoding.encode(text))
    return max(1, round(len(text) / 4))


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_1M_TOKENS.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


@dataclass
class Budget:
    """Every way this loop is allowed to stop early."""

    max_steps: int = 14
    max_tool_calls: int = 20
    max_tool_errors: int = 6
    max_identical_calls: int = 2
    max_cost_usd: float = 0.25
    max_wall_clock_s: float = 90.0
    tool_max_attempts: int = 3
    tool_backoff_base_s: float = 0.05


@dataclass
class CostLedger:
    """Running token and dollar totals for one episode.

    `projected_cost_usd` prices the same traffic at `projection_model`, which
    is what makes an offline $0.00 run still say something useful about the
    bill: "this task shape costs $0.004 on Sonnet".
    """

    model: str
    projection_model: str = "claude-sonnet-5"
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def cost_usd(self) -> float:
        return price(self.model, self.input_tokens, self.output_tokens)

    @property
    def projected_cost_usd(self) -> float:
        return price(self.projection_model, self.input_tokens, self.output_tokens)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at
