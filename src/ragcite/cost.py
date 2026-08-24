"""Token counting and $-cost accounting per query.

Pricing is expressed as USD per 1M tokens and is deliberately explicit and
editable (rather than hidden inside a provider SDK) since the whole point
of tracking it is to make cost auditable for a regulated deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_PER_1M_TOKENS = {
    # provider, model -> (input $/1M, output $/1M)
    ("mock", "extractive-v1"): (0.0, 0.0),
    ("anthropic", "claude-sonnet-5"): (3.00, 15.00),
    ("anthropic", "claude-haiku-4-5"): (0.80, 4.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
}

_encoding = None
_encoding_unavailable = False


def _get_encoding():
    global _encoding, _encoding_unavailable
    if _encoding_unavailable:
        return None
    if _encoding is None:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - any failure means "fall back to the char estimate"
            _encoding_unavailable = True
            return None
    return _encoding


def count_tokens(text: str) -> int:
    """Best-effort token count: real BPE tokenization via tiktoken when
    available, else a ~4-chars-per-token approximation so the pipeline
    still works fully offline."""
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, round(len(text) / 4))


def estimate_cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_1M_TOKENS.get((provider, model))
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


@dataclass
class Stopwatch:
    """Tiny helper for per-stage latency bookkeeping."""

    marks: dict

    def __init__(self):
        self.marks = {}

    def lap(self, name: str, seconds: float) -> None:
        self.marks[name] = seconds

    def total(self) -> float:
        return sum(self.marks.values())
