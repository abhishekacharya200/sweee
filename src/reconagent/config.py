from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AgentSettings:
    """Env-driven configuration for the reconciliation agent.

    Defaults are chosen so `make agent-eval` runs offline, deterministically
    and at $0 — the same constraint Project A is built under, for the same
    reason: an eval gate that needs a secret cannot run in CI.
    """

    policy: str = field(default_factory=lambda: os.getenv("AGENT_POLICY", "rules"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    anthropic_model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "claude-sonnet-5"))

    max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "14")))
    max_tool_calls: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_CALLS", "20")))
    max_tool_errors: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_ERRORS", "6")))
    max_cost_usd: float = field(default_factory=lambda: float(os.getenv("AGENT_MAX_COST_USD", "0.25")))
    max_wall_clock_s: float = field(default_factory=lambda: float(os.getenv("AGENT_MAX_WALL_CLOCK_S", "90")))

    tool_max_attempts: int = field(default_factory=lambda: int(os.getenv("AGENT_TOOL_MAX_ATTEMPTS", "3")))
    tool_backoff_base_s: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TOOL_BACKOFF_BASE_S", "0.05"))
    )

    world_seed: int = field(default_factory=lambda: int(os.getenv("AGENT_WORLD_SEED", "20260301")))
    fault_rate: float = field(default_factory=lambda: float(os.getenv("AGENT_FAULT_RATE", "0.12")))

    eval_dir: Path = field(default_factory=lambda: ROOT_DIR / os.getenv("AGENT_EVAL_DIR", "eval/agent"))

    def resolve_policy(self) -> str:
        """Fall back to the offline rules policy when the LLM has no key."""
        if self.policy in {"anthropic", "pydantic-ai"} and not self.anthropic_api_key:
            return "rules"
        return self.policy


settings = AgentSettings()
