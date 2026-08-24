"""The same tool surface, driven by a framework instead of the loop above.

Built with pydantic-ai because the tools are already Pydantic models, so the
comparison is about the *loop*, not about schema plumbing. What the framework
gives away for free is real: message assembly, tool dispatch, typed retries,
usage limits and streaming, none of which had to be written.

What it does not give is the two things this task actually needs, and both
show up in the code below:

- No notion of a terminal tool. The framework stops when the model produces
  final output, so ending on a successful `record_resolution` means raising
  out of the tool to unwind the run.
- No safe default on abandonment. Blowing a usage limit raises; what happens
  to the exception afterwards is still the caller's problem, so the same
  safety net from the hand-rolled loop is reattached here by hand.

Requires an API key. `build_agent` is importable and inspectable without one,
which is what the offline test asserts against.
"""

from __future__ import annotations

from typing import Any

from ..budget import Budget, CostLedger, price
from ..tools.registry import TOOL_SPECS, ToolOutcome, ToolRegistry, input_json_schema
from .loop import escalate_as_safety_net, execute_with_retries
from .prompts import SYSTEM_PROMPT, build_task_prompt
from .trace import STOP_ABANDONED, STOP_ESCALATED, STOP_RESOLVED, AgentRun, AgentStep

FRAMEWORK_NAME = "pydantic-ai"


class _TerminalReached(Exception):
    """Raised out of a terminal tool to end the framework's run.

    The framework's stop condition is "the model produced final output". Ours
    is "a terminal write succeeded". Bridging the two means unwinding.
    """

    def __init__(self, tool_name: str, arguments: dict) -> None:
        super().__init__(tool_name)
        self.tool_name = tool_name
        self.arguments = arguments


class _Recorder:
    """Collects the same step trace the hand-rolled loop produces."""

    def __init__(self) -> None:
        self.steps: list[AgentStep] = []

    def add(self, tool_name: str, arguments: dict, outcome: ToolOutcome) -> None:
        self.steps.append(
            AgentStep(
                index=len(self.steps) + 1,
                tool_name=tool_name,
                arguments=arguments,
                reasoning="",
                ok=outcome.ok,
                attempts=outcome.attempts,
                latency_s=outcome.latency_s,
                input_tokens=0,
                output_tokens=0,
                error=outcome.error,
                error_kind=outcome.error_kind,
                result_digest=outcome.to_model_text()[:160],
            )
        )


def build_agent(
    registry: ToolRegistry,
    recorder: _Recorder,
    *,
    model: str = "claude-sonnet-5",
    budget: Budget | None = None,
    api_key: str | None = None,
    defer_model_check: bool = False,
) -> Any:
    """Project `TOOL_SPECS` into a pydantic-ai agent.

    `Tool.from_schema` takes the JSON schema the registry already generates,
    so the tool surface is not re-declared here — the same argument for the
    MCP server holds: two declarations of one tool means one of them is stale.
    """
    from pydantic_ai import Agent, ModelRetry, Tool

    budget = budget or Budget()

    def make_handler(spec):
        def handler(**kwargs: Any) -> str:
            outcome = execute_with_retries(registry, spec.name, kwargs, budget)
            recorder.add(spec.name, kwargs, outcome)
            if not outcome.ok:
                # ModelRetry is the framework-native channel for "that call was
                # wrong, try differently" — cleaner than the hand-rolled loop's
                # error-as-tool-result, but it burns the framework's own retry
                # budget rather than the error budget declared in `Budget`.
                raise ModelRetry(outcome.error or "tool call failed")
            if spec.terminal:
                raise _TerminalReached(spec.name, kwargs)
            return outcome.to_model_text()

        handler.__name__ = spec.name
        return handler

    tools = [
        Tool.from_schema(
            make_handler(spec),
            name=spec.name,
            description=spec.description,
            json_schema=input_json_schema(spec),
        )
        for spec in TOOL_SPECS
    ]

    model_spec: Any = f"anthropic:{model}"
    if api_key:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        model_spec = AnthropicModel(model, provider=AnthropicProvider(api_key=api_key))

    return Agent(
        model_spec,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        defer_model_check=defer_model_check,
    )


def run_framework_episode(
    exception_id: str,
    registry: ToolRegistry,
    *,
    model: str = "claude-sonnet-5",
    budget: Budget | None = None,
    api_key: str | None = None,
    task_id: str = "",
    projection_model: str = "claude-sonnet-5",
) -> AgentRun:
    from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded, UsageLimits
    from pydantic_ai.usage import RunUsage

    budget = budget or Budget()
    recorder = _Recorder()
    agent = build_agent(registry, recorder, model=model, budget=budget, api_key=api_key)

    ledger = CostLedger(model=model, projection_model=projection_model)
    run = AgentRun(
        task_id=task_id or exception_id,
        exception_id=exception_id,
        policy=FRAMEWORK_NAME,
        model=model,
        stop_reason=STOP_ABANDONED,
    )

    # `usage` is passed in rather than read off the result so the token counts
    # survive the exception that ends a successful run.
    usage = RunUsage()
    guard: str | None = None
    detail = ""
    try:
        agent.run_sync(
            build_task_prompt(exception_id),
            usage=usage,
            usage_limits=UsageLimits(
                request_limit=budget.max_steps,
                tool_calls_limit=budget.max_tool_calls,
                cost_limit=budget.max_cost_usd,
            ),
        )
        guard, detail = "policy_gave_up", "The framework run finished without a terminal tool call."
    except _TerminalReached as terminal:
        run.terminal_tool = terminal.tool_name
        run.terminal_arguments = terminal.arguments
        run.stop_reason = (
            STOP_RESOLVED if terminal.tool_name == "record_resolution" else STOP_ESCALATED
        )
    except UsageLimitExceeded as exc:
        guard, detail = "max_tool_calls", str(exc)
    except UnexpectedModelBehavior as exc:
        guard, detail = "policy_error", str(exc)

    run.steps = recorder.steps
    run.input_tokens = usage.input_tokens
    run.output_tokens = usage.output_tokens
    ledger.add(usage.input_tokens, usage.output_tokens)

    if guard is not None:
        escalate_as_safety_net(registry, exception_id, guard, run, ledger, detail=detail)

    run.cost_usd = price(model, run.input_tokens, run.output_tokens)
    run.projected_cost_usd = ledger.projected_cost_usd
    run.wall_clock_s = ledger.elapsed_s
    return run
