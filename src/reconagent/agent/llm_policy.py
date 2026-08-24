"""The Anthropic policy: a real model choosing the next tool call.

It plugs into the same hand-rolled loop as the offline policies, so
everything the loop provides — budgets, retries, repeat detection, the
safety-net escalation — applies unchanged. The policy's only job is to turn
the turn history into a request and the response into one tool call.

Requires an API key and is therefore never exercised in CI; `make agent-eval`
runs the offline policies. See docs/AGENT_SDK_TRADEOFFS.md for why the loop
was kept rather than replaced by the SDK's own.
"""

from __future__ import annotations

import time

from ..tools.registry import anthropic_tool_schemas
from .prompts import SYSTEM_PROMPT
from .state import AgentState, PolicyDecision

MAX_OUTPUT_TOKENS = 1024
_API_MAX_ATTEMPTS = 3


class AnthropicPolicy:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The anthropic policy needs the SDK: pip install -e '.[llm]'"
            ) from exc
        self.model = model
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self._tools = anthropic_tool_schemas()

    def next_action(self, state: AgentState) -> PolicyDecision:
        response = self._create(self._build_messages(state))
        text = " ".join(block.text for block in response.content if block.type == "text").strip()
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        usage = (response.usage.input_tokens, response.usage.output_tokens)

        if tool_use is None:
            # The model answered in prose instead of acting. That is a failure
            # to finish, not a disposition, so the loop's safety net takes it.
            return PolicyDecision(
                tool_name=None,
                reasoning=text or "Model returned no tool call.",
                input_tokens=usage[0],
                output_tokens=usage[1],
            )

        return PolicyDecision(
            tool_name=tool_use.name,
            arguments=dict(tool_use.input or {}),
            reasoning=text,
            input_tokens=usage[0],
            output_tokens=usage[1],
            tool_use_id=tool_use.id,
        )

    def _build_messages(self, state: AgentState) -> list[dict]:
        messages: list[dict] = [{"role": "user", "content": state.task_prompt}]
        for index, turn in enumerate(state.turns):
            tool_use_id = turn.tool_use_id or f"call_{index:03d}"
            assistant: list[dict] = []
            if turn.reasoning:
                assistant.append({"type": "text", "text": turn.reasoning})
            assistant.append(
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": turn.tool_name,
                    "input": turn.arguments,
                }
            )
            messages.append({"role": "assistant", "content": assistant})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": turn.outcome.to_model_text(),
                            "is_error": not turn.outcome.ok,
                        }
                    ],
                }
            )
        return messages

    def _create(self, messages: list[dict]):
        """Retry transport-level failures only; a bad request will stay bad."""
        from anthropic import APIStatusError

        delay = 1.0
        for attempt in range(1, _API_MAX_ATTEMPTS + 1):
            try:
                return self._client.messages.create(
                    model=self.model,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=self._tools,
                    messages=messages,
                )
            except APIStatusError as exc:
                retryable = exc.status_code in {408, 429, 500, 502, 503, 529}
                if not retryable or attempt == _API_MAX_ATTEMPTS:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")
