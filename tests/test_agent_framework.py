"""The framework agent is built from the registry, not from a second declaration.

No model is called here: what is worth testing offline is that the projection
into pydantic-ai stays in step with `TOOL_SPECS`, which is exactly the thing
that silently rots when a tool is added.
"""

from __future__ import annotations

import pytest

from reconagent.tools.registry import TOOL_SPECS, input_json_schema

pytest.importorskip("pydantic_ai")

from reconagent.agent.framework import _Recorder, _TerminalReached, build_agent


@pytest.fixture
def agent(registry):
    return build_agent(registry, _Recorder(), defer_model_check=True)


def test_the_framework_agent_registers_the_whole_registry(agent):
    assert list(agent.toolsets[0].tools) == [spec.name for spec in TOOL_SPECS]


def test_the_framework_gets_the_same_schemas_as_the_messages_api(agent):
    for spec in TOOL_SPECS:
        tool = agent.toolsets[0].tools[spec.name]
        assert tool.description == spec.description
        assert tool.function_schema.json_schema == input_json_schema(spec)


def test_a_terminal_tool_unwinds_the_run(registry):
    """The framework stops on final output, so ending on a write means raising."""
    recorder = _Recorder()
    agent = build_agent(registry, recorder, defer_model_check=True)
    handler = agent.toolsets[0].tools["escalate_exception"].function_schema.function

    exception_id = next(iter(registry.store.exceptions))
    with pytest.raises(_TerminalReached) as raised:
        handler(exception_id=exception_id, reason="Unwinding the framework run on purpose.")
    assert raised.value.tool_name == "escalate_exception"
    assert recorder.steps[-1].ok


def test_tool_failures_are_raised_as_model_retries(registry):
    from pydantic_ai import ModelRetry

    agent = build_agent(registry, _Recorder(), defer_model_check=True)
    handler = agent.toolsets[0].tools["get_invoice"].function_schema.function
    with pytest.raises(ModelRetry) as raised:
        handler(invoice_id="INV-1900-0001")
    assert "INV-1900-0001" in str(raised.value)
