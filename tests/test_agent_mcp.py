"""The MCP server must expose exactly the registry, over a real session."""

from __future__ import annotations

import json

import pytest

from reconagent.mcp_server import RESOURCES, build_server, tool_surface_summary
from reconagent.tools.registry import TOOL_SPECS

pytest.importorskip("mcp")


@pytest.fixture
def session_factory(ledger):
    from mcp.shared.memory import create_connected_server_and_client_session

    def factory():
        return create_connected_server_and_client_session(build_server(ledger))

    return factory


def _run(coro):
    import anyio

    return anyio.run(lambda: coro)


def test_surface_summary_matches_the_registry():
    summary = tool_surface_summary()
    assert [t["name"] for t in summary["tools"]] == [spec.name for spec in TOOL_SPECS]
    assert summary["terminal_tools"] == ["record_resolution", "escalate_exception"]
    assert {resource["uri"] for resource in summary["resources"]} == set(RESOURCES)


def test_a_client_discovers_every_tool(session_factory):
    async def scenario():
        async with session_factory() as session:
            await session.initialize()
            listing = await session.list_tools()
            return [tool.name for tool in listing.tools]

    assert _run(scenario()) == [spec.name for spec in TOOL_SPECS]


def test_a_client_can_call_a_tool_and_read_structured_content(session_factory):
    async def scenario():
        async with session_factory() as session:
            await session.initialize()
            return await session.call_tool("get_accounting_policy", {})

    result = _run(scenario())
    assert result.isError is False
    assert result.structuredContent["policy"]["fx_variance_tolerance_bps"] == 75


def test_tool_errors_reach_the_client_with_the_repair_instruction(session_factory):
    async def scenario():
        async with session_factory() as session:
            await session.initialize()
            return await session.call_tool("get_invoice", {})

    result = _run(scenario())
    assert result.isError is True
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "invalid_arguments"
    assert "invoice_id" in payload["error"]


def test_resources_are_readable(session_factory):
    async def scenario():
        async with session_factory() as session:
            await session.initialize()
            listing = await session.list_resources()
            body = await session.read_resource(listing.resources[0].uri)
            return [str(r.uri) for r in listing.resources], body.contents[0].text

    uris, body = _run(scenario())
    assert set(uris) == set(RESOURCES)
    assert isinstance(json.loads(body), list)


def test_the_triage_prompt_is_served_with_its_argument(session_factory):
    async def scenario():
        async with session_factory() as session:
            await session.initialize()
            listing = await session.list_prompts()
            got = await session.get_prompt("triage_exception", {"exception_id": "EXC-0001"})
            return [p.name for p in listing.prompts], got.messages[0].content.text

    names, text = _run(scenario())
    assert names == ["triage_exception"]
    assert "EXC-0001" in text


def test_writes_through_mcp_land_in_the_ledger(session_factory, ledger):
    from reconagent.models import ExceptionStatus

    exception_id = ledger.list_exceptions(status=ExceptionStatus.OPEN, limit=1)[0].exception_id

    async def scenario():
        async with session_factory() as session:
            await session.initialize()
            return await session.call_tool(
                "escalate_exception",
                {"exception_id": exception_id, "reason": "Escalated over MCP by the test suite."},
            )

    result = _run(scenario())
    assert result.isError is False
    assert ledger.get_exception(exception_id).status is ExceptionStatus.ESCALATED
