"""MCP server over the same tool registry.

This is the third consumer of `TOOL_SPECS`, after the hand-rolled loop and the
pydantic-ai agent, and it is the one that pays off the "define the tools once"
decision: exposing the reconciliation surface to Claude Desktop, Claude Code or
any other MCP client costs the file you are reading and nothing else. No tool
is redeclared, and `inputSchema` here is byte-identical to the `input_schema`
sent to the Messages API.

Argument validation is deliberately left to the registry (`validate_input=False`)
rather than to the MCP SDK's JSON Schema check. Both would reject the same
calls, but only the registry returns the repair instruction — which field, why,
and what the valid fields are — and an MCP client deserves the same quality of
error as the in-process loop.

Run it with `python scripts/mcp_server.py` (stdio), or wire it into a client:

    {
      "mcpServers": {
        "reconagent": {
          "command": "python",
          "args": ["scripts/mcp_server.py"],
          "cwd": "/path/to/this/repo"
        }
      }
    }
"""

from __future__ import annotations

import json
from typing import Any

from .agent.prompts import SYSTEM_PROMPT, build_task_prompt
from .models import ExceptionStatus
from .store import LedgerStore
from .tools.registry import TOOL_SPECS, ToolRegistry, mcp_tool_definitions
from .world import ACCOUNTING_POLICY, build_world

SERVER_NAME = "reconagent"

RESOURCES: dict[str, tuple[str, str]] = {
    "recon://queue/open": ("Open exception queue", "Exceptions still awaiting a disposition."),
    "recon://policy/accounting": (
        "Accounting thresholds",
        "Entity-wide FX tolerance, fee tolerance, write-off limit and duplicate lookback window.",
    ),
    "recon://ledger/summary": (
        "Ledger summary",
        "Row counts and settlement status across the AR ledger and the bank feed.",
    ),
}


def _resource_body(store: LedgerStore, uri: str) -> str:
    if uri == "recon://queue/open":
        rows = store.list_exceptions(status=ExceptionStatus.OPEN, limit=1000)
        return json.dumps([row.model_dump(mode="json") for row in rows], indent=2)
    if uri == "recon://policy/accounting":
        return ACCOUNTING_POLICY.model_dump_json(indent=2)
    if uri == "recon://ledger/summary":
        return json.dumps(
            {
                "invoices": len(store.invoices),
                "bank_transactions": len(store.transactions),
                "customers": len(store.policies),
                "exceptions_open": len(store.list_exceptions(status=ExceptionStatus.OPEN, limit=10_000)),
                "exceptions_resolved": len(store.resolutions),
                "exceptions_escalated": len(store.escalations),
            },
            indent=2,
        )
    raise ValueError(f"Unknown resource {uri!r}.")


def build_server(store: LedgerStore) -> Any:
    """Wire the registry, resources and one prompt onto a low-level MCP server."""
    from mcp import types
    from mcp.server.lowlevel import Server
    from pydantic import AnyUrl

    registry = ToolRegistry(store)
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=definition["name"],
                description=definition["description"],
                inputSchema=definition["inputSchema"],
                outputSchema=definition["outputSchema"],
                annotations=types.ToolAnnotations(**definition["annotations"]),
            )
            for definition in mcp_tool_definitions()
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        outcome = registry.call(name, arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=outcome.to_model_text())],
            structuredContent=outcome.payload if outcome.ok else None,
            isError=not outcome.ok,
        )

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl(uri), name=name, description=description, mimeType="application/json"
            )
            for uri, (name, description) in RESOURCES.items()
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> str:
        return _resource_body(store, str(uri))

    @server.list_prompts()
    async def list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="triage_exception",
                description="Work one reconciliation exception to a disposition or an escalation.",
                arguments=[
                    types.PromptArgument(
                        name="exception_id",
                        description="Queue id, e.g. 'EXC-0007'.",
                        required=True,
                    )
                ],
            )
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        if name != "triage_exception":
            raise ValueError(f"Unknown prompt {name!r}.")
        exception_id = (arguments or {}).get("exception_id", "")
        if not exception_id:
            raise ValueError("triage_exception requires an exception_id argument.")
        return types.GetPromptResult(
            description=f"Triage {exception_id}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text", text=f"{SYSTEM_PROMPT}\n\n{build_task_prompt(exception_id)}"
                    ),
                )
            ],
        )

    return server


async def serve_stdio(seed: int = 20260301) -> None:
    from mcp.server.stdio import stdio_server

    store, _ = build_world(seed)
    server = build_server(store)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    import anyio

    anyio.run(serve_stdio)


def tool_surface_summary() -> dict:
    """Everything a client would discover, without starting a server.

    Used by the tests and by `reconagent tools --json`, so the exported schemas
    can be diffed in CI rather than eyeballed in a client.
    """
    return {
        "server": SERVER_NAME,
        "tools": mcp_tool_definitions(),
        "resources": [{"uri": uri, "name": name} for uri, (name, _) in RESOURCES.items()],
        "prompts": ["triage_exception"],
        "terminal_tools": [spec.name for spec in TOOL_SPECS if spec.terminal],
    }


if __name__ == "__main__":
    main()
