"""Command line entry point: inspect the queue, watch one episode, dump schemas.

`reconagent run EXC-0007` printing a full step trace is the fastest way to
find out why an agent decided what it decided, which is why it exists.
"""

from __future__ import annotations

import argparse
import json

from .agent.loop import run_episode
from .agent.policies import build_policy
from .budget import Budget
from .config import settings
from .evaluation import gate, render_markdown, run_suite
from .evaluation.harness import DEFAULT_GATE_THRESHOLDS
from .mcp_server import tool_surface_summary
from .models import ExceptionStatus
from .tools.faults import FaultInjector
from .tools.registry import anthropic_tool_schemas, build_registry, tool_table_markdown
from .world import build_world


def _queue(args: argparse.Namespace) -> int:
    store, _ = build_world(args.seed)
    rows = store.list_exceptions(status=ExceptionStatus.OPEN, limit=args.limit)
    print(f"{len(rows)} open exception(s) of {len(store.exceptions)}\n")
    for exc in rows:
        target = exc.invoice_id or "-"
        delta = f"{exc.delta_cents:+d}c" if exc.delta_cents is not None else "-"
        print(f"  {exc.exception_id}  {exc.kind.value:<18} inv={target:<14} txn={exc.txn_id or '-':<12} Δ={delta}")
        print(f"      {exc.note}")
    return 0


def _run(args: argparse.Namespace) -> int:
    store, _ = build_world(args.seed)
    faults = FaultInjector(rate=args.fault_rate, seed=args.seed) if args.fault_rate else None
    registry = build_registry(store.clone(), faults=faults)
    policy = build_policy(args.policy, model=args.model, api_key=settings.anthropic_api_key)
    run = run_episode(
        args.exception_id,
        registry,
        policy,
        Budget(
            max_steps=settings.max_steps,
            max_tool_calls=settings.max_tool_calls,
            max_tool_errors=settings.max_tool_errors,
            max_cost_usd=settings.max_cost_usd,
            max_wall_clock_s=settings.max_wall_clock_s,
        ),
        projection_model=settings.anthropic_model,
    )
    print(run.transcript())
    print(
        f"\n  tool calls {run.tool_calls} (errors {run.tool_errors}, retries {run.retries})"
        f" · tokens {run.input_tokens}/{run.output_tokens}"
        f" · ${run.cost_usd:.4f} (projected ${run.projected_cost_usd:.4f} on {settings.anthropic_model})"
        f" · {run.wall_clock_s:.3f}s"
    )
    return 0


def _tools(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(tool_surface_summary() if args.mcp else anthropic_tool_schemas(), indent=2))
    else:
        print(tool_table_markdown())
    return 0


def _eval(args: argparse.Namespace) -> int:
    suite = run_suite(
        args.policy,
        profile=args.profile,
        world_seed=args.seed,
        limit=args.limit,
        model=args.model,
        api_key=settings.anthropic_api_key,
        runner=args.runner,
        projection_model=settings.anthropic_model,
    )
    print(render_markdown([suite], settings.anthropic_model))
    passed, reasons = gate(suite, DEFAULT_GATE_THRESHOLDS)
    if not passed:
        print("Gate FAILED: " + "; ".join(reasons))
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reconagent", description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.world_seed)
    subparsers = parser.add_subparsers(dest="command", required=True)

    queue = subparsers.add_parser("queue", help="List open reconciliation exceptions.")
    queue.add_argument("--limit", type=int, default=20)
    queue.set_defaults(func=_queue)

    run = subparsers.add_parser("run", help="Work one exception and print the step trace.")
    run.add_argument("exception_id")
    run.add_argument("--policy", default=settings.resolve_policy())
    run.add_argument("--model", default=settings.anthropic_model)
    run.add_argument("--fault-rate", type=float, default=0.0)
    run.set_defaults(func=_run)

    tools = subparsers.add_parser("tools", help="Show the tool surface.")
    tools.add_argument("--json", action="store_true", help="Emit JSON schemas instead of a table.")
    tools.add_argument("--mcp", action="store_true", help="Emit the MCP surface (tools, resources, prompts).")
    tools.set_defaults(func=_tools)

    evaluate = subparsers.add_parser("eval", help="Run one eval suite and print the report.")
    evaluate.add_argument("--policy", default="rules")
    evaluate.add_argument("--profile", default="clean")
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--model", default=settings.anthropic_model)
    evaluate.add_argument("--runner", default="loop", choices=["loop", "framework"])
    evaluate.set_defaults(func=_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
