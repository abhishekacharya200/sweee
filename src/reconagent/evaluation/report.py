"""Rendering eval results.

The markdown report is written for the person deciding whether to ship the
change, so it leads with the comparison across policies and profiles, then
the per-archetype breakdown that says *which* cases moved, then a couple of
real failing transcripts. Raw per-task rows go to JSON.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from ..world import ARCHETYPES
from .harness import PROFILES, SuiteResult, cost_basis

_HEADLINE = [
    ("task_success", "Task success", ".3f"),
    ("escalation_recall", "Escalation recall", ".3f"),
    ("escalation_precision", "Escalation precision", ".3f"),
    ("missed_escalation_rate", "Missed escalations", ".3f"),
    ("wrong_linkage_rate", "Wrong linkage", ".3f"),
    ("unhandled_rate", "Unhandled", ".3f"),
    ("forced_escalation_rate", "Safety-net fired", ".3f"),
    ("mean_steps", "Steps/task", ".2f"),
    ("mean_retries", "Retries/task", ".2f"),
    ("mean_latency_s", "Latency/task (s)", ".4f"),
    ("cost_usd_per_task", "Cost/task ($)", ".4f"),
    ("projected_cost_usd_per_task", "Projected $/task", ".4f"),
]


def _fmt(value: object, spec: str = ".3f") -> str:
    return format(float(value or 0), spec)


def render_markdown(suites: list[SuiteResult], projection_model: str = "claude-sonnet-5") -> str:
    if not suites:
        return "# Agent evaluation\n\nNo suites were run.\n"

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    columns = [f"{s.policy} / {s.profile}" for s in suites]
    lines = [
        "# Reconciliation agent evaluation",
        "",
        (
            f"Generated {stamp} · world seed `{suites[0].world_seed}` · "
            f"{suites[0].metrics.get('tasks', 0)} tasks per suite"
        ),
        "",
        (
            "Projected cost prices the same token volume at "
            f"`{projection_model}` rates; offline policies bill $0."
        ),
        "",
        "## Headline",
        "",
        "| Metric | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for key, label, spec in _HEADLINE:
        row = [_fmt(s.metrics.get(key, 0), spec) for s in suites]
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    basis = cost_basis(suites[0], projection_model)
    if basis:
        lines += [
            "",
            "## Cost basis",
            "",
            (
                f"Token counter: `{basis['tokenizer']}`. tiktoken downloads its encoding on "
                "first use, so a sandboxed box falls back to a ~4-chars-per-token estimate and "
                "every figure below shifts a few percent. Quote these numbers with the counter "
                "named."
            ),
            "",
            "| | |",
            "|---|---|",
            f"| Fixed prefix (system prompt + tool schemas) | {basis['fixed_prefix_tokens']} tokens |",
            f"| Mean input tokens per task | {basis['mean_input_tokens']} |",
            (
                f"| Of which re-sent prefix | {basis['resent_prefix_tokens']} "
                f"({basis['resent_prefix_share']:.1%}) |"
            ),
            (
                f"| Projected $/task on `{basis['projection_model']}` | "
                f"${basis['projected_cost_usd_per_task']:.4f} |"
            ),
            (
                "| With prompt caching on the prefix | "
                f"${basis['projected_cost_usd_per_task_cached']:.4f} "
                f"(-{basis['cache_saving']:.0%}) |"
            ),
        ]

    lines += ["", "## Profiles", ""]
    for name in dict.fromkeys(s.profile for s in suites):
        lines.append(f"- **{name}** — {PROFILES[name].description}")

    lines += ["", "## Success by archetype", "", "| Archetype | " + " | ".join(columns) + " |",
              "|---|" + "---|" * len(columns)]
    per_suite = [s.by_archetype() for s in suites]
    for archetype in ARCHETYPES:
        cells = []
        for bucket in per_suite:
            entry = bucket.get(archetype)
            cells.append(f"{entry['success']:.2f} ({entry['tasks']})" if entry else "—")
        lines.append(f"| `{archetype}` | " + " | ".join(cells) + " |")

    lines += ["", "## Failure modes", "", "| Mode | " + " | ".join(columns) + " |",
              "|---|" + "---|" * len(columns)]
    all_modes = sorted({m for s in suites for m in s.metrics.get("failure_modes", {})})
    if all_modes:
        for mode in all_modes:
            row = [str(s.metrics.get("failure_modes", {}).get(mode, 0)) for s in suites]
            lines.append(f"| `{mode}` | " + " | ".join(row) + " |")
    else:
        lines.append("| _none_ | " + " | ".join("0" for _ in suites) + " |")

    lines += ["", "## Guards fired", ""]
    for suite, column in zip(suites, columns):
        histogram = suite.guard_histogram()
        detail = ", ".join(f"`{k}` ×{v}" for k, v in histogram.items()) if histogram else "none"
        lines.append(f"- **{column}** — {detail}")

    for suite, column in zip(suites, columns):
        failures = suite.failures()
        if not failures:
            continue
        lines += ["", f"## Sample failures — {column}", ""]
        for result in failures[:3]:
            lines += [
                (
                    f"**{result.task.task_id}** (`{result.task.archetype}`) — "
                    f"{result.verdict.failure_mode}: {result.verdict.detail}"
                ),
                "",
                "```",
                result.run.transcript(),
                "```",
                "",
            ]

    return "\n".join(lines).rstrip() + "\n"


def render_json(suites: list[SuiteResult]) -> str:
    return json.dumps(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "cost_basis": cost_basis(suites[0]) if suites else {},
            "suites": [suite.to_dict() for suite in suites],
        },
        indent=2,
    )
