#!/usr/bin/env python
"""Run the agent eval suites, write the report, and enforce the gate.

Exits non-zero when a suite fails its thresholds, so CI blocks a change that
regresses task success, escalation recall, or — the one that matters most —
starts booking dispositions on cases that need a human.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reconagent.config import settings
from reconagent.evaluation import gate, render_json, render_markdown, run_suite
from reconagent.evaluation.harness import DEFAULT_GATE_THRESHOLDS

# Only the primary suite gates the build. The naive baseline is expected to
# fail — it is the floor that proves the eval discriminates — and the chaos
# suite is reported so a failure-handling regression is visible without
# making CI hostage to injected faults.
GATED_SUITE = ("rules", "clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="rules", help="Policy to gate on (default: rules).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tasks.")
    parser.add_argument("--seed", type=int, default=settings.world_seed)
    parser.add_argument(
        "--baselines",
        action="store_true",
        default=True,
        help="Also run the naive floor and the chaos profile (default: on).",
    )
    parser.add_argument("--no-baselines", dest="baselines", action="store_false")
    parser.add_argument("--out", type=Path, default=settings.eval_dir)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common = {"world_seed": args.seed, "limit": args.limit, "projection_model": settings.anthropic_model}

    print(f"Running agent eval (seed {args.seed}, policy {args.policy})…")
    suites = [run_suite(args.policy, profile="clean", **common)]
    if args.baselines:
        suites.append(run_suite("naive", profile="clean", **common))
        suites.append(run_suite(args.policy, profile="chaos", **common))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(render_markdown(suites, settings.anthropic_model))
    (args.out / "results.json").write_text(render_json(suites))

    for suite in suites:
        metrics = suite.metrics
        print(
            f"  {suite.policy:<12} {suite.profile:<6} "
            f"success {metrics['task_success']:.3f}  "
            f"esc-recall {metrics['escalation_recall']:.3f}  "
            f"steps {metrics['mean_steps']:.2f}  "
            f"proj ${metrics['projected_cost_usd_per_task']:.4f}/task"
        )

    gated = next(
        (s for s in suites if (s.policy, s.profile) == (args.policy, "clean")), suites[0]
    )
    passed, reasons = gate(gated, DEFAULT_GATE_THRESHOLDS)
    print(f"\nReport: {args.out / 'report.md'}")
    if passed:
        print(f"Eval gate PASSED for {gated.policy}/{gated.profile}.")
        return 0
    print(f"Eval gate FAILED for {gated.policy}/{gated.profile}:")
    for reason in reasons:
        print(f"  - {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
