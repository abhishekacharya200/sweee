"""Scoring one episode against its known-correct disposition.

Task success is all-or-nothing on purpose. A reconciliation booked against
the right invoice with the wrong adjustment is not 80% correct — it is a
wrong journal entry, and an eval that awards partial credit for it will
happily let that regression through the gate.

The failure taxonomy matters more than the headline number. `missed_escalation`
(booking a disposition that needed a human) and `wrong_linkage` (touching the
wrong invoice) are the expensive ones; `over_escalation` costs an operator ten
minutes. They are counted separately so the gate can be strict about the first
kind and tolerant of the second.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.trace import STOP_ABANDONED, AgentRun
from ..world import GoldenTask

FAILURE_MODES = (
    "wrong_disposition",
    "wrong_linkage",
    "wrong_adjustment",
    "missed_escalation",
    "over_escalation",
    "unhandled",
)


@dataclass
class Verdict:
    success: bool
    failure_mode: str | None
    actual_disposition: str
    detail: str = ""


def _actual_disposition(run: AgentRun) -> str:
    if run.stop_reason == STOP_ABANDONED or run.terminal_tool is None:
        return "none"
    if run.terminal_tool == "escalate_exception":
        return "escalate"
    return str(run.terminal_arguments.get("resolution_type", "unknown"))


def score_run(task: GoldenTask, run: AgentRun) -> Verdict:
    actual = _actual_disposition(run)

    if actual == "none":
        return Verdict(False, "unhandled", actual, "Episode ended without a terminal write.")

    if task.expected_escalation:
        if actual == "escalate":
            return Verdict(True, None, actual)
        return Verdict(
            False,
            "missed_escalation",
            actual,
            f"Booked {actual} on a case that needed a human decision.",
        )

    if actual == "escalate":
        return Verdict(
            False,
            "over_escalation",
            actual,
            f"Escalated a case that should have been booked as {task.expected_resolution_type.value}.",
        )

    args = run.terminal_arguments
    if actual != task.expected_resolution_type.value:
        return Verdict(
            False,
            "wrong_disposition",
            actual,
            f"Booked {actual}, expected {task.expected_resolution_type.value}.",
        )
    if args.get("invoice_id") != task.expected_invoice_id or args.get("txn_id") != task.expected_txn_id:
        return Verdict(
            False,
            "wrong_linkage",
            actual,
            f"Linked {args.get('invoice_id')}/{args.get('txn_id')}, "
            f"expected {task.expected_invoice_id}/{task.expected_txn_id}.",
        )
    adjustment = int(args.get("adjustment_cents", 0))
    if abs(adjustment - task.expected_adjustment_cents) > task.adjustment_tolerance_cents:
        return Verdict(
            False,
            "wrong_adjustment",
            actual,
            f"Posted {adjustment} cents, expected {task.expected_adjustment_cents} "
            f"(±{task.adjustment_tolerance_cents}).",
        )
    return Verdict(True, None, actual)
