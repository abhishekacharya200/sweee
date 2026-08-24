from __future__ import annotations

import json

import pytest

from reconagent.models import ExceptionStatus
from reconagent.tools.faults import FaultInjector, ToolError
from reconagent.tools.registry import (
    TERMINAL_TOOLS,
    TOOL_SPECS,
    anthropic_tool_schemas,
    inline_defs,
    mcp_tool_definitions,
)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


@pytest.mark.parametrize("schema", anthropic_tool_schemas(), ids=lambda s: s["name"])
def test_anthropic_schemas_carry_no_unresolved_refs(schema):
    for node in _walk(schema["input_schema"]):
        assert "$ref" not in node
        assert "$defs" not in node


def test_schema_dialects_describe_the_same_surface():
    anthropic = {s["name"]: s["input_schema"] for s in anthropic_tool_schemas()}
    mcp = {d["name"]: d["inputSchema"] for d in mcp_tool_definitions()}
    assert anthropic.keys() == mcp.keys() == {spec.name for spec in TOOL_SPECS}
    assert anthropic == mcp, "the two dialects must project from one schema, not two"


def test_enum_values_survive_inlining():
    schema = next(s for s in anthropic_tool_schemas() if s["name"] == "record_resolution")
    assert schema["input_schema"]["properties"]["resolution_type"]["enum"] == [
        "match_payment",
        "bank_fee",
        "fx_variance",
        "short_payment",
        "duplicate_payment",
    ]


def test_mcp_marks_only_the_writing_tools_as_mutating():
    read_only = {
        d["name"] for d in mcp_tool_definitions() if d["annotations"]["readOnlyHint"]
    }
    assert read_only == {spec.name for spec in TOOL_SPECS} - TERMINAL_TOOLS


def test_inline_defs_survives_a_self_referencing_schema():
    cyclic = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
    }
    flattened = inline_defs(cyclic)
    assert "$defs" not in flattened
    assert list(_walk(flattened))


def test_validation_error_names_the_field_and_the_alternatives(registry):
    outcome = registry.call("get_invoice", {})
    assert not outcome.ok
    assert outcome.error_kind == "invalid_arguments"
    assert "invoice_id" in outcome.error
    assert not outcome.retryable


def test_unknown_tool_lists_the_real_ones(registry):
    outcome = registry.call("reconcile_everything", {})
    assert outcome.error_kind == "unknown_tool"
    assert "get_exception" in outcome.error


def test_domain_errors_are_not_retryable(registry):
    outcome = registry.call("get_invoice", {"invoice_id": "INV-1900-0001"})
    assert outcome.error_kind == "domain_error"
    assert not outcome.retryable


def test_tool_result_text_is_json(registry):
    outcome = registry.call("get_accounting_policy", {})
    assert outcome.ok
    assert json.loads(outcome.to_model_text())["policy"]["fx_variance_tolerance_bps"] == 75


def test_search_reports_truncation(registry):
    outcome = registry.call("search_invoices", {"limit": 2})
    assert outcome.ok
    assert len(outcome.payload["invoices"]) == 2
    assert outcome.payload["truncated"] is True


def test_reference_extraction_can_be_restricted_to_a_shortlist(registry, ledger):
    invoice_id = next(iter(ledger.invoices))
    outcome = registry.call(
        "extract_invoice_reference",
        {"memo": f"ACH CREDIT REF {invoice_id.replace('-', '')}", "candidate_invoice_ids": [invoice_id]},
    )
    assert [c["invoice_id"] for c in outcome.payload["candidates"]] == [invoice_id]


def test_terminal_write_closes_the_exception(registry, ledger):
    exception = ledger.list_exceptions(status=ExceptionStatus.OPEN, limit=1)[0]
    outcome = registry.call(
        "escalate_exception",
        {"exception_id": exception.exception_id, "reason": "Testing the write path end to end."},
    )
    assert outcome.ok and outcome.terminal
    assert ledger.get_exception(exception.exception_id).status is ExceptionStatus.ESCALATED

    again = registry.call(
        "escalate_exception",
        {"exception_id": exception.exception_id, "reason": "Trying to close it a second time."},
    )
    assert not again.ok, "an exception must not be closed twice"


def test_fault_injector_spares_the_terminal_writes():
    injector = FaultInjector(rate=1.0, seed=7)
    injector.maybe_fail("record_resolution")
    injector.maybe_fail("escalate_exception")
    with pytest.raises(ToolError):
        injector.maybe_fail("search_invoices")


def test_fault_injection_is_reproducible():
    def sequence(seed: int) -> list[bool]:
        injector = FaultInjector(rate=0.5, seed=seed)
        results = []
        for _ in range(40):
            try:
                injector.maybe_fail("search_invoices")
                results.append(True)
            except ToolError:
                results.append(False)
        return results

    assert sequence(11) == sequence(11)
    assert sequence(11) != sequence(12)
