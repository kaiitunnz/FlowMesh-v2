"""Trace/span id derivation: bijection, non-zero, and kind separation."""

import uuid

from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.utils.ids import new_workflow_id


def test_workflow_to_trace_id_int_is_a_bijection_on_the_uuid_hex() -> None:
    workflow_id = new_workflow_id()
    hex_body = workflow_id.removeprefix("wfl-").replace("-", "")

    assert workflow_to_trace_id_int(workflow_id) == int(hex_body, 16)


def test_workflow_to_trace_id_int_is_case_insensitive() -> None:
    workflow_id = f"wfl-{uuid.uuid4()}"

    assert workflow_to_trace_id_int(workflow_id.upper()) == workflow_to_trace_id_int(
        workflow_id.lower()
    )


def test_derived_span_id_is_non_zero_for_a_large_sample() -> None:
    for i in range(10_000):
        assert derived_span_id(SpanIdKind.WORK_ITEM, f"wki-sample-{i}") != 0


def test_derived_span_id_is_deterministic() -> None:
    durable_id = "wki-fixed"

    assert derived_span_id(SpanIdKind.WORK_ITEM, durable_id) == derived_span_id(
        SpanIdKind.WORK_ITEM, durable_id
    )


def test_derived_span_id_distinguishes_kind_for_the_same_durable_id() -> None:
    durable_id = "shared-durable-id"

    work_item_span_id = derived_span_id(SpanIdKind.WORK_ITEM, durable_id)
    attempt_span_id = derived_span_id(SpanIdKind.ATTEMPT, durable_id)

    assert work_item_span_id != attempt_span_id
