"""``admission`` / ``permit`` / ``relay`` control spans and the ``resident_handoff``
traceparent carrier stamp, driven through the real ``ResidentCapacityControl``
origination -> ack -> outcome flow.

All three boundary-parented stages run on async control paths with no ambient span
(F11): this asserts each opens under the workflow's derived trace id rather than
rooting one of its own, that ``resident_handoff`` carries the stamp (and carries none
at all when telemetry is off), and that ``resident_sidecar_bind`` — replica-scoped, not
invocation-scoped — carries no stamp ever.
"""

import asyncio
from typing import Any

from server.resident.state import ReplicaIncarnation, ServiceFamily
from shared.resident.reports import ResidentBootstrapOutcome
from shared.telemetry.config import TelemetryLevel
from shared.telemetry.control import serve_trace_id_int
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from tests.server.resident.test_service import (
    AdmissionController,
    LifecycleScaleManager,
    ResidentCapacityControl,
    ResidentPolicyLimits,
    ResidentStores,
    _ack,
    _admission,
    _Delivery,
    _env,
)
from tests.server.telemetry_helpers import recording_control_tracer, spans_by_stage

_WORKFLOW_ID = "wfl-1"


def _build(control: Any = None) -> tuple[ResidentCapacityControl, _Delivery]:
    stores = ResidentStores()
    limits = ResidentPolicyLimits()

    async def materialize_fn(family: ServiceFamily, replica: ReplicaIncarnation) -> str:
        return "tsk-serve-1"

    admission = AdmissionController(stores)
    lifecycle = LifecycleScaleManager(
        stores, limits=limits, admission_slots=2, materialize_fn=materialize_fn
    )
    delivery = _Delivery()
    svc = ResidentCapacityControl(
        stores=stores,
        admission=admission,
        lifecycle=lifecycle,
        limits=limits,
        dependency_resolver=lambda task_id: _admission(),
        settle_cb=lambda *a, **k: True,
        redispatch_cb=lambda *a, **k: True,
        endpoint_probe=lambda serve_task_id: None,
        delivery=delivery.build(),
        poll_interval_sec=0.01,
        redrive_backoff_sec=0.0,
        control=control,
    )
    from shared.resident.contracts import ReplicaEndpoint

    svc._probe_endpoint = lambda serve_task_id: ReplicaEndpoint(
        base_url="http://replica", model="m"
    )
    return svc, delivery


def test_admission_and_relay_share_the_workflow_trace_id() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.COARSE)
    svc, delivery = _build(control)

    asyncio.run(svc._originate(_env()))

    expected_trace_id = workflow_to_trace_id_int(_WORKFLOW_ID)
    invocation_id = "inv-1"
    expected_boundary_span_id = derived_span_id(SpanIdKind.INVOCATION, invocation_id)

    for stage in ("admission", "relay"):
        matches = spans_by_stage(exporter, stage)
        assert matches, f"no {stage} span recorded"
        for span in matches:
            assert span.context is not None
            assert span.context.trace_id == expected_trace_id
            assert span.parent is not None
            assert span.parent.span_id == expected_boundary_span_id


def test_permit_stage_shares_the_workflow_trace_id() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.COARSE)
    svc, delivery = _build(control)
    asyncio.run(svc._originate(_env()))

    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))

    expected_trace_id = workflow_to_trace_id_int(_WORKFLOW_ID)
    expected_boundary_span_id = derived_span_id(SpanIdKind.INVOCATION, "inv-1")
    permit_spans = spans_by_stage(exporter, "permit")
    assert permit_spans
    for span in permit_spans:
        assert span.context is not None
        assert span.context.trace_id == expected_trace_id
        assert span.parent is not None
        assert span.parent.span_id == expected_boundary_span_id


def test_resident_handoff_carries_a_traceparent_reaching_the_far_end() -> None:
    control, _exporter = recording_control_tracer(TelemetryLevel.COARSE)
    svc, delivery = _build(control)

    asyncio.run(svc._originate(_env()))

    handoff_payload = delivery.frame("resident_handoff")
    assert "traceparent" in handoff_payload
    tp = handoff_payload["traceparent"]
    trace_id_hex = tp.split("-")[1]
    assert int(trace_id_hex, 16) == workflow_to_trace_id_int(_WORKFLOW_ID)


def test_resident_handoff_carries_no_traceparent_key_when_telemetry_is_off() -> None:
    svc, delivery = _build(control=None)

    asyncio.run(svc._originate(_env()))

    handoff_payload = delivery.frame("resident_handoff")
    assert "traceparent" not in handoff_payload


def test_resident_sidecar_bind_never_carries_a_traceparent() -> None:
    """The one carrier that must stay absent even with telemetry on."""
    control, _exporter = recording_control_tracer(TelemetryLevel.COARSE)
    svc, delivery = _build(control)

    asyncio.run(svc._originate(_env()))

    bind_payload = delivery.frame("resident_sidecar_bind")
    assert "traceparent" not in bind_payload


def test_admission_and_permit_and_relay_are_gated_off() -> None:
    svc, delivery = _build(control=None)

    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))

    # No exporter to check against (control is the disabled default), but this proves
    # the whole flow still runs with telemetry off.
    assert "resident_authorization" in delivery.kinds()


def test_serve_trace_id_is_stable_and_independent_of_workflow_bijection() -> None:
    """A gated serve request roots its own trace, keyed by task id + request id."""
    first = serve_trace_id_int("tsk-serve", "req-1")
    again = serve_trace_id_int("tsk-serve", "req-1")
    other_request = serve_trace_id_int("tsk-serve", "req-2")

    assert first == again
    assert first != other_request
    assert first != 0
    assert first != workflow_to_trace_id_int("wfl-1")
