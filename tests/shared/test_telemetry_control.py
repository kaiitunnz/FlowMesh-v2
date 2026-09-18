"""``ControlPlaneTracer``: null-twin discipline, explicit parenting, and the
``ledger_snapshot`` full-only gate and ambient carve-out.
"""

from opentelemetry import trace as otel_trace

from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.telemetry.control import (
    NULL_CONTROL_TRACER,
    ControlPlaneTracer,
    format_traceparent,
    serve_trace_id_int,
)
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.telemetry.provider import build_tracer
from shared.telemetry.semconv import ControlPlaneStage, ControlPlaneWindow
from tests.server.telemetry_helpers import recording_control_tracer, spans_by_stage


def _config(level: TelemetryLevel) -> TelemetryConfig:
    return TelemetryConfig(
        level=level,
        traces_enabled=True,
        metrics_enabled=False,
        sample_ratio=1.0,
        otlp_endpoint=None,
    )


def test_off_reuses_one_context_manager_across_many_calls() -> None:
    tracer = ControlPlaneTracer(
        build_tracer(_config(TelemetryLevel.OFF), {}), _config(TelemetryLevel.OFF)
    )

    contexts = {
        tracer.workflow_stage(
            ControlPlaneStage.ENGINE_BUILD,
            ControlPlaneWindow.SUBMIT,
            "wfl-deadbeefdeadbeefdeadbeefdeadbeef",
        )
        for _ in range(1000)
    }

    assert len(contexts) == 1
    assert NULL_CONTROL_TRACER.enabled is False


def test_null_control_tracer_is_disabled() -> None:
    assert NULL_CONTROL_TRACER.enabled is False
    with NULL_CONTROL_TRACER.episode_stage(
        ControlPlaneStage.DISPATCH,
        ControlPlaneWindow.QUEUE,
        "wfl-deadbeefdeadbeefdeadbeefdeadbeef",
        "wki-y",
    ) as span:
        assert span.is_recording() is False


def test_ledger_snapshot_gated_to_full_even_when_other_stages_are_on() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.FINE)
    assert control.enabled is True

    with control.ledger_snapshot("wfl-deadbeefdeadbeefdeadbeefdeadbeef"):
        pass

    assert exporter.get_finished_spans() == ()


def test_ledger_snapshot_emits_at_full() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.FULL)

    with control.ledger_snapshot("wfl-deadbeefdeadbeefdeadbeefdeadbeef"):
        pass

    assert len(spans_by_stage(exporter, "ledger_snapshot")) == 1


def test_ledger_snapshot_nests_under_an_open_enclosing_stage() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.FULL)

    with control.workflow_stage(
        ControlPlaneStage.DS_INITIAL_ADVANCE,
        ControlPlaneWindow.SUBMIT,
        "wfl-deadbeefdeadbeefdeadbeefdeadbeef",
    ) as outer:
        with control.ledger_snapshot("wfl-deadbeefdeadbeefdeadbeefdeadbeef") as inner:
            pass

    outer_id = outer.get_span_context().span_id
    inner_span = spans_by_stage(exporter, "ledger_snapshot")[0]
    assert inner_span.parent is not None
    assert inner_span.parent.span_id == outer_id
    assert inner is not None


def test_ledger_snapshot_falls_back_to_workflow_parent_with_no_enclosing_stage() -> (
    None
):
    control, exporter = recording_control_tracer(TelemetryLevel.FULL)

    with control.ledger_snapshot("wfl-deadbeefdeadbeefdeadbeefdeadbeef"):
        pass

    span = spans_by_stage(exporter, "ledger_snapshot")[0]
    assert span.parent is not None
    assert span.parent.span_id == derived_span_id(
        SpanIdKind.WORKFLOW, "wfl-deadbeefdeadbeefdeadbeefdeadbeef"
    )
    assert span.context is not None
    assert span.context.trace_id == workflow_to_trace_id_int(
        "wfl-deadbeefdeadbeefdeadbeefdeadbeef"
    )


def test_ledger_snapshot_ignores_a_foreign_trace_ambient_span() -> None:
    """An inbound external ``traceparent`` must become a Link, never a parent.

    If some other producer's ambient context ever leaked in as the current span
    during a ``ledger_snapshot`` call, adopting it would silently carry the snapshot
    out of the workflow's own trace. It must fall back to the explicit workflow
    parent instead, exactly as the no-ambient-span case does.
    """
    control, exporter = recording_control_tracer(TelemetryLevel.FULL)
    workflow_id = "wfl-deadbeefdeadbeefdeadbeefdeadbeef"
    foreign_workflow_id = "wfl-cafebabecafebabecafebabecafebabe"

    with control.workflow_stage(
        ControlPlaneStage.ENGINE_BUILD, ControlPlaneWindow.SUBMIT, foreign_workflow_id
    ):
        with control.ledger_snapshot(workflow_id):
            pass

    span = spans_by_stage(exporter, "ledger_snapshot")[0]
    assert span.context is not None
    assert span.context.trace_id == workflow_to_trace_id_int(workflow_id)
    assert span.parent is not None
    assert span.parent.span_id == derived_span_id(SpanIdKind.WORKFLOW, workflow_id)


def test_boundary_stage_uses_the_supplied_trace_id_not_a_workflow_derivation() -> None:
    """A boundary may belong to a serve invocation, which owns no workflow_id.

    Driven at ``fine`` because that is where a boundary stage emits: it parents on the
    invocation span, which is synthesized at that level and no lower.
    """
    control, exporter = recording_control_tracer(TelemetryLevel.FINE)
    serve_trace_id = serve_trace_id_int("tsk-serve", "req-1")

    with control.boundary_stage(
        ControlPlaneStage.ADMISSION,
        ControlPlaneWindow.POST_START,
        serve_trace_id,
        "inv-1",
    ):
        pass

    span = spans_by_stage(exporter, "admission")[0]
    assert span.context is not None
    assert span.context.trace_id == serve_trace_id
    assert span.parent is not None
    assert span.parent.span_id == derived_span_id(SpanIdKind.INVOCATION, "inv-1")


def test_format_traceparent_round_trips_through_the_w3c_propagator() -> None:
    trace_id = workflow_to_trace_id_int("wfl-deadbeefdeadbeefdeadbeefdeadbeef")
    span_id = derived_span_id(SpanIdKind.INVOCATION, "inv-1")

    tp = format_traceparent(trace_id, span_id)

    carrier = {"traceparent": tp}
    ctx = otel_trace.propagation.tracecontext.TraceContextTextMapPropagator().extract(
        carrier
    )
    extracted = otel_trace.get_current_span(ctx).get_span_context()
    assert extracted.trace_id == trace_id
    assert extracted.span_id == span_id


def test_serve_trace_id_uses_a_different_input_domain_than_the_workflow_bijection() -> (
    None
):
    # Not a proof of non-collision (both are 128-bit hashes) — asserts the two use
    # visibly different input domains, which is what the design relies on.
    assert serve_trace_id_int(
        "wfl-deadbeefdeadbeefdeadbeefdeadbeef", ""
    ) != workflow_to_trace_id_int("wfl-deadbeefdeadbeefdeadbeefdeadbeef")
