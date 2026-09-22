"""The control-plane instrumentation seam: F11 (every span shares the workflow's
trace id) plus the stage-count, level-gating and v1/v2 gating rules.

Six of the thirteen control stages run off the request thread (the dispatcher loop,
worker-event handling, async resident control paths) where there is no ambient span.
This drives real ``register`` + ``dispatch_once`` + ``mark_succeeded`` calls against a
tracer wired to an in-memory exporter and asserts every recorded span's trace id equals
the workflow's derived trace id — a span opened with no explicit parent there would
silently root a random trace instead, which a submit-only test would never catch.
"""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

import pytest

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.telemetry.config import TelemetryLevel
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import FakeRegistry, _NoopSecretVault
from tests.server.telemetry_helpers import recording_control_tracer, spans_by_stage

_ECHO_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: seam}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""

_ECHO_V1 = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: seam-v1
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
"""


def _runtime(*, control: Any = None, episode_lowering: bool = False) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(episode_lowering=episode_lowering),
        make_result_reader(),
        logging.getLogger("control-plane-seam-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
        control=control,
    )


def _register_v2(runtime: TaskRuntime, payload: str = _ECHO_V2) -> tuple[str, str]:
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", payload, format="native")
    )
    return workflow_id, results[0].task_id


def _dispatch(runtime: TaskRuntime, task_id: str, *, control: Any = None) -> None:
    worker = Worker(
        id="wkr-1",
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1
    disp = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("control-plane-seam-dispatch"),
        control=control,
    )
    disp.dispatch_once(task_id)


@pytest.mark.parametrize(
    ("stage",),
    [
        ("compile_lower",),
        ("compile_assemble",),
        ("compile_finalize",),
        ("compile_validate",),
        ("engine_build",),
        ("ds_initial_advance",),
    ],
)
def test_submit_stages_share_the_workflow_trace_id(stage: str) -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.COARSE)
    runtime = _runtime(control=control)

    workflow_id, _task_id = _register_v2(runtime)

    matches = spans_by_stage(exporter, stage)
    assert len(matches) == 1
    span = matches[0]
    assert span.context is not None
    assert span.context.trace_id == workflow_to_trace_id_int(workflow_id)
    assert span.parent is not None
    assert span.parent.span_id == derived_span_id(SpanIdKind.WORKFLOW, workflow_id)


def test_compile_stages_are_four_under_transparent_lowering() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.COARSE)
    runtime = _runtime(control=control, episode_lowering=False)

    _register_v2(runtime)

    compile_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name.startswith("flowmesh.control.compile_")
    ]
    assert len(compile_spans) == 4
    assert not spans_by_stage(exporter, "compile_episodes")


def test_compile_stages_are_five_under_episode_cut_lowering() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.COARSE)
    runtime = _runtime(control=control, episode_lowering=True)

    _register_v2(runtime)

    compile_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name.startswith("flowmesh.control.compile_")
    ]
    assert len(compile_spans) == 5
    assert len(spans_by_stage(exporter, "compile_episodes")) == 1


def test_ledger_snapshot_absent_below_full() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.FINE)
    runtime = _runtime(control=control)

    _register_v2(runtime)

    assert not spans_by_stage(exporter, "ledger_snapshot")


def test_ledger_snapshot_present_at_full_and_shares_the_workflow_trace_id() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.FULL)
    runtime = _runtime(control=control)

    workflow_id, _task_id = _register_v2(runtime)

    snapshots = spans_by_stage(exporter, "ledger_snapshot")
    assert snapshots
    for span in snapshots:
        assert span.context is not None
        assert span.context.trace_id == workflow_to_trace_id_int(workflow_id)


def test_dispatch_and_ds_drive_are_episode_parented_off_the_request_thread() -> None:
    """F11: the two off-request-thread stages exercised by a real dispatch/settle."""
    control, exporter = recording_control_tracer(TelemetryLevel.COARSE)
    runtime = _runtime(control=control)

    workflow_id, task_id = _register_v2(runtime)
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item_id = engine.work_item_id_for_task(task_id)
    assert work_item_id is not None
    expected_trace_id = workflow_to_trace_id_int(workflow_id)
    expected_episode_span_id = derived_span_id(SpanIdKind.WORK_ITEM, work_item_id)

    _dispatch(runtime, task_id, control=control)

    dispatch_spans = spans_by_stage(exporter, "dispatch")
    assert len(dispatch_spans) == 1
    dispatch_span = dispatch_spans[0]
    assert dispatch_span.context is not None
    assert dispatch_span.context.trace_id == expected_trace_id
    assert dispatch_span.parent is not None
    assert dispatch_span.parent.span_id == expected_episode_span_id

    runtime.mark_succeeded(task_id, "wkr-1", {}, "2026-01-01T00:00:00Z")

    ds_drive_spans = spans_by_stage(exporter, "ds_drive")
    assert ds_drive_spans
    for span in ds_drive_spans:
        assert span.context is not None
        assert span.context.trace_id == expected_trace_id
        # No control span is its own trace root (F11's core assertion).
        assert span.parent is not None


def test_no_control_span_roots_its_own_trace() -> None:
    """Every one of the thirteen call sites exercised here lands under one trace id."""
    control, exporter = recording_control_tracer(TelemetryLevel.FULL)
    runtime = _runtime(control=control)

    workflow_id, task_id = _register_v2(runtime)
    _dispatch(runtime, task_id, control=control)
    runtime.mark_succeeded(task_id, "wkr-1", {}, "2026-01-01T00:00:00Z")

    expected_trace_id = workflow_to_trace_id_int(workflow_id)
    spans = exporter.get_finished_spans()
    control_spans = [s for s in spans if s.name.startswith("flowmesh.control.")]
    assert len(control_spans) >= 8
    for span in control_spans:
        assert span.context is not None
        assert span.context.trace_id == expected_trace_id
        assert span.parent is not None, f"{span.name} rooted its own trace"


def test_v1_task_dispatches_with_no_dispatch_span() -> None:
    """dispatch is v2-only: a v1 task that actually dispatches gets no control span."""
    control, exporter = recording_control_tracer(TelemetryLevel.COARSE)
    runtime = _runtime(control=control)

    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _ECHO_V1, format="native")
    )
    task_id = results[0].task_id
    assert runtime.orchestration_engine(workflow_id) is None

    _dispatch(runtime, task_id, control=control)

    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.DISPATCHED
    assert not spans_by_stage(exporter, "dispatch")


def test_off_constructs_no_spans_and_still_dispatches() -> None:
    control, exporter = recording_control_tracer(TelemetryLevel.OFF)
    runtime = _runtime(control=control)

    _workflow_id, task_id = _register_v2(runtime)
    _dispatch(runtime, task_id, control=control)
    runtime.mark_succeeded(task_id, "wkr-1", {}, "2026-01-01T00:00:00Z")

    assert exporter.get_finished_spans() == ()
    record = runtime.get_record(task_id)
    assert record is not None and record.status == TaskStatus.DONE
