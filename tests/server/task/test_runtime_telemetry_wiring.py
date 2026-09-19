"""The TaskRuntime carries its tracers into every engine it builds.

The runtime constructs an orchestration engine on two paths — a fresh submission and a
restart's rehydration — and a tracer omitted on either one goes unnoticed: the engine
works, the suite stays green, and the only symptom is a workflow whose spans stop at the
restart. Both paths are asserted here through the runtime's own API rather than by
constructing an engine directly, because constructing one directly reproduces whatever
argument list the runtime already passes, including an omission.
"""

import logging
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.task.runtime import TaskRuntime
from shared.telemetry.config import TelemetryLevel
from tests.server.task.test_v2_orchestration import (
    _TS,
    AUTORESEARCH,
    FakeRegistry,
    _NoopSecretVault,
    _register,
    _worker,
    _WorkerRegistryStub,
    _write_result,
)
from tests.server.telemetry_helpers import recording_control_tracer, recording_tracer


def _runtime(
    registry: FakeRegistry, results_dir: Any, name: str
) -> tuple[TaskRuntime, Any, Any]:
    control, control_exporter = recording_control_tracer(TelemetryLevel.FULL)
    tracer, span_exporter, config = recording_tracer(TelemetryLevel.FULL)
    runtime = TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        results_dir,
        logging.getLogger(name),
        secret_vault=cast(Any, _NoopSecretVault()),
        control=control,
        tracer=tracer,
        telemetry=config,
    )
    return runtime, control_exporter, span_exporter


def _control_span_names(exporter: Any) -> set[str]:
    return {
        span.name
        for span in exporter.get_finished_spans()
        if span.name.startswith("flowmesh.control.")
    }


@pytest.mark.anyio
async def test_a_submitted_workflow_records_control_stages(tmp_path: Any) -> None:
    registry = FakeRegistry()
    runtime, control_exporter, _ = _runtime(registry, tmp_path, "submit")

    _, ids = await _register(runtime, AUTORESEARCH)
    runtime.mark_dispatched(ids["planner"], cast(Any, _worker()))
    runtime.mark_succeeded(ids["planner"], "wkr-1", {}, _TS)

    assert _control_span_names(control_exporter)


@pytest.mark.anyio
async def test_a_rehydrated_workflow_still_records_control_stages(
    tmp_path: Any,
) -> None:
    registry = FakeRegistry()
    runtime, _, _ = _runtime(registry, tmp_path, "live")
    workflow_id, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    runtime.mark_dispatched(planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", {}, _TS)
    _write_result(tmp_path, planner, ["h1", "h2", "h3"])

    restored, control_exporter, _ = _runtime(registry, tmp_path, "rehydrate")
    assert await restored.rehydrate() == 1
    engine = restored.orchestration_engine(workflow_id)
    assert engine is not None

    # ``ds_drive`` is emitted from inside the engine, so it is the stage that proves
    # the rehydrated engine itself carries the tracer -- the stages the runtime emits
    # around the engine appear either way and would make this pass for free.
    assert "flowmesh.control.ds_drive" in _control_span_names(control_exporter)


@pytest.mark.anyio
async def test_a_rehydrated_workflow_still_emits_ledger_spans(tmp_path: Any) -> None:
    registry = FakeRegistry()
    runtime, _, _ = _runtime(registry, tmp_path, "live")
    workflow_id, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    runtime.mark_dispatched(planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", {}, _TS)
    _write_result(tmp_path, planner, ["h1", "h2", "h3"])

    restored, _, span_exporter = _runtime(registry, tmp_path, "rehydrate")
    assert await restored.rehydrate() == 1
    assert restored.orchestration_engine(workflow_id) is not None

    assert {
        span.name
        for span in span_exporter.get_finished_spans()
        if not span.name.startswith("flowmesh.control.")
    }


@pytest.mark.anyio
async def test_a_rehydrated_workflow_dispatches_with_a_traceparent(
    tmp_path: Any,
) -> None:
    """A rehydrated workflow's dispatches still name their episode, so a worker picked
    up after a restart joins the workflow's trace rather than rooting its own."""
    registry = FakeRegistry()
    runtime, _, _ = _runtime(registry, tmp_path, "live")
    workflow_id, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    runtime.mark_dispatched(planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", {}, _TS)
    _write_result(tmp_path, planner, ["h1", "h2", "h3"])

    restored, _, _ = _runtime(registry, tmp_path, "rehydrate")
    assert await restored.rehydrate() == 1
    assert restored.orchestration_engine(workflow_id) is not None

    children = [
        task_id
        for task_id in await registry.get_dynamic_task_ids_async(workflow_id)
        if restored.dispatch_traceparent(task_id) is not None
    ]
    assert children, "a rehydrated workflow's dispatches must carry a traceparent"
