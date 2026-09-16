"""The v2 control-plane stage breakdown a real submission produces."""

import asyncio
import logging
import pathlib
import tempfile
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.services.metrics import MetricsRecorder
from server.services.profiling import build_profiler
from server.task.runtime import TaskRuntime

_V1 = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: shared}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""

_V2 = _V1.replace("flowmesh/v1", "flowmesh/v2")


class _CapturingRegistry:
    async def register_workflow_async(
        self, workflow_id: str, tasks: list, v2: Any = None
    ) -> None:
        return None

    async def save_task_states_async(self, items: list) -> None:
        return None

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        return None

    async def save_ledger_snapshot_async(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoopSecretVault:
    async def put(self, *args: Any, **kwargs: Any) -> None:
        return None


def _submit(body: str, *, enabled: bool) -> tuple[str, MetricsRecorder]:
    recorder = MetricsRecorder(
        pathlib.Path(tempfile.mkdtemp()),
        logging.getLogger("control-profiling-test"),
        enable_control_profiling=enabled,
    )
    worker_stub = SimpleNamespace(
        get_worker=lambda wid: SimpleNamespace(id=wid, node_id="nde-1"),
        publish_interrupt=lambda *a: 0,
    )
    runtime = TaskRuntime(
        cast(Any, _CapturingRegistry()),
        cast(Any, worker_stub),
        OrchestrationConfig(),
        pathlib.Path(tempfile.gettempdir()),
        logging.getLogger("control-profiling-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
        profiler=build_profiler(recorder, enabled=enabled),
    )
    workflow_id, _results = asyncio.run(runtime.register("owner", "org", body))
    return workflow_id, recorder


def test_a_v2_submission_decomposes_its_submit_window() -> None:
    workflow_id, recorder = _submit(_V2, enabled=True)
    submit = recorder.control_plane_breakdown()["workflows"][workflow_id]["windows"][
        "submit"
    ]
    assert set(submit["stages"]) >= {
        "compile_lower",
        "compile_assemble",
        "compile_validate",
        "engine_build",
        "ds_initial_advance",
        "ledger_snapshot",
    }
    assert submit["total_sec"] > 0.0


def test_the_submit_window_sum_excludes_the_nested_snapshot() -> None:
    workflow_id, recorder = _submit(_V2, enabled=True)
    submit = recorder.control_plane_breakdown()["workflows"][workflow_id]["windows"][
        "submit"
    ]
    partition = sum(
        entry["total_sec"]
        for name, entry in submit["stages"].items()
        if name != "ledger_snapshot"
    )
    assert submit["total_sec"] == pytest.approx(partition)
    assert submit["nested_sec"] == pytest.approx(
        submit["stages"]["ledger_snapshot"]["total_sec"]
    )


def test_a_v1_submission_records_no_control_plane_stage() -> None:
    workflow_id, recorder = _submit(_V1, enabled=True)
    assert workflow_id not in recorder.control_plane_breakdown()["workflows"]


def test_the_gate_off_records_nothing_for_a_v2_submission() -> None:
    _workflow_id, recorder = _submit(_V2, enabled=False)
    assert recorder.control_plane_breakdown() == {"workflows": {}}
