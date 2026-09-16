"""The v2 control-plane stage breakdown a real submission produces."""

import asyncio
import logging
import pathlib
import statistics
import tempfile
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.resident.service import _subject_workflow_id
from server.resident.state import InvocationSubject, InvocationSubjectKind
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

_V1_WIDE = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: wide}
spec:
  taskType: echo
  resources: {hardware: {cpu: 1, memory: 256Mi}}
  graph:
    nodes:
""" + "\n".join(
    f"      - name: n{i}\n"
    f"        spec: {{taskType: echo, data: {{type: list, items: [x]}}}}"
    for i in range(30)
)

_V2_WIDE = _V1_WIDE.replace("flowmesh/v1", "flowmesh/v2")

# Enough repetitions that the median is not one scheduling hiccup.
_WALL_SAMPLES = 15


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


def _runtime_for(recorder: MetricsRecorder, *, enabled: bool) -> TaskRuntime:
    worker_stub = SimpleNamespace(
        get_worker=lambda wid: SimpleNamespace(id=wid, node_id="nde-1"),
        publish_interrupt=lambda *a: 0,
    )
    return TaskRuntime(
        cast(Any, _CapturingRegistry()),
        cast(Any, worker_stub),
        OrchestrationConfig(),
        pathlib.Path(tempfile.gettempdir()),
        logging.getLogger("control-profiling-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
        profiler=build_profiler(recorder, enabled=enabled),
    )


def _submit(
    body: str, *, enabled: bool
) -> tuple[str, MetricsRecorder, TaskRuntime, list]:
    recorder = MetricsRecorder(
        pathlib.Path(tempfile.mkdtemp()),
        logging.getLogger("control-profiling-test"),
        enable_control_profiling=enabled,
    )
    runtime = _runtime_for(recorder, enabled=enabled)
    workflow_id, results = asyncio.run(runtime.register("owner", "org", body))
    return workflow_id, recorder, runtime, results


def test_a_v2_submission_decomposes_its_submit_window() -> None:
    workflow_id, recorder, _runtime, _results = _submit(_V2, enabled=True)
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


def test_the_submit_window_total_is_the_sum_of_its_stages() -> None:
    workflow_id, recorder, _runtime, _results = _submit(_V2, enabled=True)
    submit = recorder.control_plane_breakdown()["workflows"][workflow_id]["windows"][
        "submit"
    ]
    partition = sum(entry["total_sec"] for entry in submit["stages"].values())
    assert submit["total_sec"] == pytest.approx(partition)


def test_the_submit_window_counts_the_ledger_snapshot() -> None:
    """The snapshot is a sibling of the submit stages, so the total includes it."""
    workflow_id, recorder, _runtime, _results = _submit(_V2, enabled=True)
    submit = recorder.control_plane_breakdown()["workflows"][workflow_id]["windows"][
        "submit"
    ]
    assert "ledger_snapshot" in submit["stages"]
    assert submit["nested"] == {}
    assert submit["nested_sec"] == 0.0


def test_the_submit_window_accounts_for_the_measured_v2_register_delta() -> None:
    """Anchored to an independent wall, not to the breakdown's own partition.

    A window total that dropped a stage would still self-reconcile, so the check
    that catches it compares against time measured outside the instrument: the
    difference between submitting the same body on v1 and on v2 is what the v2
    control plane costs, and the submit window is meant to account for it.
    """

    def median_register_wall(body: str) -> tuple[float, dict[str, Any]]:
        recorder = MetricsRecorder(
            pathlib.Path(tempfile.mkdtemp()),
            logging.getLogger("control-profiling-test"),
            enable_control_profiling=True,
        )
        runtime = _runtime_for(recorder, enabled=True)
        for _ in range(3):
            asyncio.run(runtime.register("owner", "org", body))
        walls: list[float] = []
        submit: dict[str, Any] = {}
        for _ in range(_WALL_SAMPLES):
            started = time.perf_counter()
            workflow_id, _results = asyncio.run(runtime.register("owner", "org", body))
            walls.append(time.perf_counter() - started)
            entry = recorder.control_plane_breakdown()["workflows"].get(workflow_id)
            if entry:
                submit = entry["windows"]["submit"]
        return statistics.median(walls), submit

    v1_wall, _ = median_register_wall(_V1_WIDE)
    v2_wall, submit = median_register_wall(_V2_WIDE)
    delta = v2_wall - v1_wall
    assert delta > 0, f"v2 register ({v2_wall}) is not slower than v1 ({v1_wall})"
    assert submit["total_sec"] >= delta * 0.5, (
        f"submit window {submit['total_sec']:.6f}s accounts for too little of the "
        f"measured {delta:.6f}s v1-to-v2 register() delta: {sorted(submit['stages'])}"
    )


def test_a_v1_submission_records_no_control_plane_stage() -> None:
    workflow_id, recorder, _runtime, _results = _submit(_V1, enabled=True)
    assert workflow_id not in recorder.control_plane_breakdown()["workflows"]


def test_the_gate_off_records_nothing_for_a_v2_submission() -> None:
    _workflow_id, recorder, _runtime, _results = _submit(_V2, enabled=False)
    assert recorder.control_plane_breakdown() == {"workflows": {}}


def test_a_ledger_transition_records_a_post_start_drive() -> None:
    workflow_id, recorder, runtime, results = _submit(_V2, enabled=True)
    root = next(r for r in results if r.graph_node_name == "a")
    runtime._engines[workflow_id].on_succeeded(root.task_id, empty=True)
    post_start = recorder.control_plane_breakdown()["workflows"][workflow_id][
        "windows"
    ]["post_start"]
    assert post_start["stages"]["ds_drive"]["count"] == 1


def test_a_serve_subject_attributes_to_no_workflow() -> None:
    workflow = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")
    external = InvocationSubject(kind=InvocationSubjectKind.EXTERNAL, id="user-1")
    assert _subject_workflow_id(workflow) == "wfl-1"
    assert _subject_workflow_id(external) is None


def test_a_v1_workflow_is_not_a_v2_engine_owner() -> None:
    """The dispatch stage keys off this predicate, so a v1 body must not satisfy it."""
    v1_id, _recorder, v1_runtime, _results = _submit(_V1, enabled=True)
    v2_id, _recorder2, v2_runtime, _results2 = _submit(_V2, enabled=True)
    assert v1_runtime.is_v2_workflow(v1_id) is False
    assert v2_runtime.is_v2_workflow(v2_id) is True
