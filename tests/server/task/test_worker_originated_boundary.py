"""The worker-originated mediated-tool-boundary path through the real runtime.

With the flag on, a ``search/v1`` boundary an agent emits is stripped to its digest by
the worker, dispatched as an off-lane operation pinned to the agent's own worker, and
its fenced outcome settles the boundary — the raw request never entering the ledger.
With the flag off the same boundary keeps its request and routes to the in-server
broker. If the origin worker is lost the boundary fails clean.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from server.config import OrchestrationConfig
from server.orchestration.state import WorkItemStatus
from server.orchestration.tool_dispatch import SEARCH_INTERFACE, ToolInvocationEnvelope
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.harness import BoundaryEventKind, HarnessCapsule
from shared.tasks.task_type import TaskType
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _register,
)
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep
from worker.lifecycle import PendingToolRequestStore

_TS = "2026-04-28T00:00:00Z"
_QUERY_TOKEN = "supernova-remnants"
_PAYLOAD = f'{{"query": "{_QUERY_TOKEN}", "max_results": 3}}'

_SEARCH_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: search-agent}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [search/v1], delegate: []}
            tools: [{name: web_search, interface: "search/v1"}]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""

_SCRIPT = [
    ScriptedStep(
        op="boundary",
        kind=BoundaryEventKind.INVOCATION,
        call="m0",
        interface=SEARCH_INTERFACE,
        payload=_PAYLOAD,
    ),
    ScriptedStep(op="complete", value_from="m0"),
]


class _WorkerStub:
    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(id=worker_id, node_id="nde-1", incarnation=7)

    def publish_interrupt(self, *args: Any) -> int:
        return 0


def _runtime(*, flag: bool) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerStub()),
        OrchestrationConfig(worker_originated_boundaries=flag),
        Path(tempfile.gettempdir()),
        logging.getLogger("wo-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def _dispatch_agent(runtime: TaskRuntime, task_id: str, worker: str = "wkr-1") -> Any:
    """Mimic a dispatch: pin the worker and run one scripted step, worker-side strip
    included, then report the step to the runtime."""
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id)
    assert engine is not None and dispatch is not None
    capsule = (
        HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
        if dispatch.capsule_blob is not None
        else None
    )
    engine.on_dispatched(task_id, worker)
    record = runtime._tasks[task_id]
    record.assigned_worker = worker
    record.status = TaskStatus.DISPATCHED
    result = ScriptedHarnessAdapter(_SCRIPT, "v1").start(
        task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes
    )
    if dispatch.worker_originated_boundaries:
        result = AgentEpisodeExecutor._capture_local_request(
            PendingToolRequestStore(), task_id, result
        )
    runtime.mark_succeeded(
        task_id, worker, {"agent_episode": result.model_dump(mode="json")}, _TS
    )
    return engine


def test_worker_originated_boundary_settles_and_keeps_payload_out_of_ledger() -> None:
    async def run() -> None:
        runtime = _runtime(flag=True)
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        engine = _dispatch_agent(runtime, writer)

        # An off-lane op is pinned to the agent's worker, carrying a permit bound to it.
        op_ids = [
            t
            for t, r in runtime._tasks.items()
            if r.task_type == TaskType.TOOL_OPERATION
        ]
        assert len(op_ids) == 1
        op_id = op_ids[0]
        assert runtime._tasks[op_id].selected_worker == ["wkr-1"]
        permit = runtime.tool_operation_dispatch(op_id)
        assert permit is not None and permit.target_id == "wkr-1"
        assert permit.target_generation == 7 and permit.agent_task_id == writer

        # The ledger holds the digest, never the raw request.
        snap = engine.to_snapshot()
        m0 = next(e for e in snap.boundary_events if e.call_correlation == "m0")
        assert m0.request_digest == permit.request_digest
        assert m0.request_payload is None
        assert _QUERY_TOKEN not in snap.model_dump_json()

        # The origin worker runs the op and reports a bounded outcome; it settles the
        # boundary and the resumed episode injects it and completes.
        runtime.mark_succeeded(
            op_id,
            "wkr-1",
            {"tool_operation": {"outcome": {"status": "success", "value": "sunny"}}},
            _TS,
        )
        assert op_id not in runtime._tasks  # the carrier is consumed
        _dispatch_agent(runtime, writer)  # resume: inject the outcome and complete
        writer_wi = engine.work_item(writer)
        assert writer_wi is not None and writer_wi.status is WorkItemStatus.SETTLED
        pub = engine.resolve_output(f"legacy:{writer}")
        assert pub is not None and pub.outcome.value == "success"

    asyncio.run(run())


def test_flag_off_routes_the_boundary_to_the_broker_with_its_request() -> None:
    async def run() -> None:
        runtime = _runtime(flag=False)
        broker: list[ToolInvocationEnvelope] = []
        runtime.set_tool_broker(broker.append)
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        _dispatch_agent(runtime, writer)

        # No off-lane op; the broker gets the boundary with its raw request intact.
        assert not [
            t
            for t, r in runtime._tasks.items()
            if r.task_type == TaskType.TOOL_OPERATION
        ]
        assert [e.interface for e in broker] == [SEARCH_INTERFACE]
        assert broker[0].request_payload is not None
        assert _QUERY_TOKEN in broker[0].request_payload

    asyncio.run(run())


def test_origin_worker_loss_fails_the_boundary_clean() -> None:
    async def run() -> None:
        runtime = _runtime(flag=True)
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        engine = _dispatch_agent(runtime, writer)
        assert engine.work_item(writer).status is WorkItemStatus.BLOCKED

        # The origin worker departs before the op settles: the boundary fails clean and
        # the workflow errors rather than resuming the agent with no outcome, and the
        # dead operation carrier is dropped.
        runtime.recover_tasks_for_worker("wkr-1")
        assert (
            runtime._tasks[writer].status == TaskStatus.FAILED
        )  # workflow errors clean
        assert not [
            t
            for t, r in runtime._tasks.items()
            if r.task_type == TaskType.TOOL_OPERATION
        ]

    asyncio.run(run())
