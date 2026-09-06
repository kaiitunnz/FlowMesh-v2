"""The worker-originated mediated-tool-boundary path through the real runtime.

A ``search/v1`` boundary an agent emits is stripped to its digest by the worker; the
runtime mints an audience-bound permit and relays it to the agent's own worker over the
attachment, and the worker's fenced outcome settles the boundary — the raw request never
entering the ledger. If the origin worker is lost the boundary fails clean.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from server.config import OrchestrationConfig
from server.orchestration.state import WorkItemStatus
from server.orchestration.tool_dispatch import SEARCH_INTERFACE
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.harness import BoundaryEventKind, HarnessCapsule
from shared.schemas.event import parse_event
from shared.tools.contract import (
    MediatedOperationOutcome,
    MediatedOperationPermit,
    ToolOutcome,
    ToolOutcomeStatus,
)
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _register,
)
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep
from worker.lifecycle import PendingToolRequestStore
from worker.supervisor_client import SupervisorClient

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
    def __init__(self) -> None:
        self.frames: list[tuple[str, str, dict[str, Any]]] = []

    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(id=worker_id, node_id="nde-1", incarnation=7)

    def publish_interrupt(self, *args: Any) -> int:
        return 0

    def publish_mediated_op(self, worker: Any, payload: Any) -> int:
        self.frames.append((worker.id, payload.frame_kind, payload.payload))
        return 0


def _runtime() -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerStub()),
        OrchestrationConfig(),
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
    result = AgentEpisodeExecutor._capture_local_request(
        PendingToolRequestStore(), task_id, result
    )
    runtime.mark_succeeded(
        task_id, worker, {"agent_episode": result.model_dump(mode="json")}, _TS
    )
    return engine


def _permit_frames(runtime: TaskRuntime) -> list[dict[str, Any]]:
    frames = cast(Any, runtime._worker_registry).frames
    return [payload for _, kind, payload in frames if kind == "permit"]


def _reap_frames(runtime: TaskRuntime) -> list[dict[str, Any]]:
    frames = cast(Any, runtime._worker_registry).frames
    return [payload for _, kind, payload in frames if kind == "reap"]


def _serialize_outcome_frame(outcome: MediatedOperationOutcome) -> dict[str, Any]:
    """The exact event frame the worker enqueues for a mediated outcome report."""
    client = SupervisorClient(
        worker_token="t",
        owner_principal=None,
        grpc_target="x",
        worker_namespace="ns",
        worker_cluster="c",
        worker_alias="a",
        logger=logging.getLogger("wo-frame"),
    )
    client._worker_id = "wkr-1"
    client._stub = cast(Any, object())
    client._event_ready.set()
    client.push_mediated_outcome(outcome)
    return cast(dict[str, Any], client._event_queue.get_nowait())


def test_worker_originated_boundary_settles_and_keeps_payload_out_of_ledger() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        engine = _dispatch_agent(runtime, writer)

        # A permit bound to the agent's worker is relayed over the attachment; no task.
        permits = _permit_frames(runtime)
        assert len(permits) == 1
        permit = MediatedOperationPermit.model_validate(permits[0])
        assert permit.target_id == "wkr-1" and permit.target_generation == 7
        assert permit.agent_task_id == writer

        # The ledger holds the digest, never the raw request.
        snap = engine.to_snapshot()
        m0 = next(e for e in snap.boundary_events if e.call_correlation == "m0")
        assert m0.request_digest == permit.request_digest
        assert m0.request_payload is None
        assert _QUERY_TOKEN not in snap.model_dump_json()

        # The origin worker reports a bounded outcome over the attachment; it settles
        # the boundary, reaps custody, and the resumed episode injects it and completes.
        runtime.settle_mediated_operation(
            MediatedOperationOutcome(
                permit_id=permit.permit_id,
                agent_task_id=writer,
                call_correlation="m0",
                invocation_id=permit.invocation_id,
                idempotency_key=permit.idempotency_key,
                outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="sunny"),
            )
        )
        assert len(_reap_frames(runtime)) == 1  # delete-on-committed-ack
        _dispatch_agent(runtime, writer)  # resume: inject the outcome and complete
        writer_wi = engine.work_item(writer)
        assert writer_wi is not None and writer_wi.status is WorkItemStatus.SETTLED
        pub = engine.resolve_output(f"legacy:{writer}")
        assert pub is not None and pub.outcome.value == "success"

    asyncio.run(run())


def test_worker_outcome_frame_settles_through_event_parse() -> None:
    """The worker's serialized outcome frame settles the boundary once parsed.

    The report ships as an ordinary worker event and the server reads the outcome from
    ``event.payload``; a frame that nested the outcome anywhere else would be dropped by
    the event listener and strand the boundary. This drives the worker's own
    serialization through ``parse_event`` and into ``settle_mediated_operation``.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        engine = _dispatch_agent(runtime, writer)
        permit = MediatedOperationPermit.model_validate(_permit_frames(runtime)[0])

        outcome = MediatedOperationOutcome(
            permit_id=permit.permit_id,
            agent_task_id=writer,
            call_correlation="m0",
            invocation_id=permit.invocation_id,
            idempotency_key=permit.idempotency_key,
            outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="sunny"),
        )
        event = parse_event(_serialize_outcome_frame(outcome))
        assert "outcome" in event.payload  # not absorbed as a top-level extra

        runtime.settle_mediated_operation(
            MediatedOperationOutcome.model_validate(event.payload["outcome"])
        )
        assert len(_reap_frames(runtime)) == 1
        _dispatch_agent(runtime, writer)  # resume: inject the outcome and complete
        writer_wi = engine.work_item(writer)
        assert writer_wi is not None and writer_wi.status is WorkItemStatus.SETTLED

    asyncio.run(run())


def test_origin_worker_loss_fails_the_boundary_clean() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        engine = _dispatch_agent(runtime, writer)
        assert engine.work_item(writer).status is WorkItemStatus.BLOCKED

        # The origin worker departs before the op settles: the boundary fails clean and
        # the workflow errors rather than resuming the agent with no outcome, and the
        # stale pending-op mapping is dropped.
        runtime.recover_tasks_for_worker("wkr-1")
        assert runtime._tasks[writer].status == TaskStatus.FAILED
        assert not runtime._pending_ops

    asyncio.run(run())
