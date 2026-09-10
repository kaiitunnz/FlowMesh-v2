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

from pydantic import SecretStr

from server.config import OrchestrationConfig
from server.orchestration.state import WorkItemStatus
from server.orchestration.tool_dispatch import MODEL_INTERFACE, SEARCH_INTERFACE
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.harness import (
    BoundaryEventKind,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
)
from shared.private_state import OwnerFence
from shared.schemas.event import parse_event
from shared.tools.contract import (
    AgentModelTurnProposal,
    MediatedOperationOutcome,
    MediatedOperationPermit,
    ToolOutcome,
    ToolOutcomeStatus,
)
from shared.tools.facade import (
    FacadeCallMember,
    FacadeCompletionMode,
    FacadeTurnGroup,
)
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _register,
)
from worker.egress import PendingEgressRequestStore
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep
from worker.supervisor_client import SupervisorClient

_HOLDER = OwnerFence(worker_id="wkr-1", incarnation=1)

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

_PROMPT_TOKEN = "andromeda-galaxy"
_MODEL_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: model-agent}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
          model_binding: {mode: openai, url: "http://up/v1", model: qwen}
"""

_MODEL_SCRIPT = [
    ScriptedStep(
        op="boundary",
        kind=BoundaryEventKind.INVOCATION,
        call="m0",
        interface=MODEL_INTERFACE,
        payload=_PROMPT_TOKEN,
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


class _StubVault:
    """A model-secret vault that stores and resolves a workflow-scoped key in memory."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], SecretStr] = {}

    async def store(self, workflow_id: str, ref: str, secret: SecretStr) -> None:
        self._store[(workflow_id, ref)] = secret

    def resolve(self, workflow_id: str, ref: str | None) -> SecretStr | None:
        return self._store.get((workflow_id, ref)) if ref else None

    def purge(self, workflow_id: str) -> None:
        return None


def _runtime(vault: Any | None = None) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerStub()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("wo-test"),
        secret_vault=cast(Any, vault or _NoopSecretVault()),
    )


def _dispatch_agent(
    runtime: TaskRuntime,
    task_id: str,
    worker: str = "wkr-1",
    script: list[ScriptedStep] = _SCRIPT,
) -> Any:
    """Mimic a dispatch: pin the worker and run one scripted step, worker-side strip
    included, then report the step to the runtime."""
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id, _HOLDER)
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
    result = ScriptedHarnessAdapter(script, "v1").start(
        task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes
    )
    result = AgentEpisodeExecutor._capture_local_request(
        PendingEgressRequestStore(), task_id, result, dispatch.model_binding
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


def _deny_frames(runtime: TaskRuntime) -> list[dict[str, Any]]:
    frames = cast(Any, runtime._worker_registry).frames
    return [payload for _, kind, payload in frames if kind == "deny"]


def _hold_dispatch(runtime: TaskRuntime, task_id: str, worker: str = "wkr-1") -> Any:
    """Pin a worker and hold the agent mid-turn without running the episode.

    Mimics a synchronous-turn-only harness (Codex) whose lane is held across an in-turn
    model call: the activation and work item are live and DISPATCHED, but no boundary
    settles, so ``authorize_model_turn`` mints from the propose alone.
    """
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    assert engine is not None
    engine.on_dispatched(task_id, worker)
    record = runtime._tasks[task_id]
    record.assigned_worker = worker
    record.status = TaskStatus.DISPATCHED
    return engine


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


def test_worker_originated_model_boundary_mints_a_worker_permit() -> None:
    """An external model boundary is worker-originated: a permit relays to the agent's
    worker under the ``model`` interface, and the ledger holds only the request digest.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]

        engine = _dispatch_agent(runtime, writer, script=_MODEL_SCRIPT)

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        permit = MediatedOperationPermit.model_validate(permits[0])
        assert permit.interface == MODEL_INTERFACE
        assert permit.target_id == "wkr-1" and permit.agent_task_id == writer

        # The prompt stays worker-private; only its digest reaches the ledger.
        snap = engine.to_snapshot()
        m0 = next(e for e in snap.boundary_events if e.call_correlation == "m0")
        assert m0.request_digest == permit.request_digest
        assert m0.request_payload is None
        assert _PROMPT_TOKEN not in snap.model_dump_json()

        runtime.settle_mediated_operation(
            MediatedOperationOutcome(
                permit_id=permit.permit_id,
                agent_task_id=writer,
                call_correlation="m0",
                invocation_id=permit.invocation_id,
                idempotency_key=permit.idempotency_key,
                outcome=ToolOutcome(
                    status=ToolOutcomeStatus.SUCCESS, value="a completion"
                ),
            )
        )
        _dispatch_agent(runtime, writer, script=_MODEL_SCRIPT)
        writer_wi = engine.work_item(writer)
        assert writer_wi is not None and writer_wi.status is WorkItemStatus.SETTLED

    asyncio.run(run())


_BYOK_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: byok-agent}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
          model_binding:
            {mode: openai, url: "http://up/v1", model: qwen, api_key: sk-byok-abc}
"""


def test_model_permit_carries_the_workflow_key_and_never_the_ledger() -> None:
    """A binding that pins its own key resolves it onto the one-use permit; the key
    rides the permit down to the worker and never enters the ledger."""

    async def run() -> None:
        runtime = _runtime(_StubVault())
        _, ids = await _register(runtime, _BYOK_WF)
        writer = ids["writer"]

        engine = _dispatch_agent(runtime, writer, script=_MODEL_SCRIPT)

        permit = MediatedOperationPermit.model_validate(_permit_frames(runtime)[0])
        assert permit.credential == "sk-byok-abc"
        # The key never lands in the durable ledger, and the permit hides it from repr.
        assert "sk-byok-abc" not in engine.to_snapshot().model_dump_json()
        assert "sk-byok-abc" not in repr(permit)

    asyncio.run(run())


def test_held_model_turn_mints_a_worker_permit_without_a_settle() -> None:
    """A held in-turn model propose mints a permit relayed to the agent's worker.

    The permit is model-interface, audience-bound, and fenced on the worker-computed
    digest, yet no suspending boundary is recorded and no server settle is expected: the
    held turn consumes the outcome in-worker, so the runtime tracks no pending op.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]

        _hold_dispatch(runtime, writer)
        runtime.authorize_model_turn(
            AgentModelTurnProposal(
                agent_task_id=writer, call_correlation="t0", request_digest="deadbeef"
            )
        )

        permits = _permit_frames(runtime)
        assert len(permits) == 1 and not _deny_frames(runtime)
        permit = MediatedOperationPermit.model_validate(permits[0])
        assert permit.interface == MODEL_INTERFACE
        assert permit.target_id == "wkr-1" and permit.target_generation == 7
        assert permit.agent_task_id == writer and permit.call_correlation == "t0"
        assert permit.request_digest == "deadbeef"
        assert permit.invocation_id and permit.idempotency_key
        # No suspending boundary and no pending op: the held turn settles in-worker.
        assert not runtime._pending_ops

    asyncio.run(run())


def test_held_model_turn_permit_carries_the_workflow_key() -> None:
    """A held model turn on a byok binding resolves the pinned key onto the permit."""

    async def run() -> None:
        runtime = _runtime(_StubVault())
        _, ids = await _register(runtime, _BYOK_WF)
        writer = ids["writer"]

        _hold_dispatch(runtime, writer)
        runtime.authorize_model_turn(
            AgentModelTurnProposal(
                agent_task_id=writer, call_correlation="t0", request_digest="d"
            )
        )

        permit = MediatedOperationPermit.model_validate(_permit_frames(runtime)[0])
        assert permit.credential == "sk-byok-abc"
        assert "sk-byok-abc" not in repr(permit)

    asyncio.run(run())


def test_held_model_turn_denied_relays_a_deny_frame() -> None:
    """An agent with no model authority is denied: a deny frame, never a permit.

    The held turn fails fast on the deny frame rather than waiting out the permit
    deadline, and no permit is minted for an unauthorized egress.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        _hold_dispatch(runtime, writer)
        runtime.authorize_model_turn(
            AgentModelTurnProposal(
                agent_task_id=writer, call_correlation="t0", request_digest="d"
            )
        )

        assert not _permit_frames(runtime)
        denies = _deny_frames(runtime)
        assert len(denies) == 1
        assert denies[0]["agent_task_id"] == writer
        assert denies[0]["call_correlation"] == "t0"

    asyncio.run(run())


def _search_group(agent: str, base: int, digest: str) -> FacadeTurnGroup:
    gid = f"{agent}:{base}"
    member = FacadeCallMember(
        ordinal=0,
        kind=BoundaryEventKind.INVOCATION,
        completion_mode=FacadeCompletionMode.AWAIT_OUTCOME,
        call_correlation=f"{gid}:0",
        harness_call_id="call0",
        tool_name="web_search",
        interface_or_region=SEARCH_INTERFACE,
        request_digest=digest,
    )
    return FacadeTurnGroup(
        group_id=gid, activation_id=agent, turn_id=str(base), members=(member,)
    )


def test_worker_facade_group_routes_search_by_digest_to_worker_egress() -> None:
    """A worker-reported facade group routes its digest-bearing search to the worker.

    The report records the group under the busy fence; the clean-turn completion routes
    it, and the search member dispatches to the agent's own worker under its digest — a
    permit relayed over the attachment, never the in-server broker.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)

        runtime.receive_worker_facade_group(writer, _search_group(writer, 0, "sha-xyz"))
        assert runtime.has_pending_facade(writer)
        # A second group while one is open is refused; the first stays.
        runtime.receive_worker_facade_group(writer, _search_group(writer, 1, "sha-2"))

        # The clean-turn completion routes the pending group and dispatches the search.
        capsule = HarnessCapsule(
            backend=HarnessBackendKey(backend="scripted", version="v1"), blob="cap"
        )
        completion = HarnessResult(
            kind=HarnessResultKind.COMPLETION, value="done", capsule=capsule
        )
        runtime.mark_succeeded(
            writer,
            "wkr-1",
            {"agent_episode": completion.model_dump(mode="json")},
            _TS,
        )

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        permit = MediatedOperationPermit.model_validate(permits[0])
        assert permit.interface == SEARCH_INTERFACE
        assert permit.request_digest == "sha-xyz"
        assert permit.call_correlation == f"{writer}:0:0"

    asyncio.run(run())


def test_completion_carrying_its_facade_group_routes_it_not_done() -> None:
    """A captured group rides the completion's own metadata and routes atomically.

    The worker carries the captured group in the same ``TASK_SUCCEEDED`` payload as the
    turn completion, so the group is ingested and routed in one ordered event — never on
    a separate channel that could deliver it after the completion (settling the episode
    DONE and dropping its searches) or drop it entirely. The completion here carries the
    group with no prior report: the search must still dispatch and the episode must not
    settle DONE.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)

        group = _search_group(writer, 0, "sha-carry")
        capsule = HarnessCapsule(
            backend=HarnessBackendKey(backend="scripted", version="v1"), blob="cap"
        )
        completion = HarnessResult(
            kind=HarnessResultKind.COMPLETION, value="done", capsule=capsule
        )
        runtime.mark_succeeded(
            writer,
            "wkr-1",
            {
                "agent_episode": completion.model_dump(mode="json"),
                "agent_episode_facade_group": group.model_dump(mode="json"),
            },
            _TS,
        )

        # The carried group routed its search to the worker; the episode did not settle
        # DONE on the clean-turn placeholder.
        permits = _permit_frames(runtime)
        assert len(permits) == 1
        permit = MediatedOperationPermit.model_validate(permits[0])
        assert permit.interface == SEARCH_INTERFACE
        assert permit.request_digest == "sha-carry"
        record = runtime.get_record(writer)
        assert record is not None and record.status is not TaskStatus.DONE

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


def test_cancellation_reaps_a_pending_mediated_op() -> None:
    """Cancelling a workflow reaps its agents' outstanding mediated egress.

    A cancelled agent's in-flight operation is dropped rather than left to egress and
    report into a boundary no longer wanted: the worker gets a reap and the pending-op
    mapping clears.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]

        _dispatch_agent(runtime, writer)
        assert runtime._pending_ops  # a permit is outstanding on the origin worker

        runtime.cancel_workflow(runtime._tasks[writer].workflow_id)
        assert len(_reap_frames(runtime)) == 1
        assert not runtime._pending_ops

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
