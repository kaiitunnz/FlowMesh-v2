"""The worker-originated mediated-tool-boundary path through the real runtime.

A ``search/v1`` boundary an agent emits is stripped to its digest by the worker; the
runtime mints an audience-bound permit and relays it to the agent's own worker over the
attachment, and the worker's fenced outcome settles the boundary — the raw request never
entering the ledger. If the origin worker is lost the boundary fails clean.
"""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.config import AgentBindingConfig, OrchestrationConfig
from server.orchestration.state import WorkItemStatus
from server.orchestration.tool_dispatch import MODEL_INTERFACE, SEARCH_INTERFACE
from server.registries.worker import Worker
from server.task.models import PublishGate, TaskStatus
from server.task.runtime import TaskRuntime
from server.task.runtime.facade import _is_default_url, _OpCredential
from shared.harness import (
    BoundaryEventKind,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
)
from shared.private_state import (
    OwnerFence,
    PrivateStateSealReport,
    PrivateStateUnavailableReason,
)
from shared.schemas.event import WorkerEvent, parse_event
from shared.tasks.specs import ModelBindingMode
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
from shared.tools.search.schema import parse_search_request
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader
from tests.server.task.test_private_state_ledger import _manifest
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
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
    """Records the frames control relays and applies each reap to one worker's egress
    store, which the dispatched steps capture into."""

    def __init__(self) -> None:
        self.frames: list[tuple[str, str, dict[str, Any]]] = []
        self.egress = PendingEgressRequestStore()
        self.node_alias = "box"

    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(
            id=worker_id, node_id="nde-1", node_alias=self.node_alias, incarnation=7
        )

    def publish_interrupt(self, *args: Any) -> int:
        return 0

    def publish_revoke(self, *args: Any) -> int:
        return 0

    def publish_mediated_op(self, worker: Any, payload: Any) -> int:
        self.frames.append((worker.id, payload.frame_kind, payload.payload))
        if payload.frame_kind == "reap":
            self.egress.delete(
                payload.payload["agent_task_id"], payload.payload["call_correlation"]
            )
        return 0


class _StubVault(InMemoryCredentialVault):
    """A credential vault whose keys outlive their workflow until dropped."""

    def purge(self, workflow_id: str) -> None:
        return None

    def expire_all(self) -> None:
        self.redis.hashes.clear()


def _runtime(
    vault: Any | None = None,
    assigned: list[tuple[str, str]] | None = None,
    config: OrchestrationConfig | None = None,
) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerStub()),
        config or OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("wo-test"),
        credential_vault=cast(Any, vault or InMemoryCredentialVault()),
        content_scope_authority=(
            None
            if assigned is None
            else lambda idem, scope: assigned.append((idem, scope))
        ),
    )


def _dispatch_agent(
    runtime: TaskRuntime,
    task_id: str,
    worker: str = "wkr-1",
    script: list[ScriptedStep] = _SCRIPT,
    seal_in: Path | None = None,
) -> Any:
    """Mimic a dispatch: pin the worker and run one scripted step, worker-side strip
    included, then report the step to the runtime, with the private state the worker
    sealed under ``seal_in`` when given."""
    engine, payload = _run_agent_step(runtime, task_id, worker, script, seal_in)
    runtime.mark_succeeded(task_id, worker, payload, _TS)
    return engine


def _run_agent_step(
    runtime: TaskRuntime,
    task_id: str,
    worker: str = "wkr-1",
    script: list[ScriptedStep] = _SCRIPT,
    seal_in: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Pin the worker and run one scripted step on it; return the step's report."""
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
        _egress(runtime), task_id, result, dispatch.model_binding, None
    )
    payload: dict[str, Any] = {"agent_episode": result.model_dump(mode="json")}
    if seal_in is not None and (attachment := dispatch.private_state_attachment):
        manifest = _manifest(
            seal_in, attachment.reference_id, attachment.generation + 1
        )
        payload["agent_episode_private_state"] = PrivateStateSealReport(
            manifest=manifest, write_epoch=attachment.write_epoch
        ).model_dump(mode="json")
    return engine, payload


def _egress(runtime: TaskRuntime) -> PendingEgressRequestStore:
    """The egress store of the worker the runtime relays to."""
    return cast(Any, runtime._worker_registry).egress


def _permit_frames(runtime: TaskRuntime) -> list[dict[str, Any]]:
    frames = cast(Any, runtime._worker_registry).frames
    return [payload for _, kind, payload in frames if kind == "permit"]


def _reap_frames(runtime: TaskRuntime) -> list[dict[str, Any]]:
    frames = cast(Any, runtime._worker_registry).frames
    return [payload for _, kind, payload in frames if kind == "reap"]


def _deny_frames(runtime: TaskRuntime) -> list[dict[str, Any]]:
    frames = cast(Any, runtime._worker_registry).frames
    return [payload for _, kind, payload in frames if kind == "deny"]


HELD_DISPATCH = "dsp-held"


def _propose_held_turn(runtime: TaskRuntime, writer: str, digest: str) -> None:
    """Propose the held agent's turn ``t0`` from its dispatch on its worker."""
    runtime.authorize_model_turn(
        AgentModelTurnProposal(
            agent_task_id=writer,
            call_correlation="t0",
            request_digest=digest,
            dispatch_id=HELD_DISPATCH,
        ),
        "wkr-1",
    )


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
    record.dispatch_id = HELD_DISPATCH
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
    _generation, frame = cast(
        tuple[int, dict[str, Any]], client._event_queue.get_nowait()
    )
    return frame


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
        pub = engine.output_publication(f"legacy:{writer}")
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
        _propose_held_turn(runtime, writer, "deadbeef")

        permits = _permit_frames(runtime)
        assert len(permits) == 1 and not _deny_frames(runtime)
        permit = MediatedOperationPermit.model_validate(permits[0])
        assert permit.interface == MODEL_INTERFACE
        assert permit.target_id == "wkr-1" and permit.target_generation == 7
        assert permit.agent_task_id == writer and permit.call_correlation == "t0"
        assert permit.request_digest == "deadbeef"
        assert permit.invocation_id and permit.idempotency_key
        # No suspending boundary and no pending op: the held turn settles in-worker.
        assert not runtime._mediated_ops.pending_ops

    asyncio.run(run())


def test_held_model_turn_permit_carries_the_workflow_key() -> None:
    """A held model turn on a byok binding resolves the pinned key onto the permit."""

    async def run() -> None:
        runtime = _runtime(_StubVault())
        _, ids = await _register(runtime, _BYOK_WF)
        writer = ids["writer"]

        _hold_dispatch(runtime, writer)
        _propose_held_turn(runtime, writer, "d")

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
        _propose_held_turn(runtime, writer, "d")

        assert not _permit_frames(runtime)
        denies = _deny_frames(runtime)
        assert len(denies) == 1
        assert denies[0]["agent_task_id"] == writer
        assert denies[0]["call_correlation"] == "t0"
        assert denies[0]["reason"] == "model turn egress denied"

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
        # A permit is outstanding on the origin worker.
        assert runtime._mediated_ops.pending_ops

        runtime.cancel_workflow(runtime._tasks[writer].workflow_id)
        assert len(_reap_frames(runtime)) == 1
        assert not runtime._mediated_ops.pending_ops

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
        runtime.recover_tasks_for_worker("wkr-1", spend_attempt=True)
        assert runtime._tasks[writer].status == TaskStatus.FAILED
        assert not runtime._mediated_ops.pending_ops

    asyncio.run(run())


def test_the_permit_carries_the_scope_its_result_materializes_under() -> None:
    """A boundary's outcome lands in the task owner's scope, not the worker's.

    The permit is what tells the egressing worker where to write, and the worker holds
    store access for exactly the scope control assigned its task, so a permit carrying
    none cannot materialize its outcome at all.
    """

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        _dispatch_agent(runtime, ids["writer"])

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        assert permits[0]["content_scope"] == "org"

    asyncio.run(run())


def test_control_records_the_scope_the_boundary_finalizes_under() -> None:
    """The scope is recorded against the permit's key, before the worker can report.

    The egressing worker reports the finalization its outcome produced, and the binding
    is checked against the scope control assigned that key rather than one the worker
    names for itself.
    """

    async def run() -> None:
        assigned: list[tuple[str, str]] = []
        runtime = _runtime(assigned=assigned)
        _, ids = await _register(runtime, _SEARCH_WF)
        _dispatch_agent(runtime, ids["writer"])

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        assert assigned == [(permits[0]["idempotency_key"], "org")]

    asyncio.run(run())


_DEFAULT_URL = "http://gateway/v1"


_DEFAULT_URL_CONFIG = OrchestrationConfig(
    agent_binding=AgentBindingConfig(
        default_mode=ModelBindingMode.OPENAI,
        default_url=_DEFAULT_URL,
        default_model="q",
    )
)


def _model_wf(binding: str) -> str:
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: model-agent}}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {{invoke: [model], delegate: []}}
            tools: [{{name: model}}]
          harness: {{backend: scripted, version: v1, params: {{script: []}}}}
          {binding}
"""


@pytest.mark.parametrize(
    ("binding", "granted"),
    [
        ("", True),
        (f'model_binding: {{mode: openai, url: "{_DEFAULT_URL}/", model: q}}', True),
        ('model_binding: {mode: openai, url: "http://author/v1", model: q}', False),
    ],
)
def test_only_the_deployment_model_url_is_granted_the_deployment_key(
    binding: str, granted: bool
) -> None:
    """The deployment key is granted on the permit only for the current default url."""

    async def run() -> None:
        runtime = _runtime(_StubVault(), config=_DEFAULT_URL_CONFIG)
        _, ids = await _register(runtime, _model_wf(binding))
        writer = ids["writer"]

        _dispatch_agent(runtime, writer, script=_MODEL_SCRIPT)
        _hold_dispatch(runtime, writer)
        _propose_held_turn(runtime, writer, "d")

        permits = [
            MediatedOperationPermit.model_validate(p) for p in _permit_frames(runtime)
        ]
        assert len(permits) == 2
        for permit in permits:
            assert permit.credential is None
            assert permit.deployment_credential is granted

    asyncio.run(run())


@pytest.mark.parametrize(
    ("url", "default_url", "granted"),
    [
        ("http://gateway/v1", "http://gateway/v1/", True),
        ("http://gateway/v1", None, False),
        ("http://gateway/v1", "", False),
        (None, "http://gateway/v1", False),
        ("", "", False),
        ("http://gateway/v1", "http://gateway/v2", False),
    ],
)
def test_only_a_set_matching_default_url_is_the_default(
    url: str | None, default_url: str | None, granted: bool
) -> None:
    assert _is_default_url(url, default_url) is granted


def test_a_resolved_credential_hides_its_value_from_repr() -> None:
    assert "sk-secret" not in repr(_OpCredential(credential="sk-secret"))


_GONE_KEY_WF = _model_wf(
    f'model_binding: {{mode: openai, url: "{_DEFAULT_URL}", model: q, '
    "api_key: sk-byok-gone}"
)


def test_a_gone_vaulted_key_fails_the_boundary_without_a_permit() -> None:
    """A pinned key missing from the vault fails the boundary, even on the default URL:
    the deployment key is never substituted and nothing egresses."""

    async def run() -> None:
        vault = _StubVault()
        runtime = _runtime(vault, config=_DEFAULT_URL_CONFIG)
        _, ids = await _register(runtime, _GONE_KEY_WF)
        writer = ids["writer"]
        vault.expire_all()

        _dispatch_agent(runtime, writer, script=_MODEL_SCRIPT)

        assert not _permit_frames(runtime)
        assert not runtime._mediated_ops.pending_ops
        # The worker drops the captured request it can no longer egress.
        assert _reap_frames(runtime) == [
            {"agent_task_id": writer, "call_correlation": "m0"}
        ]
        record = runtime._tasks[writer]
        assert record.status == TaskStatus.FAILED
        assert record.error == "agent boundary failed: model credential unavailable"

    asyncio.run(run())


def test_a_gone_vaulted_key_denies_the_held_model_turn() -> None:
    async def run() -> None:
        vault = _StubVault()
        runtime = _runtime(vault, config=_DEFAULT_URL_CONFIG)
        _, ids = await _register(runtime, _GONE_KEY_WF)
        writer = ids["writer"]
        vault.expire_all()

        _hold_dispatch(runtime, writer)
        _propose_held_turn(runtime, writer, "d")

        assert not _permit_frames(runtime)
        (deny,) = _deny_frames(runtime)
        assert deny["reason"] == "model credential unavailable"

    asyncio.run(run())


def _failing_script(error: str) -> list[ScriptedStep]:
    return [ScriptedStep(op="fail", error=error)]


@pytest.mark.parametrize(
    "error", ["boom", "held model egress rejected: model credential unavailable"]
)
def test_a_failed_episode_shows_its_reason_on_the_task(error: str) -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]

        _dispatch_agent(runtime, writer, script=_failing_script(error))

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.FAILED
        assert record.error == error

    asyncio.run(run())


_DENIED_SEARCH_STEP = ScriptedStep(
    op="boundary",
    kind=BoundaryEventKind.INVOCATION,
    call="s0",
    interface=SEARCH_INTERFACE,
    payload='{"query": "q", "max_results": 1}',
)


def test_a_gone_key_after_a_denied_boundary_shows_the_credential_reason() -> None:
    async def run() -> None:
        vault = _StubVault()
        runtime = _runtime(vault, config=_DEFAULT_URL_CONFIG)
        _, ids = await _register(runtime, _GONE_KEY_WF)
        writer = ids["writer"]
        script = [_DENIED_SEARCH_STEP, *_MODEL_SCRIPT]

        _dispatch_agent(runtime, writer, script=script)
        vault.expire_all()
        _dispatch_agent(runtime, writer, script=script)

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.FAILED
        assert record.error == "agent boundary failed: model credential unavailable"

    asyncio.run(run())


def test_a_failure_after_a_denied_boundary_shows_its_own_reason() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]
        script = [_DENIED_SEARCH_STEP, *_failing_script("boom")]

        _dispatch_agent(runtime, writer, script=script)
        _dispatch_agent(runtime, writer, script=script)

        record = runtime._tasks[writer]
        assert record.status == TaskStatus.FAILED
        assert record.error == "boom"

    asyncio.run(run())


def test_a_permit_that_cannot_be_minted_reaps_the_captured_request() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]
        engine = runtime.orchestration_engine(runtime._tasks[writer].workflow_id)
        assert engine is not None
        setattr(engine, "mint_operation_permit", lambda *a, **k: None)

        _dispatch_agent(runtime, writer, script=_MODEL_SCRIPT)

        assert not _permit_frames(runtime)
        assert _reap_frames(runtime) == [
            {"agent_task_id": writer, "call_correlation": "m0"}
        ]
        assert runtime._tasks[writer].error == (
            "agent boundary failed: could not mint a permit"
        )

    asyncio.run(run())


def _drain(runtime: TaskRuntime, worker: str = "wkr-1") -> None:
    _monitor(runtime)._handle_worker_event(
        WorkerEvent(type="UNREGISTER", worker_id=worker, graceful=True)
    )


def _report(runtime: TaskRuntime, agent: str, call: str, value: str) -> None:
    """The origin worker's outcome for its one outstanding permit."""
    permit = MediatedOperationPermit.model_validate(_permit_frames(runtime)[-1])
    runtime.settle_mediated_operation(
        MediatedOperationOutcome(
            permit_id=permit.permit_id,
            agent_task_id=agent,
            call_correlation=call,
            invocation_id=permit.invocation_id,
            idempotency_key=permit.idempotency_key,
            outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value=value),
        )
    )


def _placement_failures(runtime: TaskRuntime, task_id: str) -> list[str]:
    """What the dispatcher fails the task with once its worker has left."""
    registry = MagicMock()
    registry.idle_satisfying_pool.return_value = []
    registry.get_worker.return_value = None
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("wo-test"),
        no_worker_grace_sec=0,
    )
    dispatcher.dispatch_once(task_id)
    return [message for _, message, _ in dispatcher.failed]


@pytest.mark.parametrize("interface", ["search", "model"])
def test_an_agent_whose_drained_worker_finished_its_call_fails_at_its_next_step(
    interface: str, tmp_path: Path
) -> None:
    workflow, script = (
        (_SEARCH_WF, _SCRIPT) if interface == "search" else (_MODEL_WF, _MODEL_SCRIPT)
    )

    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, workflow)
        writer = ids["writer"]
        engine = _dispatch_agent(runtime, writer, script=script, seal_in=tmp_path)
        assert runtime.private_state_owner(writer) == _HOLDER

        # The draining worker reports the outcome, and unregisters once control
        # acknowledges it.
        _report(runtime, writer, "m0", "sunny")
        assert len(_reap_frames(runtime)) == 1
        _drain(runtime)

        record = runtime.get_record(writer)
        assert record is not None and record.status is TaskStatus.PENDING
        _, outcomes = engine.episode_context(writer)
        assert [o.value for o in outcomes] == [
            ToolOutcome(
                status=ToolOutcomeStatus.SUCCESS, value="sunny"
            ).model_dump_json()
        ]
        # The agent's next step can run only where its private state is.
        (failure,) = _placement_failures(runtime, writer)
        assert PrivateStateUnavailableReason.OWNER_LOST.value in failure

    asyncio.run(run())


@pytest.mark.parametrize("how", ["expired", "drained"])
def test_an_agent_suspended_on_a_search_group_through_its_worker_leaving(
    how: str,
) -> None:
    async def run() -> None:
        runtime = _runtime()
        workflow_id, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)
        first = _search_group(writer, 0, "sha-a").members[0]
        second = first.model_copy(
            update={"ordinal": 1, "call_correlation": f"{writer}:0:1"}
        )
        group = _search_group(writer, 0, "sha-a").model_copy(
            update={"members": (first, second)}
        )
        runtime.receive_worker_facade_group(writer, group)
        completion = HarnessResult(
            kind=HarnessResultKind.COMPLETION,
            value="done",
            capsule=HarnessCapsule(
                backend=HarnessBackendKey(backend="scripted", version="v1"), blob="c"
            ),
        )
        runtime.mark_succeeded(
            writer, "wkr-1", {"agent_episode": completion.model_dump(mode="json")}, _TS
        )
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        if how == "expired":
            runtime.recover_tasks_for_worker("wkr-1", spend_attempt=True)
            record = runtime.get_record(writer)
            assert record is not None and record.status is TaskStatus.FAILED
            return
        for permit in list(_permit_frames(runtime)):
            outcome = MediatedOperationPermit.model_validate(permit)
            runtime.settle_mediated_operation(
                MediatedOperationOutcome(
                    permit_id=outcome.permit_id,
                    agent_task_id=writer,
                    call_correlation=outcome.call_correlation,
                    invocation_id=outcome.invocation_id,
                    idempotency_key=outcome.idempotency_key,
                    outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="hit"),
                )
            )
        assert len(_reap_frames(runtime)) == 2
        _drain(runtime)

        record = runtime.get_record(writer)
        assert record is not None and record.status is TaskStatus.PENDING
        _, outcomes = engine.episode_context(writer)
        assert len(outcomes) == 2

    asyncio.run(run())


def test_a_boundary_a_drained_worker_could_not_finish_fails_the_agent() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _dispatch_agent(runtime, writer)

        _drain(runtime)

        record = runtime.get_record(writer)
        assert record is not None and record.status is TaskStatus.FAILED
        assert not runtime._mediated_ops.pending_ops

    asyncio.run(run())


@pytest.mark.parametrize("interface", ["search", "model"])
def test_a_boundary_whose_settle_a_crash_cut_short_reaches_its_origin_again(
    interface: str, tmp_path: Path
) -> None:
    workflow, script = (
        (_SEARCH_WF, _SCRIPT) if interface == "search" else (_MODEL_WF, _MODEL_SCRIPT)
    )

    async def run() -> None:
        registry = FakeRegistry()

        def runtime_on(store: FakeRegistry) -> TaskRuntime:
            return TaskRuntime(
                cast(Any, store),
                cast(Any, _WorkerStub()),
                OrchestrationConfig(),
                make_result_reader(),
                logging.getLogger("wo-test"),
                credential_vault=InMemoryCredentialVault(),
            )

        runtime = runtime_on(registry)
        _, ids = await _register(runtime, workflow)
        writer = ids["writer"]
        _dispatch_agent(runtime, writer, script=script, seal_in=tmp_path)
        permit = MediatedOperationPermit.model_validate(_permit_frames(runtime)[0])
        save = registry.save_ledger

        def crash(*_: Any, **__: Any) -> None:
            raise ConnectionError("crash before the ledger save")

        registry.save_ledger = crash  # type: ignore[method-assign]
        _report(runtime, writer, "m0", "sunny")
        runtime.shutdown()
        registry.save_ledger = save  # type: ignore[method-assign]

        restored = runtime_on(registry)
        await restored.rehydrate()

        record = restored.get_record(writer)
        assert record is not None and record.status is TaskStatus.DISPATCHED
        assert record.assigned_worker == "wkr-1"
        # Its worker gets the grace a restart gives the workers it finds in flight.
        assert restored.has_rehydrated_in_flight("wkr-1", 60.0)
        frames = cast(Any, restored._worker_registry).frames
        reissued = [
            (target, MediatedOperationPermit.model_validate(payload))
            for target, kind, payload in frames
            if kind == "permit"
        ]
        assert [(target, p.idempotency_key) for target, p in reissued] == [
            ("wkr-1", permit.idempotency_key)
        ]
        _report(restored, writer, "m0", "sunny")
        dispatch = restored.agent_episode_dispatch(writer, _HOLDER)
        assert dispatch is not None
        assert [o.value for o in dispatch.delivered_outcomes] == [
            ToolOutcome(
                status=ToolOutcomeStatus.SUCCESS, value="sunny"
            ).model_dump_json()
        ]

    asyncio.run(run())


def _frames(runtime: TaskRuntime, kind: str) -> list[tuple[str, dict[str, Any]]]:
    frames = cast(Any, runtime._worker_registry).frames
    return [(target, payload) for target, k, payload in frames if k == kind]


_SEARCH_S0 = [
    ScriptedStep(
        op="boundary",
        kind=BoundaryEventKind.INVOCATION,
        call="s0",
        interface=SEARCH_INTERFACE,
        payload=_PAYLOAD,
    ),
    ScriptedStep(op="complete", value_from="s0"),
]


def test_a_denied_boundary_reaps_the_request_its_worker_captured(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]

        _dispatch_agent(runtime, writer, script=_SEARCH_S0, seal_in=tmp_path)

        assert _permit_frames(runtime) == []
        assert _frames(runtime, "reap") == [
            ("wkr-1", {"agent_task_id": writer, "call_correlation": "s0"})
        ]

    asyncio.run(run())


def test_a_search_past_the_turn_cap_reaps_the_request_its_worker_captured() -> None:
    async def run() -> None:
        config = OrchestrationConfig()
        config.web_search = replace(config.web_search, max_parallel=1)
        runtime = _runtime(config=config)
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)
        first = _search_group(writer, 0, "sha-a").members[0]
        second = first.model_copy(
            update={"ordinal": 1, "call_correlation": f"{writer}:0:1"}
        )
        group = _search_group(writer, 0, "sha-a").model_copy(
            update={"members": (first, second)}
        )
        runtime.receive_worker_facade_group(writer, group)
        completion = HarnessResult(
            kind=HarnessResultKind.COMPLETION,
            value="done",
            capsule=HarnessCapsule(
                backend=HarnessBackendKey(backend="scripted", version="v1"), blob="c"
            ),
        )

        runtime.mark_succeeded(
            writer, "wkr-1", {"agent_episode": completion.model_dump(mode="json")}, _TS
        )

        assert len(_permit_frames(runtime)) == 1
        assert _frames(runtime, "reap") == [
            ("wkr-1", {"agent_task_id": writer, "call_correlation": f"{writer}:0:1"})
        ]

    asyncio.run(run())


def test_a_stale_step_reaps_the_request_its_worker_captured() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer, worker="wkr-2")
        stale = AgentEpisodeExecutor._capture_local_request(
            PendingEgressRequestStore(),
            writer,
            ScriptedHarnessAdapter(_SCRIPT, "v1").start(
                writer, capsule=None, outcomes=[]
            ),
            None,
            None,
        )

        runtime.mark_succeeded(
            writer, "wkr-1", {"agent_episode": stale.model_dump(mode="json")}, _TS
        )

        assert _permit_frames(runtime) == []
        assert _frames(runtime, "reap") == [
            ("wkr-1", {"agent_task_id": writer, "call_correlation": "m0"})
        ]

    asyncio.run(run())


_RESIDENT_SEARCH_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: resident-search-agent}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model, search/v1], delegate: []}
            tools: [{name: model}, {name: web_search, interface: "search/v1"}]
          harness: {backend: scripted, version: v1, params: {script: []}}
          model_binding: {mode: resident, service_model_ref: Qwen/Qwen3-4B}
"""


@pytest.mark.parametrize("failed", [False, True])
def test_a_resident_bound_agent_s_search_is_reaped_when_it_settles(
    failed: bool, tmp_path: Path
) -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _RESIDENT_SEARCH_WF)
        writer = ids["writer"]
        engine = _dispatch_agent(runtime, writer, script=_SEARCH_S0, seal_in=tmp_path)
        assert engine.service_dependency(writer) is not None
        assert _egress(runtime).occurrences() == [(writer, "s0")]
        permit = MediatedOperationPermit.model_validate(_permit_frames(runtime)[0])

        runtime.settle_mediated_operation(
            MediatedOperationOutcome(
                permit_id=permit.permit_id,
                agent_task_id=writer,
                call_correlation="s0",
                invocation_id=permit.invocation_id,
                idempotency_key=permit.idempotency_key,
                error="upstream unavailable" if failed else None,
                outcome=(
                    None
                    if failed
                    else ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="sunny")
                ),
            )
        )

        assert _egress(runtime).occurrences() == []

    asyncio.run(run())


def test_a_stale_step_leaves_what_a_new_dispatch_to_its_worker_captured() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _, stale = _run_agent_step(runtime, writer)
        runtime._tasks[writer].dispatch_id = "dsp-1"
        runtime.return_dispatch(writer, "wkr-1", increment_retry=False, front=True)
        # The next dispatch goes to the same worker, which captures the same call anew.
        worker = cast(Worker, SimpleNamespace(id="wkr-1", node_id="nde-1"))
        assert runtime.begin_publish(writer, worker, "dsp-2") is PublishGate.PUBLISH

        runtime.mark_succeeded(writer, "wkr-1", stale, _TS, dispatch_id="dsp-1")

        assert _frames(runtime, "reap") == []

    asyncio.run(run())


def _stash_search_group(runtime: TaskRuntime, writer: str) -> FacadeTurnGroup:
    """A two-member search group a held turn captured, its requests in the worker's
    egress store as the facade stashes them."""
    first = _search_group(writer, 0, "sha-a").members[0]
    second = first.model_copy(
        update={"ordinal": 1, "call_correlation": f"{writer}:0:1"}
    )
    group = _search_group(writer, 0, "sha-a").model_copy(
        update={"members": (first, second)}
    )
    for member in group.members:
        _egress(runtime).put(
            writer, member.call_correlation, parse_search_request(_PAYLOAD), None
        )
    return group


def test_a_step_landing_on_a_cancel_reaps_the_request_its_worker_captured(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        runtime = _runtime()
        workflow_id, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _, payload = _run_agent_step(runtime, writer, seal_in=tmp_path)
        assert _egress(runtime).occurrences() == [(writer, "m0")]
        runtime.cancel_workflow(workflow_id)

        runtime.mark_succeeded(writer, "wkr-1", payload, _TS)

        record = runtime.get_record(writer)
        assert record is not None and record.status is TaskStatus.CANCELLED
        assert _egress(runtime).occurrences() == []

    asyncio.run(run())


def test_a_completion_racing_a_cancel_reaps_its_facade_group() -> None:
    async def run() -> None:
        runtime = _runtime()
        workflow_id, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)
        group = _stash_search_group(runtime, writer)
        runtime.cancel_workflow(workflow_id)
        completion = HarnessResult(
            kind=HarnessResultKind.COMPLETION,
            value="done",
            capsule=HarnessCapsule(
                backend=HarnessBackendKey(backend="scripted", version="v1"), blob="c"
            ),
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

        record = runtime.get_record(writer)
        assert record is not None and record.status is TaskStatus.CANCELLED
        assert _egress(runtime).occurrences() == []

    asyncio.run(run())


def test_a_late_step_of_a_dispatch_settled_cancelled_reaps_and_routes_nothing(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        runtime = _runtime()
        workflow_id, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _, payload = _run_agent_step(runtime, writer, seal_in=tmp_path)
        record = runtime._tasks[writer]
        record.dispatch_id = "dsp-1"
        runtime.cancel_workflow(workflow_id)
        runtime.resolve_disowned_dispatch(writer, "dsp-1", "wkr-1", 0.0)
        assert record.status is TaskStatus.CANCELLED

        runtime.mark_succeeded(writer, "wkr-1", payload, _TS, dispatch_id="dsp-1")

        assert record.status is TaskStatus.CANCELLED
        assert _permit_frames(runtime) == []
        assert _egress(runtime).occurrences() == []

    asyncio.run(run())


def test_a_late_completion_of_a_dispatch_settled_cancelled_routes_no_group() -> None:
    async def run() -> None:
        runtime = _runtime()
        workflow_id, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)
        record = runtime._tasks[writer]
        record.dispatch_id = "dsp-1"
        group = _stash_search_group(runtime, writer)
        runtime.cancel_workflow(workflow_id)
        runtime.resolve_disowned_dispatch(writer, "dsp-1", "wkr-1", 0.0)
        completion = HarnessResult(
            kind=HarnessResultKind.COMPLETION,
            value="done",
            capsule=HarnessCapsule(
                backend=HarnessBackendKey(backend="scripted", version="v1"), blob="c"
            ),
        )

        runtime.mark_succeeded(
            writer,
            "wkr-1",
            {
                "agent_episode": completion.model_dump(mode="json"),
                "agent_episode_facade_group": group.model_dump(mode="json"),
            },
            _TS,
            dispatch_id="dsp-1",
        )

        assert record.status is TaskStatus.CANCELLED
        assert record.pending_facade_group is None
        assert _permit_frames(runtime) == []
        assert _egress(runtime).occurrences() == []

    asyncio.run(run())


@pytest.mark.parametrize("carried", [True, False])
@pytest.mark.parametrize(
    "kind", [HarnessResultKind.FAILURE, HarnessResultKind.CANCELLATION]
)
def test_a_turn_that_fails_reaps_its_facade_group(
    kind: HarnessResultKind, carried: bool
) -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)
        group = _stash_search_group(runtime, writer)
        payload: dict[str, Any] = {
            "agent_episode": HarnessResult(kind=kind, error="turn failed").model_dump(
                mode="json"
            )
        }
        if carried:
            payload["agent_episode_facade_group"] = group.model_dump(mode="json")
        else:
            runtime.receive_worker_facade_group(writer, group)

        runtime.mark_succeeded(writer, "wkr-1", payload, _TS)

        assert _permit_frames(runtime) == []
        assert _egress(runtime).occurrences() == []

    asyncio.run(run())
