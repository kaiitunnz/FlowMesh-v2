"""The server routes agent-episode boundaries and re-dispatches through the runtime.

Driving the real runtime ingest with the real scripted adapter, an agent spawns a child
in a declared region, seals it, and completes — the workflow settling only after the
child settles. This is the server-side half of the worker seam: the dispatcher hands the
episode its durable context, and each boundary the "worker" returns routes into the
ledger and re-readies or suspends the lane.
"""

import asyncio
from typing import Any

from server.orchestration import ProgressAxis, WorkItemStatus
from server.task.models import TaskStatus
from shared.harness import (
    BoundaryEventKind,
    BoundaryRequest,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _runtime
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_TS = "2026-08-29T00:00:00Z"

_AGENT_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: agent-episode}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: reviewer
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: reviewer
        spec: {taskType: echo, data: {type: list, items: [placeholder]}}
"""

_SCRIPT = [
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN, call="c0", region="reviewer"
    ),
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN_SEAL, call="c1", region="reviewer"
    ),
    ScriptedStep(op="complete", value="reviewed"),
]


def _adapter() -> ScriptedHarnessAdapter:
    return ScriptedHarnessAdapter(_SCRIPT, "v1")


def _step(runtime, adapter, task_id: str, worker: str = "wkr-1") -> None:
    """Simulate one dispatch: give the episode its context, run and report it."""
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id)
    assert engine is not None and dispatch is not None
    capsule = (
        HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
        if dispatch.capsule_blob is not None
        else None
    )
    engine.on_dispatched(task_id, worker)
    result = adapter.start(
        task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes
    )
    runtime.mark_succeeded(
        task_id, worker, {"agent_episode": result.model_dump(mode="json")}, _TS
    )


def test_agent_episode_spawns_seals_and_settles_after_the_child() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = _adapter()

        # Step 1: the agent defers a spawn; a child materializes, the agent re-readies.
        _step(runtime, adapter, writer)
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None
        snap = engine.to_snapshot()
        children = [
            wi.legacy_task_id
            for wi in snap.work_items
            if wi.legacy_task_id.startswith("act-")
        ]
        assert len(children) == 1
        child = children[0]
        assert runtime._tasks[writer].status is not None  # re-enqueued, not terminal

        # Step 2: the agent seals the region; Step 3: it completes.
        _step(runtime, adapter, writer)  # spawn seal
        writer_act = engine.work_item(writer)
        assert writer_act is not None
        region_scope = engine.region_scope_for(writer_act.activation_id, "reviewer")
        cap = engine.capability(region_scope, ProgressAxis.CHILD_INIT)
        assert cap is not None and cap.status.value == "sealed"
        _step(runtime, adapter, writer)  # completion → terminal

        # Premature-completion guard: the agent finished, but the child is still in
        # flight, so the region has not closed and its scope is not released.
        assert child not in engine.to_snapshot().released_scopes
        child_cap = engine.capability(region_scope, ProgressAxis.CHILD_INIT)
        assert child_cap is not None and not child_cap.closed

        # The child settles: only now does the region drain and the workflow complete.
        engine.on_dispatched(child, "wkr-1")
        runtime.mark_succeeded(child, "wkr-1", {}, _TS)
        closed = engine.capability(region_scope, ProgressAxis.CHILD_INIT)
        assert closed is not None and closed.closed
        writer_wi = engine.work_item(writer)
        assert writer_wi is not None and writer_wi.status is WorkItemStatus.SETTLED
        pub = engine.resolve_output(f"legacy:{writer}")
        assert pub is not None and pub.outcome.value == "success"

    asyncio.run(run())


def test_model_boundary_settle_returns_the_record_to_pending() -> None:
    # The wait-boundary settle must re-ready the record, not only the work item, or the
    # dispatcher skips it (a ready task dispatches only from a pending record).
    async def run() -> None:
        runtime = _runtime(FakeRegistry())

        def _settle(task: str, call: str, payload: str | None) -> None:
            runtime.settle_episode_invocation(task, call, f"model:{payload}")

        runtime.set_invocation_settler(_settle)
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(
            [
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.INVOCATION,
                    call="m0",
                    interface="model",
                    payload="draft",
                ),
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.SPAWN,
                    call="c0",
                    region="reviewer",
                ),
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.SPAWN_SEAL,
                    call="c1",
                    region="reviewer",
                ),
                ScriptedStep(op="complete", value_from="m0"),
            ],
            "v1",
        )
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        _step(runtime, adapter, writer)  # model boundary → suspend → canned settle
        # The synchronous settle re-readied both the work item and the record.
        assert runtime._tasks[writer].status is TaskStatus.PENDING
        wi = engine.work_item(writer)
        assert wi is not None and wi.status is WorkItemStatus.READY

        _step(runtime, adapter, writer)  # injects the model result → spawns the child
        child = [
            w.legacy_task_id
            for w in engine.to_snapshot().work_items
            if w.legacy_task_id.startswith("act-")
        ]
        assert len(child) == 1
        _step(runtime, adapter, writer)  # spawn seal
        _step(runtime, adapter, writer)  # completion carries the injected model result
        engine.on_dispatched(child[0], "wkr-1")
        runtime.mark_succeeded(child[0], "wkr-1", {}, _TS)
        writer_wi = engine.work_item(writer)
        assert writer_wi is not None and writer_wi.status is WorkItemStatus.SETTLED
        pub = engine.resolve_output(f"legacy:{writer}")
        assert pub is not None and pub.outcome.value == "success"

    asyncio.run(run())


def _canned_settler(runtime):
    def _settle(task: str, call: str, payload: str | None) -> None:
        runtime.settle_episode_invocation(task, call, f"model:{payload}")

    return _settle


def test_a_consumed_outcome_is_not_reinjected_across_a_query_step() -> None:
    # A state-access step between an outcome-bearing boundary and a later step must not
    # re-ship the earlier outcome: pending_outcome_call is cleared as each step runs.
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        runtime.set_invocation_settler(_canned_settler(runtime))
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = ScriptedHarnessAdapter(
            [
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.INVOCATION,
                    call="m0",
                    interface="model",
                    payload="draft",
                ),
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.STATE_ACCESS,
                    call="s0",
                    payload="peek",
                ),
                ScriptedStep(op="complete", value_from="m0"),
            ],
            "v1",
        )
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        _step(runtime, adapter, writer)  # model boundary → suspend → canned settle
        after_model = runtime.agent_episode_dispatch(writer)
        assert after_model is not None and len(after_model.delivered_outcomes) == 1
        _step(runtime, adapter, writer)  # injects m0 → state-access (inline)
        # The model outcome was consumed by the previous dispatch; it must not re-ship.
        after_query = runtime.agent_episode_dispatch(writer)
        assert after_query is not None and after_query.delivered_outcomes == ()
        _step(runtime, adapter, writer)  # completion still reads the injected m0 value
        wi = engine.work_item(writer)
        assert wi is not None and wi.status is WorkItemStatus.SETTLED

    asyncio.run(run())


def test_a_denied_boundary_re_readies_with_a_denied_outcome() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        runtime.set_invocation_settler(_canned_settler(runtime))
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        # "danger" is outside the declared invoke face, so the boundary is denied.
        adapter = ScriptedHarnessAdapter(
            [
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.INVOCATION,
                    call="d0",
                    interface="danger",
                ),
                ScriptedStep(op="complete", value="ok"),
            ],
            "v1",
        )
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        _step(runtime, adapter, writer)  # denied boundary → re-ready with a denial
        assert runtime._tasks[writer].status is TaskStatus.PENDING
        dispatch = runtime.agent_episode_dispatch(writer)
        assert dispatch is not None and len(dispatch.delivered_outcomes) == 1
        assert dispatch.delivered_outcomes[0].kind is OutcomeKind.DENIED
        _step(runtime, adapter, writer)  # the agent handles the denial and completes
        wi = engine.work_item(writer)
        assert wi is not None and wi.status is WorkItemStatus.SETTLED

    asyncio.run(run())


class _AlwaysCompleteAdapter(HarnessAdapter):
    """A harness whose every turn completes cleanly, as a Codex turn does under G.

    Origination is not the adapter's job here: a facade is captured at the gateway,
    which clean-completes the turn, so the adapter only ever reports a completion.
    """

    def __init__(self, value: str = "clean") -> None:
        self._value = value

    def backend_key(self) -> HarnessBackendKey:
        return HarnessBackendKey(backend="scripted", version="v1")

    def start(
        self, activation_id: str, *, capsule: Any, outcomes: Any
    ) -> HarnessResult:
        return HarnessResult(
            kind=HarnessResultKind.COMPLETION,
            value=self._value,
            capsule=HarnessCapsule(backend=self.backend_key(), blob="{}"),
        )

    def cancel(self, activation_id: str) -> None:
        return None


def _complete(
    runtime: Any,
    adapter: HarnessAdapter,
    task_id: str,
    *,
    originate: BoundaryRequest | None = None,
    worker: str = "wkr-1",
) -> None:
    """Drive one clean-completing turn, optionally originating a facade first."""
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    dispatch = runtime.agent_episode_dispatch(task_id)
    capsule = (
        HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
        if dispatch.capsule_blob is not None
        else None
    )
    engine.on_dispatched(task_id, worker)
    result = adapter.start(
        task_id, capsule=capsule, outcomes=dispatch.delivered_outcomes
    )
    if originate is not None:
        runtime.originate_episode_boundary(task_id, originate)
    runtime.mark_succeeded(
        task_id, worker, {"agent_episode": result.model_dump(mode="json")}, _TS
    )


def _spawn_reviewer(writer: str) -> BoundaryRequest:
    return BoundaryRequest(
        kind=BoundaryEventKind.SPAWN,
        call_correlation=f"{writer}:0",
        child_region_ref="reviewer",
    )


def test_gateway_originated_spawn_reroutes_a_clean_completion() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = _AlwaysCompleteAdapter()
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        # Turn 1: the model called spawn_agent; the gateway captured it server-side and
        # clean-completed the turn. The guard reroutes the completion into the boundary
        # instead of settling the episode, and a child materializes.
        _complete(runtime, adapter, writer, originate=_spawn_reviewer(writer))
        assert runtime._tasks[writer].status is TaskStatus.PENDING
        children = [
            w.legacy_task_id
            for w in engine.to_snapshot().work_items
            if w.legacy_task_id.startswith("act-")
        ]
        assert len(children) == 1
        child = children[0]
        writer_act = engine.work_item(writer)
        assert writer_act is not None
        region_scope = engine.region_scope_for(writer_act.activation_id, "reviewer")

        # Turn 2: the next clean completion is terminal; it auto-seals child-init.
        _complete(runtime, adapter, writer)
        cap = engine.capability(region_scope, ProgressAxis.CHILD_INIT)
        assert cap is not None and cap.status.value == "sealed"
        # Premature-completion guard: the agent completed and sealed, but the region has
        # not drained while the child is still in flight, so its scope is not released.
        assert child not in engine.to_snapshot().released_scopes
        assert not cap.closed

        # The child settles: only now does the region drain and the workflow complete.
        engine.on_dispatched(child, "wkr-1")
        runtime.mark_succeeded(child, "wkr-1", {}, _TS)
        closed = engine.capability(region_scope, ProgressAxis.CHILD_INIT)
        assert closed is not None and closed.closed
        pub = engine.resolve_output(f"legacy:{writer}")
        assert pub is not None and pub.outcome.value == "success"

    asyncio.run(run())


def test_gateway_origination_redrive_creates_no_second_child() -> None:
    # A crash before the outcome injects re-dispatches the turn; the gateway re-captures
    # the same facade and re-originates the same stable correlation. The boundary
    # machinery maps it to the recorded envelope, so no second child is materialized.
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = _AlwaysCompleteAdapter()
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None

        _complete(runtime, adapter, writer, originate=_spawn_reviewer(writer))
        _complete(runtime, adapter, writer, originate=_spawn_reviewer(writer))
        children = [
            w.legacy_task_id
            for w in engine.to_snapshot().work_items
            if w.legacy_task_id.startswith("act-")
        ]
        assert len(children) == 1

    asyncio.run(run())


def test_agent_episode_dispatch_ships_the_capsule_and_outcome() -> None:
    async def run() -> None:
        runtime = _runtime(FakeRegistry())
        workflow_id, ids = await _register(runtime, _AGENT_WF)
        writer = ids["writer"]
        adapter = _adapter()

        # First dispatch carries no capsule; after the spawn the next carries the
        # advanced capsule and the spawn's ack outcome.
        first = runtime.agent_episode_dispatch(writer)
        assert first is not None and first.capsule_blob is None
        assert first.delivered_outcomes == ()
        _step(runtime, adapter, writer)
        second = runtime.agent_episode_dispatch(writer)
        assert second is not None and second.capsule_blob is not None
        assert len(second.delivered_outcomes) == 1
        assert second.delivered_outcomes[0].call_correlation == "c0"

    asyncio.run(run())
