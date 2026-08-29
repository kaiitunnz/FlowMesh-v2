"""The scripted backend drives a real agent episode through the engine boundary path.

This is the cross-check that the scripted backend is a legitimate binding, not an engine
bypass: an agent suspends on a mediated model boundary and releases its lane, resumes
with the injected outcome, spawns one child in a declared region and seals it, and
completes with the injected value — every side effect flowing through the engine's
validation. It also proves the episode replays from the durable capsule after a restart.
"""

from server.orchestration import ProgressAxis, WorkItemStatus
from server.orchestration.harness import (
    AgentEpisode,
    DeliveredOutcome,
    HarnessResultKind,
)
from server.orchestration.state import InvocationState
from shared.harness import BoundaryEventKind
from tests.server.task.test_v2_agent_harness import (
    _dispatch_agent,
    _engine,
    _leaf,
    _spawning_agent,
)
from worker.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_SCRIPT = [
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.INVOCATION, call="c0", interface="model"
    ),
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN, call="c1", region="worker"
    ),
    ScriptedStep(
        op="boundary", kind=BoundaryEventKind.SPAWN_SEAL, call="c2", region="worker"
    ),
    ScriptedStep(op="complete", value_from="c0"),
]


def _adapter() -> ScriptedHarnessAdapter:
    return ScriptedHarnessAdapter(_SCRIPT, "v1")


def test_scripted_episode_runs_the_full_lifecycle_through_the_engine() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)
    ep = AgentEpisode(eng, _adapter())

    # 1. The first step defers the model boundary; the agent suspends and releases its
    # lane until a durable outcome arrives.
    ep.resume("A", act)
    wi = eng.work_item("A")
    assert wi is not None and wi.status is WorkItemStatus.BLOCKED
    env = eng.boundary_envelope(act, "c0")
    assert env is not None and env.idempotency_key is not None
    assert env.invocation_id is not None
    assert eng._invocations[env.invocation_id].state is InvocationState.ISSUED  # type: ignore[attr-defined]

    # 2. Only the durable outcome re-readies the lane; the injected value threads back.
    assert eng.deliver_boundary_outcome("A", "c0").ready == ["A"]
    ep.deliver(DeliveredOutcome(call_correlation="c0", value="ANSWER"))
    eng.on_dispatched("A", "w1")
    spawn = ep.resume("A", act)  # injects ANSWER, defers the spawn
    assert spawn.request is not None and spawn.request.kind is BoundaryEventKind.SPAWN
    children = [
        wi.legacy_task_id
        for wi in eng.to_snapshot().work_items
        if wi.legacy_task_id.startswith("act-")
    ]
    assert len(children) == 1
    child = children[0]

    # 3. The spawn seal closes the region; the completion carries the injected value.
    ep.resume("A", act)  # spawn seal
    region_scope = eng.region_scope_for(act, "worker")
    cap = eng.capability(region_scope, ProgressAxis.CHILD_INIT)
    assert cap is not None and cap.status.value == "sealed"
    done = ep.resume("A", act)
    assert done.kind is HarnessResultKind.COMPLETION and done.value == "ANSWER"

    # 4. The workflow settles only after the child settles — not on the agent alone.
    eng.on_dispatched(child, "w1")
    eng.on_succeeded(child)
    eng.on_succeeded("A")
    closed_cap = eng.capability(region_scope, ProgressAxis.CHILD_INIT)
    assert closed_cap is not None and closed_cap.closed
    pub = eng.resolve_output("out:A")
    assert pub is not None and pub.outcome.value == "success"


def test_scripted_episode_replays_from_the_capsule_after_a_restart() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    act = _dispatch_agent(eng)

    # Drive the model boundary, then throw the in-memory episode away and rebuild the
    # ledger from its snapshot — the capsule and the settled outcome are durable.
    ep = AgentEpisode(eng, _adapter())
    ep.resume("A", act)
    capsule_blob = eng.boundary_envelope(act, "c0").continuation  # type: ignore[union-attr]
    assert capsule_blob is not None

    from server.orchestration.engine import OrchestrationEngine

    restored = OrchestrationEngine(
        eng.to_snapshot(), _spawning_agent(child=_leaf("child"))
    )
    assert restored.deliver_boundary_outcome("A", "c0").ready == ["A"]

    # A fresh adapter resumes from the durable capsule and the injected outcome alone.
    from shared.harness import HarnessBackendKey, HarnessCapsule

    fresh = _adapter()
    step = fresh.start(
        act,
        capsule=HarnessCapsule(
            backend=HarnessBackendKey(backend="scripted", version="v1"),
            blob=capsule_blob,
        ),
        outcomes=[DeliveredOutcome(call_correlation="c0", value="ANSWER")],
    )
    assert step.request is not None and step.request.kind is BoundaryEventKind.SPAWN
