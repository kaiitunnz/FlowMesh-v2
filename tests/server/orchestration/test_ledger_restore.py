"""A stored ledger restores every fact it captured, and the region core settles nested
work in one transition."""

from server.orchestration import LedgerSnapshot, OrchestrationEngine
from server.orchestration.state import BoundaryEvent, PublicationOutcome, WorkItemStatus
from server.task.v2.compiler.bindings import leaf_profile
from server.task.v2.representations.operators import (
    AgentOperator,
    AuthorityCeiling,
    BindingKey,
    BoundaryEventKind,
    ChildRegionRef,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    Port,
    ResidualPolicy,
    SpawnRegion,
)
from server.task.v2.representations.template import TemplateEdge
from shared.tasks import TaskType
from tests.server.orchestration.helpers import (
    _SIGNATURE,
    _bundle,
    _decl,
    chain_bundle,
    engine,
    rehydrated,
)


def _failed_chain() -> OrchestrationEngine:
    eng = engine(chain_bundle())
    eng.on_dispatched("A", "w1")
    eng.on_failed("A", "boom", retryable=False)
    return eng


def test_a_snapshot_captures_every_ledger_field() -> None:
    snapshot = _failed_chain().to_snapshot()

    assert snapshot.model_fields_set == set(LedgerSnapshot.model_fields)


def test_a_restored_ledger_captures_the_same_snapshot() -> None:
    live = _failed_chain()

    restored = rehydrated(live)

    assert (
        restored.to_snapshot().model_dump_json() == live.to_snapshot().model_dump_json()
    )


def test_a_ledger_stored_without_failure_reasons_names_each_failed_work_item() -> None:
    live = _failed_chain()
    stored = live.to_snapshot().model_copy(update={"failure_reasons": {}})

    restored = OrchestrationEngine(stored, live._topology.bundle)

    assert live.declared_failures() == {"A": "boom", "B": "Dependency A failed"}
    assert restored.declared_failures() == {
        "A": "boom",
        "B": "declared-failure obligation",
    }


def test_a_stored_failure_reason_wins_over_the_work_items_own() -> None:
    live = _failed_chain()
    stored = live.to_snapshot().model_copy(update={"failure_reasons": {"A": "kept"}})

    restored = OrchestrationEngine(stored, live._topology.bundle)

    assert restored.failure_reason("A") == "kept"
    assert restored.failure_reason("B") == "declared-failure obligation"


def _recursive_any_cancel() -> OrchestrationEngine:
    """Agent ``A`` whose ``worker`` region spawns ``A`` again under an ``any`` join that
    cancels its residual children, feeding external-effect leaf ``D`` that the root
    grant does not cover."""
    spawn = SpawnRegion(
        operator_id="worker:spawn",
        source_ref="worker:spawn",
        outputs=(Port(name="children"),),
        child_template_ref="A",
    )
    join = JoinRegion(
        operator_id="worker:spawn:join",
        source_ref="worker:spawn:join",
        inputs=(Port(name="children"),),
        outputs=(Port(name="out"),),
        completion=JoinCompletion.ANY,
        residual_policy=ResidualPolicy.CANCEL,
    )
    agent = AgentOperator(
        operator_id="A",
        source_ref="A",
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(invoke=("model",), delegate=()),
        boundary=_SIGNATURE,
        child_region_refs=(ChildRegionRef(name="worker", spawn_ref="worker:spawn"),),
        outputs=(Port(name="out"),),
    )
    effect = LeafOperator(
        operator_id="D",
        source_ref="D",
        outputs=(Port(name="out"),),
        profile=leaf_profile(TaskType.SSH),
    )
    bundle = _bundle(
        [agent, spawn, join, effect],
        [
            TemplateEdge(from_op="worker:spawn", to_op="worker:spawn:join"),
            TemplateEdge(from_op="worker:spawn:join", to_op="D"),
        ],
        (_decl("out:A", "A"), _decl("out:D", "D")),
    )
    return engine(bundle)


def _spawn(eng: OrchestrationEngine, parent: str, call: str) -> str:
    """Spawn one ``worker`` child under ``parent`` and return its task id."""
    before = set(eng._ledger.activations)
    eng.route_boundary_event(
        parent,
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation=call,
            child_region_ref="worker",
        ),
    )
    (child,) = (
        activation
        for activation_id, activation in eng._ledger.activations.items()
        if activation_id not in before and activation.kind == "child"
    )
    wi = eng._ledger.work_items[eng._ledger.wi_by_activation[child.activation_id]]
    return wi.legacy_task_id


def test_a_join_release_cancels_a_nested_subtree_and_fails_a_denied_admission() -> None:
    eng = _recursive_any_cancel()
    eng.on_dispatched("A", "w1")
    first = _spawn(eng, "A", "s1")
    eng.on_dispatched(first, "w2")
    grandchild = _spawn(eng, first, "g1")
    second = _spawn(eng, "A", "s2")
    eng.on_dispatched(second, "w3")
    eng.on_started(second)

    advance = eng.on_succeeded(second)

    # The first child and the grandchild in the region it entered are cancelled as
    # residual, and the released join's record reaches D, whose admission is denied.
    assert set(advance.cancelled) == {first, grandchild}
    assert advance.failed == ["D"]
    assert advance.ready == []
    for task_id in (first, grandchild):
        wi = eng.work_item(task_id)
        assert wi is not None and wi.status is WorkItemStatus.CANCELLED
    denied = eng.work_item("D")
    assert denied is not None and denied.outcome is PublicationOutcome.DECLARED_FAILURE
    assert eng.failure_reason("D") is not None
    kinds = [event.kind for event in eng._ledger.trace]
    assert kinds.index("join_released") < kinds.index("scope_cancelled")
    assert kinds.index("scope_cancelled") < kinds.index("authority_denied")
