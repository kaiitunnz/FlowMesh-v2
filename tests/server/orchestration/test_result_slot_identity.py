"""A result slot's identity is unambiguous and carries its scope.

A keyed collection publishes one slot per child of each scope that spawns, so two
scopes publishing the same child index are two slots.
"""

from server.orchestration import OrchestrationEngine, PublicationOutcome
from server.orchestration.state import Activation, ValueRef
from server.task.outputs import published_members
from server.task.v2 import PersistedV2Workflow
from server.task.v2.representations.results import Visibility
from shared.content import reference_for
from tests.server.task.test_v2_episode_dispatch import _engine, _fanout_bundle


def _child(scope_id: str, index: int) -> Activation:
    return Activation(
        activation_id=f"act-{scope_id}-{index}",
        instance_id="wfl-x",
        scope_id=scope_id,
        operator_id="trial",
        kind="child",
        child_index=index,
    )


def _value(label: str) -> ValueRef:
    return ValueRef(
        kind="legacy_task_result",
        content=reference_for("org", label.encode(), media_type="application/json"),
    )


def test_two_scopes_publishing_one_index_are_two_members() -> None:
    eng = _engine()
    for scope_id in ("scp-a", "scp-b"):
        eng._publication.publish_keyed(
            "exp", _child(scope_id, 0), PublicationOutcome.SUCCESS, _value(scope_id)
        )

    a = eng.output_publication("results", "scp-a", "0")
    b = eng.output_publication("results", "scp-b", "0")
    assert a is not None and b is not None
    assert a.value_ref == _value("scp-a") and b.value_ref == _value("scp-b")
    assert {slot.scope_id for slot in eng.output_slots("results")} == {"scp-a", "scp-b"}

    restored = OrchestrationEngine(eng.to_snapshot(), _published_bundle())
    assert restored.output_publication("results", "scp-b", "0") == b
    members = [m for m in published_members(restored) if m.name == "exp"]
    assert [(m.scope_id, m.key) for m in members] == [("scp-a", "0"), ("scp-b", "0")]
    assert members[0].cursor != members[1].cursor
    assert [m.publication for m in members] == [a, b]


def _published_bundle() -> PersistedV2Workflow:
    bundle = _fanout_bundle()
    template = bundle.template
    return bundle.model_copy(
        update={
            "template": template.model_copy(
                update={
                    "result_declarations": tuple(
                        decl.model_copy(update={"visibility": Visibility.PUBLISHED})
                        for decl in template.result_declarations
                    )
                }
            )
        }
    )


def test_a_key_that_reads_like_a_sequence_is_its_own_slot() -> None:
    eng = _engine()
    snapshot = eng.to_snapshot()
    keys = {
        slot.slot_key
        for slot in (
            snapshot.result_slots[0].model_copy(
                update={"logical_key": "1#2", "sequence": None}
            ),
            snapshot.result_slots[0].model_copy(
                update={"logical_key": "1", "sequence": 2}
            ),
            snapshot.result_slots[0].model_copy(update={"logical_key": ""}),
            snapshot.result_slots[0].model_copy(update={"logical_key": None}),
        )
    }
    assert len(keys) == 4
