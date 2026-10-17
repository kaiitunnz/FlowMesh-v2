"""The marking pass: every reference a stored binding names is marked in one scan of
the keyspace, an undecodable value leaves the scopes it may name incomplete, and the
pass yields often."""

import inspect
import json
import logging
import threading
from types import ModuleType
from typing import Any

import fakeredis
import pytest

import server.clients.redis as redis_module
import server.reclamation.marking as marking
from server.clients.redis import (
    content_finalization_key,
    task_artifacts_key,
    workflow_ds_key,
    workflow_history_key,
    workflow_sources_key,
)
from server.orchestration.ledger_layout import field_name, keyed_value
from server.orchestration.state import (
    PublicationOutcome,
    Record,
    ResultPublication,
    ValueMember,
    ValueRef,
)
from server.reclamation.marking import (
    KEY_FAMILIES,
    VALUES_PER_PAUSE,
    Family,
    Marker,
    MarkPass,
)
from server.registries.artifacts import artifact_field
from server.registries.task_states import PersistedTask
from server.registries.workflow import WorkflowRegistry, WorkflowSched
from server.task.history import (
    CERTIFICATE_FIELD,
    HistoryCertificate,
    OutputHistory,
    TaskHistory,
)
from server.task.models import TaskRecord, TaskStatus
from shared.content import ContentReference, reference_for
from shared.content.tree import TREE_MEDIA_TYPE
from shared.outcome import OutcomeManifest
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from shared.tasks.result_binding import ResultBinding, ResultMember
from tests.server.orchestration.helpers import WORKFLOW_ID, chain_bundle, engine
from tests.server.redis_helpers import fake_redis_client

_A = "org-a"
_B = "org-b"
_LOGGER = logging.getLogger("test.marking")

type Identity = tuple[str, str]


def _ref(
    data: bytes, scope: str = _A, media_type: str | None = None
) -> ContentReference:
    if media_type is None:
        return reference_for(scope, data)
    return reference_for(scope, data, media_type=media_type)


def _identity(reference: ContentReference) -> Identity:
    return reference.authorization_scope, reference.content_digest


class _Fixture:
    def __init__(self) -> None:
        server = fakeredis.FakeServer()
        self.rds = fake_redis_client(server)
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.marker = Marker(self.rds, _LOGGER, page_pause_sec=0, duty=1.0)

    def run(self, stop: threading.Event | None = None) -> MarkPass:
        return self.marker.run(stop or threading.Event())


@pytest.fixture
def fx() -> _Fixture:
    return _Fixture()


def _record(value: ContentReference) -> Record:
    return Record(
        operator_id="op",
        activation_id="act-1",
        scope_id="scp-1",
        value_ref=ValueRef(kind="legacy_task_result", content=value),
    )


def _ledger(fx: _Fixture, workflow_id: str, refs: list[ContentReference]) -> None:
    publication = ResultPublication(
        slot_key="slot-1",
        output_id="out",
        outcome=PublicationOutcome.SUCCESS,
        value_ref=ValueRef(
            kind="aggregate",
            members=(
                ValueMember(
                    key="0",
                    outcome=PublicationOutcome.SUCCESS,
                    value_ref=ValueRef(kind="legacy_task_result", content=refs[0]),
                ),
            ),
        ),
    )
    fx.raw.hset(
        workflow_ds_key(workflow_id),
        mapping={
            "meta:layout": "1",
            field_name("result_publications", "slot-1"): keyed_value(0, publication),
            "records:[0]": _record(refs[1]).model_dump_json(),
            "trace:[0]": "{not decoded: the trace binds nothing}",
        },
    )


def _task_record(
    task_id: str, result: ContentReference, scope: str, workflow_id: str = "wfl-tasks"
) -> TaskRecord:
    return TaskRecord.model_validate(
        {
            "task_id": task_id,
            "workflow_id": workflow_id,
            "owner_id": "owner",
            "org_id": scope,
            "raw_yaml": "source",
            "task": {
                "apiVersion": "flowmesh/v1",
                "kind": "Task",
                "spec": {"taskType": "echo"},
            },
            "status": TaskStatus.DISPATCHED,
            "internal": {"result_reference": result},
        }
    )


def _seed_everything(fx: _Fixture) -> tuple[set[Identity], set[Identity]]:
    """Store one binding of every family, as each writer stores it, and return the
    identities they name and the tree manifests among them."""
    ledger_refs = [_ref(b"ledger-member"), _ref(b"ledger-record", _B)]
    _ledger(fx, "wfl-1", ledger_refs)

    result = _ref(b"task-result")
    dispatched = ArtifactBinding(
        name="input", kind=ArtifactKind.FILE, reference=_ref(b"input"), size_bytes=5
    )
    record = _task_record("tsk-1", result, _A)
    record.internal.dispatched_artifacts = (dispatched,)
    registry = WorkflowRegistry(fx.rds)
    registry.save_task_states([PersistedTask(record=record)])

    blueprint_result = _ref(b"blueprint-result", _B)
    blueprint_tree = _ref(b"blueprint-tree", _B, TREE_MEDIA_TYPE)
    blueprint = _task_record("tsk-bp", blueprint_result, _B, "wfl-bp")
    blueprint.internal.dispatched_artifacts = (
        ArtifactBinding(
            name="model",
            kind=ArtifactKind.TREE,
            reference=blueprint_tree,
            size_bytes=10,
            files=1,
        ),
    )
    registry.register_workflow(
        "wfl-bp", [], WorkflowSched(), blueprints=[PersistedTask(record=blueprint)]
    )

    tree = _ref(b"manifest", media_type=TREE_MEDIA_TYPE)
    largest = _ref(b"largest-member")
    file_binding = ArtifactBinding(
        name="weights", kind=ArtifactKind.FILE, reference=_ref(b"weights"), size_bytes=7
    )
    tree_binding = ArtifactBinding(
        name="ckpt",
        kind=ArtifactKind.TREE,
        reference=tree,
        size_bytes=100,
        files=2,
        largest=largest,
    )
    fx.raw.hset(
        task_artifacts_key("tsk-1"),
        mapping={
            artifact_field(*b.key): b.model_dump_json()
            for b in (file_binding, tree_binding)
        },
    )

    history_result = _ref(b"history-result")
    member_value = _ref(b"history-member", _B)
    output_value = _ref(b"output-value")
    output = OutputHistory(name="out", value=ResultBinding(reference=output_value))
    fx.raw.hset(
        workflow_history_key("wfl-2"),
        mapping={
            CERTIFICATE_FIELD: HistoryCertificate(
                schema_version=1, seq=1
            ).model_dump_json(),
            "tsk-9": TaskHistory(
                result=ResultBinding(
                    reference=history_result,
                    members=(
                        ResultMember(
                            key="0",
                            outcome="success",
                            binding=ResultBinding(reference=member_value),
                        ),
                    ),
                )
            ).model_dump_json(),
            output.field: output.model_dump_json(),
            "@outputs:out": json.dumps([output.field]),
        },
    )

    finalized = _ref(b"finalized")
    fx.raw.set(
        content_finalization_key(_A, "idm-1"),
        OutcomeManifest(content=finalized, idempotency_key="idm-1").model_dump_json(),
    )

    references = (
        *ledger_refs,
        result,
        dispatched.reference,
        blueprint_result,
        blueprint_tree,
        file_binding.reference,
        tree,
        largest,
        history_result,
        member_value,
        output_value,
        finalized,
    )
    trees = {_identity(tree), _identity(blueprint_tree)}
    return {_identity(r) for r in references}, trees


def test_every_reference_each_family_binds_is_marked(fx: _Fixture) -> None:
    expected, trees = _seed_everything(fx)

    result = fx.run()

    assert result.complete
    assert all(result.mark.binds(scope, digest) for scope, digest in expected)
    assert result.mark.counts() == {
        _A: sum(scope == _A for scope, _ in expected),
        _B: sum(scope == _B for scope, _ in expected),
    }
    assert set(result.trees) == trees
    assert all(
        (r.authorization_scope, r.content_digest) == identity
        for identity, r in result.trees.items()
    )
    assert result.unattributed == 0 and not result.faults
    assert result.gap(_A) == result.gap(_B) == ""
    assert set(result.reports) == {family.value for family in Family}
    assert all(report.keys for report in result.reports.values())


def test_one_scan_of_the_keyspace_reads_every_family(
    fx: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected, _ = _seed_everything(fx)
    pages: list[int] = []
    original = fx.marker._scan_page

    def counting(cursor: int) -> tuple[int, list[str]]:
        pages.append(cursor)
        return original(cursor)

    monkeypatch.setattr(fx.marker, "_scan_page", counting)
    result = fx.run()

    assert pages == [0]
    assert all(result.mark.binds(scope, digest) for scope, digest in expected)


def test_a_large_hash_is_read_a_page_of_fields_at_a_time(
    fx: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = [_ref(f"record-{i}".encode()) for i in range(1500)]
    fx.raw.hset(
        workflow_ds_key("wfl-big"),
        mapping={
            f"records:[{i}]": _record(ref).model_dump_json()
            for i, ref in enumerate(refs)
        },
    )
    pages: list[int] = []
    original = fx.marker._hscan_page

    def counting(key: str, cursor: int) -> tuple[int, dict[str, str]]:
        cursor, fields = original(key, cursor)
        pages.append(len(fields))
        return cursor, fields

    monkeypatch.setattr(fx.marker, "_hscan_page", counting)
    result = fx.run()

    assert all(result.mark.binds(_A, ref.content_digest) for ref in refs)
    assert result.reports[Family.LEDGER.value].values == 1500
    assert len(pages) > 1 and sum(pages) == 1500
    assert max(pages) < 1500


def test_an_undecodable_ledger_field_leaves_every_scope_incomplete(
    fx: _Fixture,
) -> None:
    _ledger(fx, "wfl-1", [_ref(b"a"), _ref(b"b")])
    fx.raw.hset(workflow_ds_key("wfl-1"), "records:[1]", '{"value_ref": 3}')

    result = fx.run()

    assert result.complete and result.unattributed == 1
    assert result.gap(_A) and result.gap(_B) and result.gap("org-unseen")
    assert result.mark.binds(_A, _ref(b"a").content_digest)
    assert result.mark.binds(_A, _ref(b"b").content_digest)


def test_a_key_of_the_wrong_type_leaves_every_scope_incomplete(fx: _Fixture) -> None:
    fx.raw.set(workflow_history_key("wfl-x"), "a string")

    result = fx.run()

    assert result.unattributed == 1
    assert result.gap(_A) and result.gap(_B)


def test_an_undecodable_finalization_leaves_only_its_scope_incomplete(
    fx: _Fixture,
) -> None:
    fx.raw.set(content_finalization_key(_B, "idm-9"), "not a manifest")

    result = fx.run()

    assert result.faults == {_B: 1} and result.unattributed == 0
    assert result.gap(_B)
    assert result.gap(_A) == ""


def test_a_task_state_whose_source_is_gone_leaves_its_scope_incomplete(
    fx: _Fixture,
) -> None:
    record = _task_record("tsk-1", _ref(b"result", _B), _B)
    WorkflowRegistry(fx.rds).save_task_states([PersistedTask(record=record)])
    fx.raw.delete(workflow_sources_key("wfl-tasks"))

    result = fx.run()

    assert result.faults == {_B: 1} and result.unattributed == 0
    assert result.gap(_B)
    assert result.gap(_A) == ""


def test_a_set_stop_ends_the_pass_incomplete(fx: _Fixture) -> None:
    _seed_everything(fx)
    stop = threading.Event()
    stop.set()

    result = fx.run(stop)

    assert not result.complete
    assert result.gap(_A) and result.gap(_B)


class _Trail:
    """What the pass did, in order: each value decoded, each page read, each wait."""

    def __init__(self) -> None:
        self.steps: list[str] = []

    def values_between_waits(self) -> int:
        longest = run = 0
        for step in self.steps:
            run = 0 if step == "wait" else run + (step == "value")
            longest = max(longest, run)
        return longest

    def pages_without_a_wait(self) -> list[str]:
        """Each page read with no wait since the previous page of its stream."""
        waited: dict[str, bool] = {}
        unpaced = []
        for step in self.steps:
            if step == "wait":
                waited = dict.fromkeys(waited, True)
            elif step.startswith("page:"):
                if waited.get(step) is False:
                    unpaced.append(step)
                waited[step] = False
        return unpaced


class _WatchedStop(threading.Event):
    def __init__(self, trail: _Trail) -> None:
        super().__init__()
        self._trail = trail

    def wait(self, timeout: float | None = None) -> bool:
        self._trail.steps.append("wait")
        return super().wait(timeout)


def test_the_pass_waits_on_stop_every_few_values_and_between_pages(
    fx: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    for w in range(3):
        fx.raw.hset(
            workflow_ds_key(f"wfl-{w}"),
            mapping={
                f"records:[{i}]": _record(_ref(f"{w}-{i}".encode())).model_dump_json()
                for i in range(700)
            },
        )
    for i in range(250):
        fx.raw.set(
            content_finalization_key(_A, f"idm-{i}"),
            OutcomeManifest(
                content=_ref(f"fin-{i}".encode()), idempotency_key=f"idm-{i}"
            ).model_dump_json(),
        )
    trail = _Trail()
    marker = Marker(fx.rds, _LOGGER, page_pause_sec=0.0001, duty=1.0)
    decode, finalize = marking.ledger_field, marking.finalization_value
    scan_page, hscan_page = marker._scan_page, marker._hscan_page

    def decoding(name: str, value: str) -> Any:
        trail.steps.append("value")
        return decode(name, value)

    def finalizing(key: str, value: str) -> Any:
        trail.steps.append("value")
        return finalize(key, value)

    def scanning(cursor: int) -> tuple[int, list[str]]:
        trail.steps.append("page:keys")
        return scan_page(cursor)

    def hscanning(key: str, cursor: int) -> tuple[int, dict[str, str]]:
        trail.steps.append(f"page:{key}")
        return hscan_page(key, cursor)

    monkeypatch.setitem(marking._FIELD_DECODERS, Family.LEDGER, decoding)
    monkeypatch.setattr(marking, "finalization_value", finalizing)
    monkeypatch.setattr(marking, "SCAN_COUNT", 50)
    monkeypatch.setattr(marker, "_scan_page", scanning)
    monkeypatch.setattr(marker, "_hscan_page", hscanning)

    result = marker.run(_WatchedStop(trail))

    assert result.complete
    assert trail.steps.count("value") == 3 * 700 + 250
    assert trail.values_between_waits() <= VALUES_PER_PAUSE
    assert trail.pages_without_a_wait() == []
    assert trail.steps.count("page:keys") > 1
    assert trail.steps.count(f"page:{workflow_ds_key('wfl-0')}") > 1


def test_a_settled_result_reference_is_marked_through_its_ledger_write(
    fx: _Fixture,
) -> None:
    registry = WorkflowRegistry(fx.rds)
    eng = engine(chain_bundle())
    initial = eng.ledger_changes()
    registry.save_ledger(WORKFLOW_ID, initial)
    eng.ledger_written(initial)
    result = reference_for("org", b"result of A", media_type="application/json")

    eng.on_dispatched("A", "w1")
    eng.on_succeeded("A", content=result)
    changes = eng.ledger_changes()
    registry.save_ledger(WORKFLOW_ID, changes)

    marked = fx.run()

    assert marked.complete and marked.unattributed == 0 and not marked.faults
    assert marked.mark.binds("org", result.content_digest)
    assert not marked.mark.binds(_A, result.content_digest)


def _unclassified(module: ModuleType) -> set[str]:
    """The module's public key builders KEY_FAMILIES does not classify."""
    builders = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (
            (inspect.isfunction(value) and name.endswith(("_key", "_channel")))
            or (name.endswith("_KEY") and isinstance(value, str))
        )
    }
    return builders - set(KEY_FAMILIES)


def test_every_control_store_key_builder_is_classified() -> None:
    assert _unclassified(redis_module) == set()


def test_an_unclassified_key_builder_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        redis_module, "brand_new_key", lambda task_id: f"x:{task_id}", raising=False
    )
    monkeypatch.setattr(redis_module, "BRAND_NEW_KEY", "x:new", raising=False)

    assert _unclassified(redis_module) == {"brand_new_key", "BRAND_NEW_KEY"}


def test_each_classified_builder_keys_its_own_family() -> None:
    assert {f for f in KEY_FAMILIES.values() if f is not None} == set(Family)
    for name, family in KEY_FAMILIES.items():
        builder = getattr(redis_module, name)
        if callable(builder):
            arity = len(inspect.signature(builder).parameters)
            key = builder(*(f"x{i}" for i in range(arity)))
        else:
            key = builder
        assert isinstance(key, str), name
        assert marking._family_of(key, Family) is family, name
