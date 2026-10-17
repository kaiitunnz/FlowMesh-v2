"""The backfill: every reference a stored binding names is protected in one scan of the
keyspace, an undecodable key withholds certification, and a generation is read once."""

import json
import logging
import threading
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import (
    RECLAIM_BACKFILL_FAULTS_KEY,
    RECLAIM_BACKFILL_KEY,
    RECLAIM_HOOKS_ON_KEY,
    content_finalization_key,
    reclaim_coverage_key,
    reclaim_protected_key,
    reclaim_unexpanded_key,
    task_artifacts_key,
    workflow_ds_key,
    workflow_history_key,
)
from server.orchestration.ledger_layout import field_name, keyed_value
from server.orchestration.state import (
    PublicationOutcome,
    Record,
    ResultPublication,
    ValueMember,
    ValueRef,
)
from server.reclamation.audit import AuditTrail
from server.reclamation.backfill import FAMILIES, Backfill, Family
from server.reclamation.certification import (
    ALL_SCOPES,
    CertificationStatus,
    CoverageStore,
)
from server.registries.artifacts import artifact_field
from server.registries.task_states import PersistedTask
from server.registries.workflow import WorkflowRegistry
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
from tests.server.redis_helpers import fake_redis_client

_A = "org-a"
_B = "org-b"
_LOGGER = logging.getLogger("test.backfill")


def _ref(
    data: bytes, scope: str = _A, media_type: str | None = None
) -> ContentReference:
    if media_type is None:
        return reference_for(scope, data)
    return reference_for(scope, data, media_type=media_type)


class _Fixture:
    def __init__(self) -> None:
        server = fakeredis.FakeServer()
        self.rds = fake_redis_client(server)
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.coverage = CoverageStore(self.rds, FAMILIES)
        self.backfill = Backfill(
            self.rds,
            self.coverage,
            _LOGGER,
            AuditTrail(self.rds),
            page_pause_sec=0,
            retry_pause_sec=0,
        )

    def protected(self, scope: str) -> set[str]:
        return cast(set[str], self.raw.smembers(reclaim_protected_key(scope)))

    def forget_protection(self) -> None:
        for key in self.raw.scan_iter("rc:*"):
            self.raw.delete(key)

    def start(self, epoch: int) -> int:
        return self.coverage.mark_hooks_on(epoch)


@pytest.fixture
def fx() -> _Fixture:
    return _Fixture()


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
    record = Record(
        operator_id="op",
        activation_id="act-1",
        scope_id="scp-1",
        value_ref=ValueRef(kind="legacy_task_result", content=refs[1]),
    )
    fx.raw.hset(
        workflow_ds_key(workflow_id),
        mapping={
            "meta:layout": "1",
            field_name("result_publications", "slot-1"): keyed_value(0, publication),
            "records:[0]": record.model_dump_json(),
            "trace:[0]": "{not decoded: the trace binds nothing}",
        },
    )


def _task_record(task_id: str, result: ContentReference, scope: str) -> TaskRecord:
    return TaskRecord.model_validate(
        {
            "task_id": task_id,
            "workflow_id": "wfl-tasks",
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


def _seed_everything(fx: _Fixture) -> set[tuple[str, str]]:
    """Store one binding of every family, as each writer stores it, and return the
    identities they name."""
    expected: set[tuple[str, str]] = set()

    ledger_refs = [_ref(b"ledger-member"), _ref(b"ledger-record", _B)]
    _ledger(fx, "wfl-1", ledger_refs)

    result = _ref(b"task-result")
    dispatched = ArtifactBinding(
        name="input", kind=ArtifactKind.FILE, reference=_ref(b"input"), size_bytes=5
    )
    record = _task_record("tsk-1", result, _A)
    record.internal.dispatched_artifacts = (dispatched,)
    WorkflowRegistry(fx.rds).save_task_states([PersistedTask(record=record)])

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
    output = OutputHistory(
        name="out", value=ResultBinding(reference=_ref(b"output-value"))
    )
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

    for reference in (
        *ledger_refs,
        result,
        dispatched.reference,
        file_binding.reference,
        tree,
        largest,
        history_result,
        member_value,
        _ref(b"output-value"),
        finalized,
    ):
        expected.add((reference.authorization_scope, reference.content_digest))
    return expected


def test_the_backfill_protects_every_reference_each_family_binds(fx) -> None:
    expected = _seed_everything(fx)
    fx.forget_protection()
    generation = fx.start(3)

    reports = fx.backfill.run(generation, threading.Event())

    assert reports is not None and set(reports) == set(FAMILIES)
    protected = {(scope, d) for scope in (_A, _B) for d in fx.protected(scope)}
    assert protected == expected
    assert fx.raw.smembers(reclaim_unexpanded_key(_A)) == {
        _ref(b"manifest").content_digest
    }
    assert fx.raw.hgetall(RECLAIM_BACKFILL_KEY) == dict.fromkeys(FAMILIES, "3")
    assert fx.raw.hget(reclaim_coverage_key(_B), Family.LEDGER.value) == "3"
    assert not fx.raw.exists(RECLAIM_BACKFILL_FAULTS_KEY)


def test_a_large_hash_is_protected_a_page_of_fields_at_a_time(fx) -> None:
    refs = [_ref(f"record-{i}".encode()) for i in range(1500)]
    fx.raw.hset(
        workflow_ds_key("wfl-big"),
        mapping={
            f"records:[{i}]": Record(
                operator_id="op",
                activation_id="act",
                scope_id="scp",
                value_ref=ValueRef(kind="legacy_task_result", content=ref),
            ).model_dump_json()
            for i, ref in enumerate(refs)
        },
    )
    generation = fx.start(1)

    reports = fx.backfill.scan([Family.LEDGER], generation, threading.Event())

    assert reports is not None
    assert fx.protected(_A) == {ref.content_digest for ref in refs}
    assert reports[Family.LEDGER].values == 1500


def test_an_undecodable_ledger_field_withholds_every_scope(fx) -> None:
    _ledger(fx, "wfl-1", [_ref(b"a"), _ref(b"b")])
    fx.raw.hset(workflow_ds_key("wfl-1"), "records:[1]", '{"value_ref": 3}')
    generation = fx.start(2)

    fx.backfill.run(generation, threading.Event())

    assert fx.raw.hget(RECLAIM_BACKFILL_FAULTS_KEY, ALL_SCOPES) == "2"
    assert fx.protected(_A) == {_ref(b"a").content_digest, _ref(b"b").content_digest}
    assert fx.coverage.certify(_A).status is CertificationStatus.UNDECODABLE


def test_an_undecodable_finalization_withholds_only_its_scope(fx) -> None:
    fx.raw.set(content_finalization_key(_B, "idm-9"), "not a manifest")
    generation = fx.start(2)

    fx.backfill.run(generation, threading.Event())

    assert fx.raw.hgetall(RECLAIM_BACKFILL_FAULTS_KEY) == {_B: "2"}
    assert fx.coverage.certify(_B).status is CertificationStatus.UNDECODABLE
    assert fx.coverage.certify(_A).certified


def test_a_task_state_whose_source_is_gone_withholds_its_scope(fx) -> None:
    record = _task_record("tsk-1", _ref(b"result", _B), _B)
    WorkflowRegistry(fx.rds).save_task_states([PersistedTask(record=record)])
    fx.raw.delete("workflow:wfl-tasks:sources")
    generation = fx.start(2)

    fx.backfill.run(generation, threading.Event())

    assert fx.raw.hgetall(RECLAIM_BACKFILL_FAULTS_KEY) == {_B: "2"}


def test_a_key_of_the_wrong_type_withholds_every_scope(fx) -> None:
    fx.raw.set(workflow_history_key("wfl-x"), "a string")
    generation = fx.start(2)

    fx.backfill.run(generation, threading.Event())

    assert fx.raw.hget(RECLAIM_BACKFILL_FAULTS_KEY, ALL_SCOPES) == "2"


def test_one_scan_of_the_keyspace_reads_every_family(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _seed_everything(fx)
    fx.forget_protection()
    pages: list[int] = []
    original = fx.backfill._scan_page

    def counting(cursor: int) -> tuple[int, list[str]]:
        pages.append(cursor)
        return original(cursor)

    monkeypatch.setattr(fx.backfill, "_scan_page", counting)
    assert fx.backfill.run(fx.start(1), threading.Event()) is not None

    assert pages == [0]
    protected = {(scope, d) for scope in (_A, _B) for d in fx.protected(scope)}
    assert protected == expected


def test_a_later_root_reads_nothing_its_generation_completed(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_everything(fx)
    fx.forget_protection()
    first = fx.start(1)
    assert fx.backfill.run(first, threading.Event()) is not None
    certified = fx.coverage.certify(_B)
    assert certified.certified

    second = fx.start(2)
    scanned: list[Any] = []

    def scan(*args: Any) -> dict[str, Any]:
        scanned.append(args)
        return {}

    monkeypatch.setattr(fx.backfill, "scan", scan)

    assert second == first
    assert fx.backfill.run(second, threading.Event()) == {}
    assert scanned == []
    assert fx.coverage.certify(_B) == certified


def test_an_interrupted_scan_reads_every_family_again_from_the_start(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_everything(fx)
    fx.forget_protection()
    generation = fx.start(1)
    stop = threading.Event()
    original = fx.backfill._flush

    def flush_then_stop(*args: Any) -> None:
        original(*args)
        stop.set()

    monkeypatch.setattr(fx.backfill, "_flush", flush_then_stop)
    monkeypatch.setattr("server.reclamation.backfill.SCAN_COUNT", 1)
    assert fx.backfill.run(generation, stop) is None
    assert fx.coverage.completed(generation) == frozenset()
    assert not [key for key in fx.raw.scan_iter("rc:*") if "cursor" in key]

    monkeypatch.setattr(fx.backfill, "_flush", original)
    assert fx.start(2) == generation
    assert fx.backfill.run(generation, threading.Event()) is not None
    assert fx.coverage.completed(generation) == set(FAMILIES)


def test_an_undecodable_key_has_a_later_root_read_everything_again(fx) -> None:
    _ledger(fx, "wfl-1", [_ref(b"a"), _ref(b"b")])
    fx.raw.hset(workflow_ds_key("wfl-1"), "records:[1]", '{"value_ref": 3}')
    first = fx.start(1)
    fx.backfill.run(first, threading.Event())
    assert fx.coverage.certify(_A).status is CertificationStatus.UNDECODABLE

    fx.raw.hdel(workflow_ds_key("wfl-1"), "records:[1]")
    second = fx.start(2)
    reports = fx.backfill.run(second, threading.Event())

    assert second == 2
    assert reports is not None and set(reports) == set(FAMILIES)
    assert fx.coverage.certify(_A).certified


def test_a_store_that_lost_its_marker_is_read_again(fx) -> None:
    _ledger(fx, "wfl-1", [_ref(b"a"), _ref(b"b")])
    fx.backfill.run(fx.start(1), threading.Event())
    fx.raw.delete(RECLAIM_HOOKS_ON_KEY, reclaim_protected_key(_A))

    generation = fx.start(4)
    reports = fx.backfill.run(generation, threading.Event())

    assert generation == 4
    assert reports is not None and set(reports) == set(FAMILIES)
    assert fx.protected(_A) == {_ref(b"a").content_digest, _ref(b"b").content_digest}


def test_a_scan_that_fails_is_retried_from_its_start(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ledger(fx, "wfl-1", [_ref(b"a"), _ref(b"b")])
    generation = fx.start(1)
    original = fx.backfill.scan
    attempts: list[int] = []

    def flaky(families: Any, generation_: int, stop: threading.Event) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("lost the store")
        return original(families, generation_, stop)

    monkeypatch.setattr(fx.backfill, "scan", flaky)
    assert fx.backfill.run(generation, threading.Event()) is not None
    assert len(attempts) == 2
    assert fx.protected(_A) == {_ref(b"a").content_digest, _ref(b"b").content_digest}


def test_a_backfill_superseded_by_a_later_generation_records_no_completion(
    fx,
) -> None:
    generation = fx.start(1)
    fx.coverage.record_fault(generation, None)
    fx.start(2)

    assert fx.backfill.run(generation, threading.Event()) is None
    assert fx.coverage.completed(1) == frozenset()
