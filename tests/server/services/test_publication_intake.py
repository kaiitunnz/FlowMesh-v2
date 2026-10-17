"""Publications leave the task-event consumer: the stream moves on at receipt, the
lane holds what a report names before binding it, and the answer and the release of
the holds wait for the binding commit to be durable."""

import asyncio
import hashlib
import threading
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import (
    RECLAIM_ROOT_EPOCH_KEY,
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_protected_key,
)
from server.reclamation.pins import PinStore
from server.reclamation.publications import PublicationHolds
from server.reclamation.trees import TreeExpander
from server.services.publications import PublicationIntake
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.content import ContentReference, StoreRefusal
from shared.content.tree import TREE_MEDIA_TYPE, TreeManifest, TreeMember
from shared.schemas.event import TaskEvent
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from shared.tasks.artifact_upload import ARTIFACTS_PENDING
from shared.tasks.publication import PublicationAnswer, PublicationDisposition
from tests.server.dispatch_helpers import record_dispatch
from tests.server.redis_helpers import fake_redis_client
from tests.server.result_store import result_payload
from tests.server.services.test_monitor_commit_fence import (
    _consume,
    _monitor,
    _Store,
)
from tests.server.services.test_task_event_fence import _ECHO_V2
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_v2_orchestration import _register
from tests.support.waiting import pop_ready


class _Answers:
    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.answers: list[PublicationAnswer] = []

    def __call__(self, answer: PublicationAnswer) -> None:
        with self.cv:
            self.answers.append(answer)
            self.cv.notify_all()

    def wait(self, count: int = 1) -> list[PublicationAnswer]:
        with self.cv:
            assert self.cv.wait_for(lambda: len(self.answers) >= count, timeout=5)
            return list(self.answers)


class _Backed:
    """A store check that answers once released."""

    def __init__(self, answer: bool | None = True) -> None:
        self.answer = answer
        self.asked = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(self, reference: ContentReference) -> bool | None:
        self.asked.set()
        assert self.release.wait(5)
        return self.answer


class _Rig:
    def __init__(self, backed: _Backed | None = None) -> None:
        self.store = _Store()
        self.runtime: TaskRuntime = _runtime(self.store)
        self.runtime.claim_root_epoch()
        self.workflow_id, ids = asyncio.run(_register(self.runtime, _ECHO_V2))
        self.task_id = next(iter(ids.values()))
        assert pop_ready(self.runtime, 0.05) == self.task_id
        record_dispatch(self.runtime, self.task_id, "wkr-1", "dsp-1")
        server = fakeredis.FakeServer()
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.raw.set(RECLAIM_ROOT_EPOCH_KEY, str(self.runtime.root_epoch))
        self.backed = backed or _Backed()
        self.answers = _Answers()
        setattr(self.runtime, "send_publication_answer", self.answers)
        self.monitor, self.telemetry, _ = _monitor(self.runtime)
        client = fake_redis_client(server)
        reader = self.runtime._results
        holds = PublicationHolds(
            PinStore(client),
            lambda: self.runtime.nonces,
            self.backed,
            trees=TreeExpander(client, reader.open, reader.stored_size),
        )
        self.intake = PublicationIntake(
            self.runtime, holds, self.monitor._apply_publication
        )
        self.monitor._publications = self.intake
        self.steps = threading.Condition()
        self.published = self.finished = 0
        self._count(self.intake, "_publish", "published")
        self._count(self.intake._finisher, "_finish", "finished")
        self.intake.start()

    def _count(self, owner: Any, method: str, counter: str) -> None:
        """Count each time ``owner``'s ``method`` returns."""
        run = getattr(owner, method)

        def counted(*args: Any) -> None:
            try:
                run(*args)
            finally:
                with self.steps:
                    setattr(self, counter, getattr(self, counter) + 1)
                    self.steps.notify_all()

        setattr(owner, method, counted)

    def wait_steps(self, published: int = 0, finished: int = 0) -> None:
        """Wait until the lane has run ``published`` publications and the finisher
        ``finished`` jobs."""
        with self.steps:
            assert self.steps.wait_for(
                lambda: self.published >= published and self.finished >= finished,
                timeout=5,
            )

    def success(self, dispatch_id: str = "dsp-1") -> TaskEvent:
        record = self.runtime.get_record(self.task_id)
        assert record is not None
        return TaskEvent(
            type="TASK_SUCCEEDED",
            task_id=self.task_id,
            worker_id="wkr-1",
            dispatch_id=dispatch_id,
            payload=result_payload(
                self.runtime._results, self.task_id, {"value": 1}, record.org_id
            ),
        )

    def scope_and_digest(self, event: TaskEvent) -> tuple[str, str]:
        reference = ContentReference.model_validate(
            (event.payload or {})["result_reference"]
        )
        return reference.authorization_scope, reference.content_digest

    def released(self, scope: str) -> bool:
        """Wait for the finisher to end the publication's holds."""
        with self.steps:
            return self.steps.wait_for(lambda: not self.pins(scope), timeout=5)

    def pins(self, scope: str) -> dict[str, Any]:
        return cast(dict[str, Any], self.raw.hgetall(reclaim_pins_key(scope)))

    def status(self) -> str:
        record = self.runtime.get_record(self.task_id)
        assert record is not None
        return record.status

    def stop(self) -> None:
        self.intake.stop(1.0)


@pytest.fixture
def rig() -> Any:
    made: list[_Rig] = []

    def make(backed: _Backed | None = None) -> _Rig:
        made.append(_Rig(backed))
        return made[-1]

    yield make
    for each in made:
        each.stop()


def test_a_success_commits_off_the_consumer_and_answers_once_durable(rig) -> None:
    r = rig()
    event = r.success()
    assert _consume(r.monitor, event) == "1-0"
    [answer] = r.answers.wait()
    assert answer.disposition is PublicationDisposition.COMMITTED
    assert answer.publication.dispatch_id == "dsp-1"
    assert r.status() == TaskStatus.DONE
    assert r.store.status(r.task_id) == TaskStatus.DONE
    scope, _ = r.scope_and_digest(event)
    assert r.released(scope)


def test_a_retiring_object_is_refused_and_binds_nothing_until_resent(rig) -> None:
    r = rig()
    event = r.success()
    scope, digest = r.scope_and_digest(event)
    r.raw.hset(reclaim_deleting_key(scope), digest, "{}")
    _consume(r.monitor, event)
    [refused] = r.answers.wait()
    assert refused.refusal is StoreRefusal.RETRY_RETIREMENT
    assert refused.digests == [digest]
    assert r.status() == TaskStatus.DISPATCHED
    assert r.pins(scope) == {}

    r.raw.hdel(reclaim_deleting_key(scope), digest)
    _consume(r.monitor, event)
    answers = r.answers.wait(2)
    assert answers[-1].disposition is PublicationDisposition.COMMITTED
    assert r.status() == TaskStatus.DONE


def test_an_object_gone_from_the_store_is_asked_for_again(rig) -> None:
    r = rig(_Backed(False))
    event = r.success()
    scope, digest = r.scope_and_digest(event)
    r.raw.hset(reclaim_history_key(scope), digest, "1")
    _consume(r.monitor, event)
    [answer] = r.answers.wait()
    assert answer.refusal is StoreRefusal.REMATERIALIZE
    assert answer.digests == [digest]
    assert r.status() == TaskStatus.DISPATCHED


def test_a_stale_success_is_rejected_where_it_arrives(rig) -> None:
    r = rig()
    _consume(r.monitor, r.success("dsp-other"))
    [answer] = r.answers.wait()
    assert answer.disposition is PublicationDisposition.REJECTED
    assert r.status() == TaskStatus.DISPATCHED


def test_a_success_reported_after_its_worker_failed_the_dispatch_is_rejected(
    rig,
) -> None:
    r = rig()
    event = r.success()
    failed = TaskEvent(
        type="TASK_FAILED",
        task_id=r.task_id,
        worker_id="wkr-1",
        dispatch_id="dsp-1",
        error="the copy of a stored object is gone",
        retryable=True,
        ambiguous=True,
    )
    _consume(r.monitor, failed)
    assert r.status() != TaskStatus.DISPATCHED

    _consume(r.monitor, event)
    [answer] = r.answers.wait()
    assert answer.disposition is PublicationDisposition.REJECTED
    assert r.status() != TaskStatus.DONE


def test_a_resent_success_after_its_commit_is_answered_committed_again(rig) -> None:
    r = rig()
    event = r.success()
    _consume(r.monitor, event)
    r.answers.wait()
    _consume(r.monitor, event)
    answers = r.answers.wait(2)
    assert [a.disposition for a in answers] == [PublicationDisposition.COMMITTED] * 2


def test_a_held_commit_answers_and_releases_only_once_durable(rig) -> None:
    r = rig()
    event = r.success()
    scope, digest = r.scope_and_digest(event)
    r.store.down.add(r.workflow_id)
    _consume(r.monitor, event)
    r.wait_steps(published=1)
    assert r.answers.answers == []
    # The binding is not durable, so the hold stays.
    assert digest in r.pins(scope)

    r.store.down.clear()
    r.runtime._retry_durability(r.workflow_id)
    [answer] = r.answers.wait()
    assert answer.disposition is PublicationDisposition.COMMITTED
    assert r.released(scope)


def test_a_cancel_while_the_objects_are_verified_settles_and_rejects(rig) -> None:
    backed = _Backed(True)
    backed.release.clear()
    r = rig(backed)
    event = r.success()
    scope, digest = r.scope_and_digest(event)
    r.raw.hset(reclaim_history_key(scope), digest, "1")
    _consume(r.monitor, event)
    assert backed.asked.wait(5)
    r.runtime.cancel_workflow(r.workflow_id)
    backed.release.set()
    [answer] = r.answers.wait()
    assert answer.disposition is PublicationDisposition.REJECTED
    assert r.status() in (TaskStatus.CANCELLED, TaskStatus.CANCELLING)
    assert r.released(scope)


def _with_tree(r: Any, manifest_bytes: bytes) -> tuple[TaskEvent, Any]:
    record = r.runtime.get_record(r.task_id)
    store = r.runtime._results.store
    manifest = store.write(record.org_id, manifest_bytes, media_type=TREE_MEDIA_TYPE)
    event = r.success()
    payload = dict(event.payload or {})
    payload["artifacts"] = [
        ArtifactBinding(
            name="model",
            kind=ArtifactKind.TREE,
            reference=manifest,
            size_bytes=1,
            files=1,
        ).model_dump(mode="json")
    ]
    return event.model_copy(update={"payload": payload}), manifest


def test_a_write_through_tree_protects_its_members_once_committed(rig) -> None:
    r = rig()
    record = r.runtime.get_record(r.task_id)
    store = r.runtime._results.store
    member = store.write(record.org_id, b"weights")
    tree = TreeManifest.of(
        [TreeMember("w.bin", member.content_digest, member.size_bytes)]
    )
    event, manifest = _with_tree(r, tree.encode())
    _consume(r.monitor, event)
    [answer] = r.answers.wait()
    assert answer.disposition is PublicationDisposition.COMMITTED
    scope = manifest.authorization_scope
    protected = reclaim_protected_key(scope)
    for _ in range(100):
        if r.raw.sismember(protected, member.content_digest):
            break
        threading.Event().wait(0.02)
    assert r.raw.sismember(protected, member.content_digest)
    assert r.released(scope)


def test_a_success_naming_a_malformed_tree_fails_unaccepted(rig) -> None:
    r = rig()
    event, _ = _with_tree(r, b'{"schema":1,"members":[["../x","00",1]]}')
    _consume(r.monitor, event)
    [answer] = r.answers.wait()
    assert answer.disposition is PublicationDisposition.REJECTED
    record = r.runtime.get_record(r.task_id)
    assert record is not None and record.status == TaskStatus.FAILED
    assert "artifacts_unaccepted" in (record.error or "")


def test_a_staged_tree_with_a_member_being_retired_is_expanded_once_it_ends(
    rig,
) -> None:
    r = rig()
    record = r.runtime.get_record(r.task_id)
    member = TreeMember("w.bin", hashlib.sha256(b"weights").hexdigest(), 7)
    event, manifest = _with_tree(r, TreeManifest.of([member]).encode())
    event = event.model_copy(
        update={"payload": {**(event.payload or {}), ARTIFACTS_PENDING: True}}
    )
    scope = manifest.authorization_scope
    r.raw.hset(reclaim_deleting_key(scope), member.digest, "{}")

    _consume(r.monitor, event)
    [answer] = r.answers.wait()
    # Accepted pending: the upload has not stored the member yet.
    assert answer.disposition is PublicationDisposition.COMMITTED
    assert r.runtime._expanding.get(r.task_id) == {manifest.content_digest}
    r.wait_steps(published=1, finished=1)
    protected = reclaim_protected_key(scope)
    assert not r.raw.sismember(protected, member.digest)

    r.raw.hdel(reclaim_deleting_key(scope), member.digest)
    r.wait_steps(published=1, finished=2)
    assert r.raw.sismember(protected, member.digest)
    assert record is not None and not r.runtime._expanding.get(r.task_id)


def test_restored_staged_trees_are_registered_before_any_event_is_consumed() -> None:
    order: list[str] = []
    store = _Store()
    runtime: TaskRuntime = _runtime(store)
    monitor, _, _ = _monitor(runtime)

    class _Intake:
        def start(self) -> None:
            started = any(t.ident is not None for t in monitor._threads or ())
            order.append("intake after a consumer" if started else "intake")

        def stop(self, timeout: float = 5.0) -> None:
            pass

    def consumes() -> None:
        order.append("consumer")

    for loop in (
        "_tasks_events_loop",
        "_workers_events_loop",
        "_node_events_loop",
        "_finalizer_loop",
    ):
        setattr(
            monitor, loop, consumes if loop == "_tasks_events_loop" else lambda: None
        )
    monitor._publications = cast(Any, _Intake())

    async def start() -> None:
        monitor.start()

    asyncio.run(start())
    for thread in monitor._threads or ():
        thread.join(1)
    assert order == ["intake", "consumer"]
