"""Commits carrying asserted state: the epoch guard follows that state through held
writes until a write carrying it lands, a later root stops this one, a faulted write
keeps its scope out of collection, and the publication check reads without recording
a dispatch."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from redis.exceptions import ResponseError

from server.reclamation.epoch import CommitGuard, EpochFenced, GuardInterrupted
from server.registries.worker import Worker
from server.task.models import PublishGate, TaskStatus
from server.task.runtime import TaskRuntime
from shared.tasks.publication import Publication
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import result_payload
from tests.server.services.test_task_event_fence import _ECHO_V2
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_v2_orchestration import FakeRegistry, _register
from tests.support.waiting import pop_ready


class _Guarded(FakeRegistry):
    """Records the guard each record commit carries, and fails commits while
    ``failure`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.guards: list[CommitGuard | None] = []
        self.failure: BaseException | None = None

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        self.guards.append(kwargs.get("guard"))
        if self.failure is not None:
            raise self.failure
        super().commit_transition(workflow_id, **kwargs)

    def save_ledger(self, workflow_id: str, ledger: Any, *args: Any, **kwargs: Any):
        if self.failure is not None:
            raise self.failure
        super().save_ledger(workflow_id, ledger, *args, **kwargs)


class _Rig:
    def __init__(self) -> None:
        self.store = _Guarded()
        self.runtime: TaskRuntime = _runtime(self.store)
        self.epoch = self.runtime.claim_root_epoch()
        self.workflow_id, ids = asyncio.run(_register(self.runtime, _ECHO_V2))
        self.task_id = next(iter(ids.values()))
        self.fenced: list[bool] = []
        self.runtime.on_fenced = lambda: self.fenced.append(True)
        self.answers: list[Any] = []
        setattr(self.runtime, "send_publication_answer", self.answers.append)

    def dispatch(self) -> None:
        assert pop_ready(self.runtime, 0.05) == self.task_id
        record_dispatch(self.runtime, self.task_id, "wkr-1", "dsp-1")

    def payload(self) -> dict[str, Any]:
        record = self.runtime.get_record(self.task_id)
        assert record is not None
        return result_payload(
            self.runtime._results, self.task_id, {"value": 1}, record.org_id
        )

    def succeed(self, payload: dict[str, Any]) -> None:
        with self.runtime.transition():
            self.runtime.assert_publication(self.task_id)
            self.runtime.mark_succeeded(
                self.task_id, "wkr-1", payload, "2026-10-17T00:00:00Z", "dsp-1"
            )
            publication = Publication.of(
                "wkr-1", "dsp-1", self.task_id, "TASK_SUCCEEDED", payload
            )
            assert publication is not None
            self.runtime.answer_publication(self.task_id, publication)


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


def test_the_guard_carries_through_a_held_write_until_it_lands(rig) -> None:
    rig.dispatch()
    rig.store.failure = ConnectionError("store down")
    rig.succeed(rig.payload())
    assert rig.store.guards[-1] == CommitGuard(rig.epoch)
    assert rig.answers == []

    rig.store.failure = None
    rig.runtime._retry_durability(rig.workflow_id)
    # The repayment is a later, ordinary write: it still carries the guard.
    assert rig.store.guards[-1] == CommitGuard(rig.epoch)
    assert [a.disposition.value for a in rig.answers] == ["committed"]

    before = len(rig.store.guards)
    rig.runtime.cancel_workflow(rig.workflow_id)
    assert all(guard is None for guard in rig.store.guards[before:])


def test_a_lost_watch_under_the_same_epoch_holds_the_write(rig) -> None:
    rig.dispatch()
    rig.store.failure = GuardInterrupted("watch lost")
    rig.succeed(rig.payload())
    assert rig.fenced == []
    assert rig.answers == []
    rig.store.failure = None
    rig.runtime._retry_durability(rig.workflow_id)
    assert [a.disposition.value for a in rig.answers] == ["committed"]


def test_a_later_root_stops_this_one_before_it_acts(rig) -> None:
    rig.dispatch()
    rig.store.failure = EpochFenced("a later root took the epoch")
    with pytest.raises(Exception):
        rig.succeed(rig.payload())
    assert rig.fenced == [True]
    # Nothing it owed is delivered, and it commits nothing more.
    rig.runtime._act_after_commit()
    assert rig.answers == []
    rig.store.failure = None
    with pytest.raises(EpochFenced):
        rig.runtime.cancel_workflow(rig.workflow_id)
    assert rig.fenced == [True]
    # Not even an answer that owes no write goes out.
    publication = Publication.of(
        "wkr-1", "dsp-1", "tsk-gone", "TASK_SUCCEEDED", rig.payload()
    )
    assert publication is not None
    rig.runtime.answer_publication("tsk-gone", publication)
    assert rig.answers == []


def test_a_write_that_faults_keeps_its_scope_out_of_collection(rig) -> None:
    rig.dispatch()
    record = rig.runtime.get_record(rig.task_id)
    assert record is not None
    rig.store.failure = ResponseError("WRONGTYPE Operation against a key")
    with pytest.raises(ResponseError):
        rig.succeed(rig.payload())
    assert rig.store.uncovered == {(record.org_id, rig.workflow_id)}

    rig.store.failure = None
    rig.runtime.cancel_workflow(rig.workflow_id)
    assert rig.store.uncovered == set()


def test_the_publication_check_reads_an_early_dispatch_without_recording_it(
    rig,
) -> None:
    assert pop_ready(rig.runtime, 0.05) == rig.task_id
    worker = cast(Worker, SimpleNamespace(id="wkr-1", node_id="nde-1"))
    assert (
        rig.runtime.begin_publish(rig.task_id, worker, "dsp-1") is PublishGate.PUBLISH
    )
    commits = len(rig.store.guards)
    assert (
        rig.runtime.binding_workflow(rig.task_id, "wkr-1", "dsp-1") == rig.workflow_id
    )
    record = rig.runtime.get_record(rig.task_id)
    assert record is not None and record.status == TaskStatus.PENDING
    assert len(rig.store.guards) == commits
    assert rig.runtime.binding_workflow(rig.task_id, "wkr-2", "dsp-1") is None
