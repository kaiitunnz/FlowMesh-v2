"""The records of a node's provisioned workers and what their recovery owes."""

from unittest.mock import MagicMock

import pytest

from server.supervisor import provisioning
from server.supervisor.provisioning import (
    DockerHandle,
    Due,
    ProvisionedWorkers,
    RecordState,
    RunState,
    WorkerProvisioningStore,
    WorkerRecord,
)
from tests.server.supervisor_helpers import memory_store, worker_record

_HANDLE = DockerHandle(container_id="c1", container_name="w1")


def _provisioned(
    *records: WorkerRecord,
) -> tuple[ProvisionedWorkers, WorkerProvisioningStore]:
    store = memory_store()
    provisioned = ProvisionedWorkers(store)
    for record in records:
        provisioned.create(record)
    return provisioned, store


def _stored(store: WorkerProvisioningStore) -> dict[str, WorkerRecord]:
    return {record.alias: record for record in store.load()}


@pytest.mark.asyncio
async def test_load_retries_until_the_store_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = memory_store()
    store.create(worker_record("w1"))
    load, failures = store.load, iter([ConnectionError("down")] * 2)
    sleeps: list[float] = []

    def flaky_load() -> list[WorkerRecord]:
        if (error := next(failures, None)) is not None:
            raise error
        return load()

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(store, "load", flaky_load)
    monkeypatch.setattr(provisioning.asyncio, "sleep", sleep)
    provisioned = ProvisionedWorkers(store)

    assert [r.alias for r in await provisioned.load()] == ["w1"]
    assert sleeps == [1.0, 2.0] and "w1" in provisioned


def test_create_refuses_an_alias_with_a_record() -> None:
    provisioned, store = _provisioned(worker_record("w1", token="first"))

    with pytest.raises(ValueError, match="already exists"):
        provisioned.create(worker_record("w1", token="second"))

    assert _stored(store)["w1"].token.get_secret_value() == "first"


def test_a_strict_update_the_store_refuses_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioned, store = _provisioned(worker_record("w1"))
    monkeypatch.setattr(store, "put", MagicMock(side_effect=ConnectionError))

    with pytest.raises(ConnectionError):
        provisioned.update("w1", run_state=RunState.STOPPED)

    record = provisioned.get("w1")
    assert record is not None and record.run_state is RunState.RUNNING


def test_a_lenient_update_the_store_refuses_is_saved_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioned, store = _provisioned(worker_record("w1"))
    put = store.put
    monkeypatch.setattr(store, "put", MagicMock(side_effect=ConnectionError))

    provisioned.update("w1", strict=False, run_state=RunState.STOPPED)
    record = provisioned.get("w1")
    assert record is not None and record.run_state is RunState.STOPPED
    assert _stored(store)["w1"].run_state is RunState.RUNNING

    monkeypatch.setattr(store, "put", put)
    provisioned.retry_unsaved()
    assert _stored(store)["w1"].run_state is RunState.STOPPED


@pytest.mark.parametrize(
    "state, handle, expected",
    [
        (RecordState.PROVISIONING, _HANDLE, RecordState.PRESENT),
        (RecordState.PROVISIONING, None, RecordState.PROVISIONING),
        (RecordState.REMOVING, _HANDLE, RecordState.REMOVING),
    ],
)
def test_only_a_launch_that_reports_a_handle_leaves_provisioning(
    state: RecordState, handle: DockerHandle | None, expected: RecordState
) -> None:
    provisioned, store = _provisioned(worker_record("w1", state=state))

    provisioned.handle_committer("w1")(handle)

    assert (_stored(store)["w1"].state, _stored(store)["w1"].handle) == (
        expected,
        handle,
    )


@pytest.mark.parametrize(
    "handle, expected",
    [(None, RecordState.PRESENT), (_HANDLE, RecordState.PROVISIONING)],
)
def test_a_launch_that_ended_without_a_handle_is_present(
    handle: DockerHandle | None, expected: RecordState
) -> None:
    provisioned, store = _provisioned(
        worker_record("w1", state=RecordState.PROVISIONING)
    )

    provisioned.launch_ended("w1", handle)

    assert _stored(store)["w1"].state is expected


def test_a_record_the_store_cannot_delete_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioned, store = _provisioned(worker_record("w1"))
    monkeypatch.setattr(store, "delete", MagicMock(side_effect=ConnectionError))

    provisioned.forget("w1")
    assert "w1" in provisioned

    monkeypatch.undo()
    provisioned.forget("w1")
    assert "w1" not in provisioned and _stored(store) == {}


def test_each_record_is_owed_what_its_state_calls_for() -> None:
    provisioned, _ = _provisioned(
        worker_record("removing", state=RecordState.REMOVING),
        worker_record("silent", handle=_HANDLE),
        worker_record("stopping", run_state=RunState.STOPPED, handle=_HANDLE),
        worker_record("stopped", run_state=RunState.STOPPED),
        worker_record("running", handle=_HANDLE),
    )
    provisioned.expect("silent")
    provisioned.open_grace(100.0)

    assert provisioned.due(99.0) == [
        ("removing", Due.REMOVE),
        ("stopping", Due.FINISH_STOP),
    ]
    for alias in ("removing", "stopping"):
        provisioned.settled(alias)
    assert provisioned.due(100.0) == [
        ("removing", Due.REMOVE),
        ("silent", Due.EXPIRE),
        ("stopping", Due.FINISH_STOP),
    ]


def test_no_grace_expires_before_it_opens() -> None:
    provisioned, _ = _provisioned(worker_record("silent", handle=_HANDLE))
    provisioned.expect("silent")

    assert provisioned.due(1e9) == []


def test_what_is_due_stays_claimed_until_settled() -> None:
    provisioned, _ = _provisioned(
        worker_record("w1", state=RecordState.REMOVING, handle=_HANDLE)
    )

    assert provisioned.due(0.0) == [("w1", Due.REMOVE)]
    assert provisioned.due(0.0) == []
    provisioned.settled("w1")
    assert provisioned.due(0.0) == [("w1", Due.REMOVE)]


def test_an_operation_ends_the_grace_and_keeps_the_heartbeat_off() -> None:
    provisioned, _ = _provisioned(worker_record("w1", handle=_HANDLE))
    provisioned.expect("w1")
    provisioned.open_grace(0.0)

    with provisioned.operating("w1"):
        provisioned.update("w1", run_state=RunState.STOPPED)
        assert provisioned.due(1.0) == []
    assert not provisioned.awaiting("w1")
    assert provisioned.due(1.0) == [("w1", Due.FINISH_STOP)]


def test_an_operation_inside_another_keeps_the_heartbeat_off() -> None:
    provisioned, _ = _provisioned(
        worker_record("w1", state=RecordState.REMOVING, handle=_HANDLE)
    )

    with provisioned.operating("w1"):
        with provisioned.operating("w1"):
            pass
        assert provisioned.due(0.0) == []
    assert provisioned.due(0.0) == [("w1", Due.REMOVE)]


def test_worker_ids_are_the_recorded_registrations() -> None:
    provisioned, _ = _provisioned(
        worker_record("w1", worker_id="wkr-1"), worker_record("w2")
    )

    assert provisioned.worker_ids() == ["wkr-1"]
