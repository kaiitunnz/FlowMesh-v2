"""A task record's internal state: stored as one unit, served by no API, and owned by
each copy of the record."""

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from server.registries.task_states import PersistedTask, load_task_state
from server.task.models import (
    TaskInfo,
    TaskInternalState,
    TaskRecord,
    TaskRecordRuntimeState,
    TaskStatus,
)
from server.task.runtime import TaskRuntime
from shared.content import ContentReference
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from tests.server.task.test_v2_orchestration import FakeRegistry, _live_runtime


def _ref(data: bytes) -> ContentReference:
    return ContentReference(
        authorization_scope="org",
        content_digest=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _file() -> ArtifactBinding:
    return ArtifactBinding(
        name="weights",
        kind=ArtifactKind.FILE,
        reference=_ref(b"weights"),
        size_bytes=7,
    )


def _record(status: str = TaskStatus.DISPATCHED, **internal: Any) -> TaskRecord:
    return TaskRecord.model_validate(
        {
            "task_id": "tsk-1",
            "workflow_id": "wfl-1",
            "owner_id": "owner",
            "raw_yaml": "source",
            "task": {
                "apiVersion": "flowmesh/v1",
                "kind": "Task",
                "spec": {"taskType": "echo"},
            },
            "status": status,
            "internal": internal,
        }
    )


def _round_trip(record: TaskRecord) -> TaskRecord:
    persisted = PersistedTask(record=record)
    return load_task_state(
        persisted.model_dump_json(), {record.source_digest(): record.raw_yaml}
    ).record


def _stored_internal(record: TaskRecord) -> dict[str, Any]:
    stored = json.loads(PersistedTask(record=record).model_dump_json())
    internal: dict[str, Any] = stored["record"]["internal"]
    return internal


def test_every_internal_field_but_the_runtime_state_is_stored() -> None:
    stored = _stored_internal(_record())

    assert set(stored) == set(TaskInternalState.model_fields) - {"runtime"}


def test_the_internal_state_survives_a_restart() -> None:
    internal: dict[str, Any] = {
        "dispatch_id": "dsp-1",
        "first_started_ts": 1.0,
        "failed_workers": ["wkr-1"],
        "merged_dispatch_worker": "wkr-2",
        "latest_update_dispatch_id": "dsp-0",
        "result_reference": _ref(b"result"),
        "result_skip": {"reason": "skipped"},
        "has_artifacts": True,
        "dispatched_artifacts": (_file(),),
        "dispatched_producers": ("tsk-0",),
        "credential_refs": {},
    }
    record = _record(**internal)

    restored = _round_trip(record)

    assert restored.internal == record.internal


def test_the_runtime_state_is_not_stored() -> None:
    record = _record()
    record.internal.runtime = TaskRecordRuntimeState(
        no_eligible_since=1.0,
        avoided_worker="wkr-1",
        succeeded_monotonic=2.0,
        members_pending=True,
        unwritten_artifacts=[_file()],
        artifacts={("final", "weights"): _file()},
    )

    assert _round_trip(record).internal.runtime == TaskRecordRuntimeState()


def test_a_settled_task_stores_no_dispatched_artifacts_and_keeps_its_own() -> None:
    record = _record(
        TaskStatus.DONE,
        dispatched_artifacts=(_file(),),
        dispatched_producers=("tsk-0",),
    )

    stored = _stored_internal(record)

    assert "dispatched_artifacts" not in stored
    assert "dispatched_producers" not in stored
    assert record.internal.dispatched_artifacts == (_file(),)
    assert record.internal.dispatched_producers == ("tsk-0",)


def test_the_internal_state_is_served_by_no_api() -> None:
    record = _record(dispatch_id="dsp-1", failed_workers=["wkr-1"])
    info = TaskInfo(
        **dict(record),
        depends_on=[],
        pending_dependencies=[],
        dependents=[],
        completed=False,
        failed=False,
    )

    assert "internal" not in info.model_dump()
    assert (
        "internal" not in TaskInfo.model_json_schema(mode="serialization")["properties"]
    )
    assert info.model_dump()["last_failed_worker"] == "wkr-1"


def test_a_copy_of_a_record_owns_its_internal_state_and_runtime_state() -> None:
    record = _record(dispatch_id="dsp-1")
    record.internal.runtime.avoided_worker = "wkr-1"

    copied = copy.copy(record)
    copied.internal.dispatch_id = "dsp-2"
    copied.internal.runtime.avoided_worker = "wkr-2"
    record.internal.first_started_ts = 1.0

    assert (record.internal.dispatch_id, copied.internal.dispatch_id) == (
        "dsp-1",
        "dsp-2",
    )
    assert record.internal.runtime.avoided_worker == "wkr-1"
    assert copied.internal.runtime.avoided_worker == "wkr-2"
    assert copied.internal.first_started_ts is None


def _mutate(record: TaskRecord, worker: str) -> None:
    assert record.selected_worker is not None
    record.selected_worker.append(worker)
    record.internal.failed_workers.append(worker)
    record.internal.runtime.unwritten_artifacts.append(_file())
    record.internal.dispatch_id = worker


def _assert_unmutated(record: TaskRecord) -> None:
    assert record.selected_worker == ["wkr-0"]
    assert record.internal.failed_workers == ["wkr-0"]
    assert record.internal.runtime.unwritten_artifacts == []
    assert record.internal.dispatch_id == "dsp-1"


def _source() -> TaskRecord:
    source = _record(dispatch_id="dsp-1", failed_workers=["wkr-0"])
    source.selected_worker = ["wkr-0"]
    return source


def test_a_detached_record_owns_its_containers_both_ways() -> None:
    record = _source()

    detached = record.detached()
    _mutate(detached, "wkr-1")
    _assert_unmutated(record)

    detached = record.detached()
    _mutate(record, "wkr-2")
    _assert_unmutated(detached)


def _runtime_holding(record: TaskRecord) -> TaskRuntime:
    runtime = _live_runtime(FakeRegistry())
    runtime._tasks[record.task_id] = record
    return runtime


def test_a_task_record_snapshot_is_detached_both_ways() -> None:
    record = _source()
    runtime = _runtime_holding(record)

    [snapshot] = runtime.task_records()
    _mutate(snapshot, "wkr-1")
    _assert_unmutated(record)

    [snapshot] = runtime.task_records()
    _mutate(record, "wkr-2")
    _assert_unmutated(snapshot)


def test_a_live_task_info_is_detached_both_ways() -> None:
    record = _source()
    runtime = _runtime_holding(record)

    info = runtime.describe_task(record.task_id)
    assert info is not None
    _mutate(info, "wkr-1")
    _assert_unmutated(record)

    info = runtime.describe_task(record.task_id)
    assert info is not None
    _mutate(record, "wkr-2")
    _assert_unmutated(info)


def test_a_stored_task_info_is_detached_both_ways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _source()
    runtime = _live_runtime(FakeRegistry())
    history = SimpleNamespace(
        info_fields=lambda: {
            "depends_on": [],
            "pending_dependencies": [],
            "dependents": [],
            "completed": False,
            "failed": False,
        }
    )
    stored = SimpleNamespace(persisted=PersistedTask(record=record), history=history)
    monkeypatch.setattr(runtime, "_stored_task", lambda _task_id: stored)

    info = runtime.describe_task(record.task_id)
    assert info is not None
    _mutate(info, "wkr-1")
    _assert_unmutated(record)

    info = runtime.describe_task(record.task_id)
    assert info is not None
    _mutate(record, "wkr-2")
    _assert_unmutated(info)


def test_a_wait_clock_starts_once_and_reports_the_time_since() -> None:
    runtime = TaskRecordRuntimeState()

    assert runtime.no_eligible_for(10.0) == 0.0
    assert runtime.no_eligible_for(12.5) == 2.5
    assert runtime.no_dispatch_for(11.0) == 0.0
    assert runtime.no_dispatch_for(14.0) == 3.0
    assert (runtime.no_eligible_since, runtime.no_dispatch_since) == (10.0, 11.0)
