"""A task record's internal state: stored as one unit, served by no API, and owned by
each copy of the record."""

import hashlib
import json
from typing import Any

from server.registries.task_states import PersistedTask, load_task_state
from server.task.models import (
    TaskInfo,
    TaskInternalState,
    TaskRecord,
    TaskRecordRuntimeState,
    TaskStatus,
)
from shared.content import ContentReference
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind


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


def _record(status: str = TaskStatus.DISPATCHED, **internal: object) -> TaskRecord:
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


def _stored_internal(record: TaskRecord) -> dict[str, object]:
    stored = json.loads(PersistedTask(record=record).model_dump_json())
    internal: dict[str, object] = stored["record"]["internal"]
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


def test_a_copy_of_a_record_owns_its_internal_state() -> None:
    record = _record(failed_workers=["wkr-1"])
    record.internal.runtime.unwritten_artifacts.append(_file())

    copied = record.model_copy(update={"task_id": "tsk-2"})
    copied.internal.failed_workers.append("wkr-2")
    copied.internal.runtime.unwritten_artifacts.clear()
    copied.internal.dispatch_id = "dsp-2"
    record.internal.failed_workers.append("wkr-3")

    assert record.internal.failed_workers == ["wkr-1", "wkr-3"]
    assert record.internal.runtime.unwritten_artifacts == [_file()]
    assert record.internal.dispatch_id is None
    assert copied.internal.failed_workers == ["wkr-1", "wkr-2"]
