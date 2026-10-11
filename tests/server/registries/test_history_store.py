"""A freed workflow's history is certified only over a complete preparation."""

from typing import Any

import fakeredis
import pytest

from server.clients.redis import (
    TASK_HISTORY_MEMBERS_KEY,
    WORKFLOWS_OPEN_KEY,
    task_history_key,
    workflow_history_key,
)
from server.registries.history import HistoryNotPrepared, HistoryPublication
from server.registries.workflow import WorkflowRegistry
from server.task.history import (
    CERTIFICATE_FIELD,
    HISTORY_SCHEMA,
    HistoryCertificate,
    TaskHistory,
    history_member,
    member_task_id,
)
from server.task.models import TaskStatus
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.test_workflow_sources import _V1, _register, _runtime


def _registered() -> tuple[WorkflowRegistry, str, list[str]]:
    registry = WorkflowRegistry(fake_redis_client(fakeredis.FakeServer()))
    workflow_id, task_ids = _register(_runtime(registry), _V1)
    return registry, workflow_id, task_ids


def _publication(
    registry: WorkflowRegistry, workflow_id: str, task_ids: list[str], seq: int = 1
) -> HistoryPublication:
    stored = registry.load_task_states(workflow_id, *task_ids)
    members: dict[str, tuple[str, str]] = {
        item.record.task_id: (
            history_member(item.record.submitted_ts, item.record.task_id),
            TaskStatus.DONE,
        )
        for item in stored
        if item is not None
    }
    return HistoryPublication(
        workflow_id=workflow_id,
        certificate=HistoryCertificate(
            schema_version=HISTORY_SCHEMA,
            nonce="nonce",
            seq=seq,
            mode="v1",
            tasks=len(task_ids),
        ),
        tasks={task_id: TaskHistory(seq=seq, completed=True) for task_id in task_ids},
        members=members,
        outputs=None,
    )


def _publish(
    registry: WorkflowRegistry, publication: HistoryPublication, revision: int
) -> str:
    registry.prepare_history(publication)
    return registry.certify_history(publication, revision)


def _certified(registry: WorkflowRegistry, workflow_id: str) -> bool:
    raw = raw_control(registry)
    key = workflow_history_key(workflow_id)
    return raw.type(key) == "hash" and raw.hexists(key, CERTIFICATE_FIELD)


def _is_open(registry: WorkflowRegistry, workflow_id: str) -> bool:
    return bool(raw_control(registry).sismember(WORKFLOWS_OPEN_KEY, workflow_id))


def test_a_certified_workflow_reads_back_and_leaves_the_open_set() -> None:
    registry, workflow_id, task_ids = _registered()
    publication = _publication(registry, workflow_id, task_ids)

    assert _publish(registry, publication, registry.workflow_revision(workflow_id)) == (
        "ok"
    )

    assert registry.history_certificate(workflow_id) == publication.certificate
    assert not _is_open(registry, workflow_id)
    assert registry.history_counts() == (len(task_ids), len(task_ids))
    stored = registry.load_history(task_ids)
    assert [entry.persisted.record.task_id for entry in stored if entry] == task_ids
    members = [
        member
        for chunk in registry._history_scan(None, iter([100]), None, None)
        for member in chunk
    ]
    assert [member_task_id(member) for member in members] == task_ids[::-1]


def test_a_write_after_the_revision_was_read_refuses_certification() -> None:
    registry, workflow_id, task_ids = _registered()
    revision = registry.workflow_revision(workflow_id)
    publication = _publication(registry, workflow_id, task_ids)
    stored = registry.load_task_states(workflow_id, task_ids[0])
    registry.commit_transition(workflow_id, records=[item for item in stored if item])

    assert _publish(registry, publication, revision) == "revision changed"

    assert registry.history_certificate(workflow_id) is None
    assert _is_open(registry, workflow_id)
    assert registry.history_counts() == (0, 0)
    assert registry.load_history(task_ids) == [None] * len(task_ids)


@pytest.mark.parametrize(
    "key",
    [workflow_history_key("{w}"), TASK_HISTORY_MEMBERS_KEY, task_history_key(None)],
)
def test_a_failed_preparation_certifies_nothing(key: str) -> None:
    registry, workflow_id, task_ids = _registered()
    raw = raw_control(registry)
    raw.set(key.format(w=workflow_id), "wrong type")
    publication = _publication(registry, workflow_id, task_ids)

    with pytest.raises(HistoryNotPrepared):
        registry.prepare_history(publication)
    assert registry.certify_history(
        publication, registry.workflow_revision(workflow_id)
    ) in {"projection missing", "index type", "index missing", "index incomplete"}

    assert not _certified(registry, workflow_id)
    assert _is_open(registry, workflow_id)
    assert registry.history_counts() == (0, 0)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (
            lambda raw, w, t: raw.hdel(workflow_history_key(w), t),
            "projection incomplete",
        ),
        (lambda raw, w, t: raw.hdel(TASK_HISTORY_MEMBERS_KEY, t), "index incomplete"),
        (lambda raw, w, t: raw.delete(task_history_key("DONE")), "index missing"),
    ],
)
def test_an_incomplete_preparation_is_refused_and_writes_nothing(
    damage: Any, reason: str
) -> None:
    registry, workflow_id, task_ids = _registered()
    publication = _publication(registry, workflow_id, task_ids)
    registry.prepare_history(publication)
    damage(raw_control(registry), workflow_id, task_ids[0])

    assert (
        registry.certify_history(publication, registry.workflow_revision(workflow_id))
        == reason
    )

    assert registry.history_certificate(workflow_id) is None
    assert _is_open(registry, workflow_id)
    assert registry.history_counts() == (0, 0)


def test_an_uncertified_preparation_reads_as_not_freed() -> None:
    registry, workflow_id, task_ids = _registered()
    registry.prepare_history(_publication(registry, workflow_id, task_ids))

    assert registry.load_history(task_ids) == [None] * len(task_ids)
    assert registry.history_counts() == (0, 0)


def test_a_repeated_certification_is_refused_and_counts_once() -> None:
    registry, workflow_id, task_ids = _registered()
    publication = _publication(registry, workflow_id, task_ids)
    revision = registry.workflow_revision(workflow_id)
    assert _publish(registry, publication, revision) == "ok"

    assert registry.certify_history(publication, revision) == "certified"

    assert registry.history_counts() == (len(task_ids), len(task_ids))


def test_the_stored_sequence_never_goes_backwards() -> None:
    registry, first, first_tasks = _registered()
    second, second_tasks = _register(_runtime(registry), _V1)
    later = _publication(registry, second, second_tasks, seq=6)
    earlier = _publication(registry, first, first_tasks, seq=5)

    assert _publish(registry, later, registry.workflow_revision(second)) == "ok"
    assert _publish(registry, earlier, registry.workflow_revision(first)) == "ok"

    assert registry.history_seq() == 6


def test_history_members_order_by_submission_then_id() -> None:
    timestamps = [-1.5, -0.0, 0.0, 1e-300, 1.0, 1.0, 2.5, 1.7e9, 1.7e9 + 1e-3]
    keys = [(ts, f"tsk-{i % 3}") for i, ts in enumerate(timestamps)]
    members = [history_member(ts, task_id) for ts, task_id in keys]

    assert sorted(members) == [history_member(*key) for key in sorted(keys)]
