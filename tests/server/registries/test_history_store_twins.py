"""The history store's sync and async twins answer alike over one store."""

import asyncio
from collections.abc import Callable
from typing import Any

import fakeredis
import pytest

from server.clients.redis import WORKFLOWS_SET_KEY
from server.registries.workflow import WorkflowRegistry
from server.task.history import OutputHistory, OutputsHistory
from server.task.models import TaskStatus
from server.task.v2.representations.results import CardinalityKind, ResultDeclaration
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.registries.test_history_store import _publication, _publish
from tests.server.test_workflow_sources import _V1, _register, _runtime


def _published() -> tuple[WorkflowRegistry, str, list[str], str]:
    """A store holding a freed workflow with published outputs and an open one."""
    registry = WorkflowRegistry(fake_redis_client(fakeredis.FakeServer()))
    runtime = _runtime(registry)
    freed, task_ids = _register(runtime, _V1)
    running, _ = _register(runtime, _V1)
    publication = _publication(registry, freed, task_ids)
    outputs = OutputsHistory(
        declarations=[
            (
                "out",
                ResultDeclaration(
                    output_id="o",
                    source_ref="out",
                    cardinality=CardinalityKind.KEYED_COLLECTION,
                ),
            )
        ],
        members=[OutputHistory(name="out", scope_id="scp-1", key=str(k)) for k in "01"],
    )
    publication = type(publication)(
        workflow_id=freed,
        certificate=publication.certificate,
        tasks=publication.tasks,
        members=publication.members,
        outputs=outputs,
    )
    assert _publish(registry, publication, registry.workflow_revision(freed)) == "ok"
    registry.set_history_index_state("building:v1:7")
    return registry, freed, task_ids, running


def _task_id(stored: Any) -> str:
    return stored.persisted.record.task_id


def _reads(freed: str, task_ids: list[str], running: str) -> dict[str, Any]:
    """Each read of the history store, with the arguments it is asked with."""
    return {
        "workflow_revision": (freed,),
        "history_certificate": (freed,),
        "history_certified": ([freed, running, "wfl-unknown"],),
        "stored_outputs": (freed,),
        "stored_outputs:named": (freed, "out"),
        "stored_output": (freed, "out", "scp-1", "1", None),
        "history_members": ([*task_ids, "tsk-unknown"],),
        "load_history": ([*task_ids, "tsk-unknown"],),
        "history_range": (None, None, False, 100),
        "history_range:status": (TaskStatus.DONE, None, True, 1),
        "history_task_ids": (freed,),
        "task_history_page": (_task_id, 2, None, False),
        "task_history_page:status": (_task_id, 10, None, True, {TaskStatus.DONE}),
        "task_history_page:workflow": (_task_id, 10, None, False, None, [freed]),
        "task_history_page:tasks": (_task_id, 1, None, True, None, None, task_ids),
        "history_counts": (),
        "history_census": (),
        "history_seq": (),
        "open_workflow_ids": (),
        "history_index_state": (),
        "scan_workflow_ids": (0, 100),
        "stored_task_counts": ([freed, running, "wfl-unknown"],),
    }


@pytest.mark.parametrize("read", list(_reads("", [], "")))
def test_each_history_read_answers_alike_sync_and_async(read: str) -> None:
    registry, freed, task_ids, running = _published()
    name = read.partition(":")[0]
    args = _reads(freed, task_ids, running)[read]

    sync = getattr(registry, name)(*args)
    asynchronous = asyncio.run(getattr(registry, f"{name}_async")(*args))

    if name == "scan_workflow_ids":
        sync, asynchronous = (
            (cursor, sorted(ids)) for cursor, ids in (sync, asynchronous)
        )
    assert sync == asynchronous


def _reopen_and_close(
    registry: WorkflowRegistry, freed: str, running: str, use_async: bool
) -> Any:
    if use_async:
        asyncio.run(registry.reopen_workflows_async(freed))
        return asyncio.run(registry.close_workflows_async(running))
    registry.reopen_workflows(freed)
    return registry.close_workflows(running)


def _index_state(
    registry: WorkflowRegistry, freed: str, running: str, use_async: bool
) -> Any:
    if use_async:
        return asyncio.run(registry.set_history_index_state_async("complete:v1"))
    return registry.set_history_index_state("complete:v1")


def _settle_open_set(
    registry: WorkflowRegistry, freed: str, running: str, use_async: bool
) -> Any:
    raw_control(registry).srem(WORKFLOWS_SET_KEY, running)
    if use_async:
        return asyncio.run(registry.settle_open_set_async())
    return registry.settle_open_set()


_WRITES: dict[str, Callable[[WorkflowRegistry, str, str, bool], Any]] = {
    "reopen and close": _reopen_and_close,
    "index state": _index_state,
    "settle open set": _settle_open_set,
}


@pytest.mark.parametrize("write", list(_WRITES))
def test_each_history_write_leaves_the_store_alike_sync_and_async(write: str) -> None:
    stores = []
    for use_async in (False, True):
        registry, freed, _, running = _published()
        answer = _WRITES[write](registry, freed, running, use_async)
        stores.append(
            (
                answer,
                sorted(
                    {freed: "freed", running: "running"}[w]
                    for w in registry.open_workflow_ids()
                ),
                registry.history_census(),
                registry.history_index_state(),
            )
        )
    assert stores[0] == stores[1]


def test_a_publication_prepares_and_certifies_alike_sync_and_async() -> None:
    answers = []
    for use_async in (False, True):
        registry = WorkflowRegistry(fake_redis_client(fakeredis.FakeServer()))
        workflow_id, task_ids = _register(_runtime(registry), _V1)
        publication = _publication(registry, workflow_id, task_ids)
        revision = registry.workflow_revision(workflow_id)
        if use_async:
            prepared = asyncio.run(registry.prepare_history_async(publication))
            certified = asyncio.run(
                registry.certify_history_async(publication, revision)
            )
            again = asyncio.run(registry.prepare_history_async(publication))
        else:
            prepared = registry.prepare_history(publication)
            certified = registry.certify_history(publication, revision)
            again = registry.prepare_history(publication)
        answers.append(
            (prepared, certified, again, registry.history_counts(), len(task_ids))
        )
    assert answers[0] == answers[1]
    assert answers[0][:3] == (True, "ok", False)
