"""A restart restores only the workflows not freed, and a store kept from before the
open-workflow index existed is classified into it once, resumably."""

import asyncio
import logging
from collections.abc import Callable, Collection
from typing import Any, cast

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from server.clients.redis import (
    HISTORY_INDEX_KEY,
    WORKFLOWS_OPEN_KEY,
    WORKFLOWS_SET_KEY,
    workflow_log_closed_key,
    workflow_log_stream_key,
    workflow_sched_key,
)
from server.config import OrchestrationConfig
from server.orchestration.telemetry import NULL_WORKFLOW_SPAN_EMITTER
from server.registries.history import HistoryPublication
from server.registries.workflow import WorkflowRegistry
from server.task.finalizer import CloseOutcome, WorkflowFinalizer
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from server.task.runtime.history_bootstrap import INDEX_COMPLETE
from server.task.workflow_retry import WorkflowRetryScheduler
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.redis_helpers import raw_control
from tests.server.result_store import make_result_reader, result_payload
from tests.server.runtime_helpers import manual_durability_retry
from tests.server.task.test_settled_history import (
    _TS,
    CHAIN,
    _certified,
    _register,
    _registry,
    _resident,
    _runtime,
    _succeed_ready,
    _Workers,
)
from tests.server.task.test_settled_history_reads import (
    _described,
    _fail_first,
    _page,
    _query,
)
from tests.server.task.test_v2_orchestration import LINEAR
from tests.support.waiting import pop_ready

_LOGGER = logging.getLogger("test.history_bootstrap")


def _unstarted(registry: WorkflowRegistry) -> TaskRuntime:
    return TaskRuntime(
        registry,
        cast(Any, _Workers()),
        OrchestrationConfig(),
        make_result_reader(),
        _LOGGER,
        credential_vault=InMemoryCredentialVault(),
        redrive=lambda fire, logger: StoreRedriveScheduler(
            fire, logger, run_thread=False
        ),
        durability_retry=manual_durability_retry,
    )


def _close(runtime: TaskRuntime, workflow_id: str) -> bool:
    finalizer = WorkflowFinalizer(
        cast(Any, runtime._workflow_registry)._rds.sync,
        runtime,
        _LOGGER,
        NULL_WORKFLOW_SPAN_EMITTER,
        retry=lambda fire, logger: WorkflowRetryScheduler(
            fire, logger, run_thread=False
        ),
    )
    return finalizer.close(workflow_id) in (
        CloseOutcome.CLOSED,
        CloseOutcome.ALREADY_CLOSED,
    )


def _never_closed(runtime: TaskRuntime, workflow_id: str) -> bool:
    return False


class _Loads:
    """Records which workflows a runtime loads from the store."""

    def __init__(self, runtime: TaskRuntime) -> None:
        self.ids: list[str] = []
        load = runtime._load_stored_workflow

        async def recorded(workflow_id: str) -> Any:
            self.ids.append(workflow_id)
            return await load(workflow_id)

        cast(Any, runtime)._load_stored_workflow = recorded


def _restarted(
    registry: WorkflowRegistry,
    close: Callable[[TaskRuntime, str], bool] = _never_closed,
    claims_terminal: Callable[[Collection[str]], bool] | None = None,
    archived: Callable[[str], bool | None] | None = None,
) -> tuple[TaskRuntime, _Loads]:
    runtime = _unstarted(registry)
    loads = _Loads(runtime)
    asyncio.run(runtime.restart(close, claims_terminal, archived))
    return runtime, loads


def _scans(registry: WorkflowRegistry) -> list[int]:
    """Record the cursor of each scan over the registered workflows."""
    scanned: list[int] = []
    scan = registry.scan_workflow_ids_async

    async def recorded(cursor: int, count: int) -> tuple[int, list[str]]:
        scanned.append(cursor)
        return await scan(cursor, count)

    cast(Any, registry).scan_workflow_ids_async = recorded
    return scanned


def _before_the_index(registry: WorkflowRegistry) -> None:
    """Make the store look as one written before the open-workflow index existed."""
    raw_control(registry).delete(WORKFLOWS_OPEN_KEY, HISTORY_INDEX_KEY)


def _open(registry: WorkflowRegistry) -> set[str]:
    return registry.open_workflow_ids()


def _reads(runtime: TaskRuntime, task_ids: list[str]) -> list[Any]:
    """What the settled tasks describe as, and every listing and count."""
    return [
        _described(runtime, task_ids),
        [_page(runtime, _query(), limit) for limit in (1, 4, 100)],
        _page(runtime, _query(status="DONE"), 100),
        runtime.task_status_counts(),
    ]


def _mixed_store(
    registry: WorkflowRegistry, each: int
) -> tuple[TaskRuntime, dict[str, set[str]]]:
    """A store of ``each`` succeeded, failed, cancelled and running workflows."""
    runtime = _runtime(registry)
    kinds: dict[str, set[str]] = {}
    kinds["succeeded"] = {_register(runtime, CHAIN) for _ in range(each)}
    _succeed_ready(runtime)
    failed: set[str] = set()
    for _ in range(each):
        failed.add(_register(runtime, CHAIN))
        _fail_first(runtime)
    kinds["failed"] = failed
    kinds["cancelled"] = {_register(runtime, LINEAR) for _ in range(each)}
    for workflow_id in kinds["cancelled"]:
        runtime.cancel_workflow(workflow_id)
    kinds["running"] = {_register(runtime, CHAIN) for _ in range(each)}
    return runtime, kinds


def _settled_task_ids(runtime: TaskRuntime, kinds: dict[str, set[str]]) -> list[str]:
    return sorted(
        task_id
        for kind in ("succeeded", "failed", "cancelled")
        for workflow_id in kinds[kind]
        for task_id in runtime._tasks.ids_of(workflow_id)
    )


def test_a_restart_restores_only_the_open_workflows_and_reads_as_before() -> None:
    registry = _registry()
    runtime, kinds = _mixed_store(registry, 2)
    task_ids = _settled_task_ids(runtime, kinds)
    runtime.settled_history.drain()
    for kind in ("succeeded", "failed", "cancelled"):
        assert not any(_resident(runtime, w) for w in kinds[kind])
    before = _reads(runtime, task_ids)

    restarted, loads = _restarted(registry)

    assert set(loads.ids) == kinds["running"] == _open(registry)
    assert _reads(restarted, task_ids) == before
    assert restarted.settled_history.published_through == registry.history_seq()


@pytest.mark.parametrize("first_lands", [True, False])
def test_a_restart_keeps_the_listing_of_workflows_certified_out_of_order(
    first_lands: bool,
) -> None:
    registry = _registry()
    runtime = _runtime(registry)
    first = _register(runtime, CHAIN)
    second = _register(runtime, CHAIN)
    _succeed_ready(runtime)
    certify = registry.certify_history

    def second_overtakes(publication: HistoryPublication, revision: int) -> str:
        if publication.workflow_id != first:
            return certify(publication, revision)
        assert runtime.settled_history.evaluate(second)
        if not first_lands:
            return "refused"
        certify(publication, revision)
        raise RedisConnectionError("reply lost")

    cast(Any, registry).certify_history = second_overtakes
    runtime.settled_history.evaluate(first)
    cast(Any, registry).certify_history = certify
    assert not _resident(runtime, second)
    assert _resident(runtime, first) is not first_lands
    second_seq = registry.history_seq()
    assert registry.history_certificate(second) is not None
    if first_lands:
        certificate = registry.history_certificate(first)
        assert certificate is not None and certificate.seq < second_seq
    before = [_page(runtime, _query(), limit) for limit in (1, 2, 6, 100)]

    restarted, loads = _restarted(registry)

    assert set(loads.ids) == (set() if first_lands else {first})
    assert restarted.settled_history.published_through == second_seq
    assert [_page(restarted, _query(), limit) for limit in (1, 2, 6, 100)] == before


def test_a_store_kept_from_before_the_index_is_classified_into_it() -> None:
    registry = _registry()
    runtime, kinds = _mixed_store(registry, 30)
    task_ids = _settled_task_ids(runtime, kinds)
    before = _reads(runtime, task_ids)
    _before_the_index(registry)

    restarted, loads = _restarted(registry, _close)

    settled = kinds["succeeded"] | kinds["failed"] | kinds["cancelled"]
    assert all(_certified(registry, w) for w in settled)
    assert _open(registry) == kinds["running"]
    assert registry.history_index_state() == INDEX_COMPLETE
    assert set(loads.ids) == kinds["running"]
    assert _reads(restarted, task_ids) == before
    assert all(
        raw_control(registry).exists(workflow_log_closed_key(w)) for w in settled
    )


def test_a_workflow_already_closed_is_freed_without_closing_it_again() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    closed, reclosed = _register(runtime, CHAIN), _register(runtime, CHAIN)
    _succeed_ready(runtime)
    assert _close(runtime, closed)
    raw = raw_control(registry)
    sealed = raw.xlen(workflow_log_stream_key(closed))
    _before_the_index(registry)

    _restarted(registry, _close)

    assert _certified(registry, closed) and _certified(registry, reclosed)
    assert raw.xlen(workflow_log_stream_key(closed)) == sealed
    assert raw.exists(workflow_log_closed_key(reclosed))


def test_a_crash_while_classifying_resumes_where_it_stopped() -> None:
    registry = _registry()
    _, kinds = _mixed_store(registry, 30)
    _before_the_index(registry)
    calls = 0

    def crashes_late(runtime: TaskRuntime, workflow_id: str) -> bool:
        nonlocal calls
        calls += 1
        if calls == 80:
            raise RuntimeError("root crashed")
        return _close(runtime, workflow_id)

    with pytest.raises(RuntimeError):
        _restarted(registry, crashes_late)
    state = registry.history_index_state()
    assert state is not None and state.startswith("building:v1:")
    scanned = _scans(registry)

    _, loads = _restarted(registry, _close)

    assert scanned[0] == int(state.removeprefix("building:v1:")) != 0
    settled = kinds["succeeded"] | kinds["failed"] | kinds["cancelled"]
    assert all(_certified(registry, w) for w in settled)
    assert _open(registry) == kinds["running"] == set(loads.ids)
    assert registry.history_index_state() == INDEX_COMPLETE


def test_a_second_start_reads_only_the_open_workflows() -> None:
    registry = _registry()
    _, kinds = _mixed_store(registry, 3)
    _before_the_index(registry)
    _restarted(registry, _close)
    scanned = _scans(registry)

    _, loads = _restarted(registry, _close)

    assert scanned == []
    assert set(loads.ids) == kinds["running"]


def test_a_workflow_something_still_needs_joins_the_open_set() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    claimed, archiving, unclosed, freed = (
        _register(runtime, LINEAR),
        _register(runtime, CHAIN),
        _register(runtime, CHAIN),
        _register(runtime, CHAIN),
    )
    _succeed_ready(runtime)
    with runtime._lock:
        claimed_invocations = runtime._reclaim_check_locked(claimed)
    assert claimed_invocations is not None and claimed_invocations.invocation_ids
    archiving_tasks = set(runtime._tasks.ids_of(archiving))
    _before_the_index(registry)

    def close(scratch: TaskRuntime, workflow_id: str) -> bool:
        return workflow_id != unclosed and _close(scratch, workflow_id)

    def claims_terminal(invocation_ids: Collection[str]) -> bool:
        return not set(invocation_ids) & claimed_invocations.invocation_ids

    def archived(task_id: str) -> bool:
        return task_id not in archiving_tasks

    _, loads = _restarted(registry, close, claims_terminal, archived)

    assert _open(registry) == {claimed, archiving, unclosed} == set(loads.ids)
    assert _certified(registry, freed)


def test_a_workflow_that_cannot_be_restored_is_left_to_the_restart() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    broken, healthy = _register(runtime, CHAIN), _register(runtime, CHAIN)
    _succeed_ready(runtime)
    raw_control(registry).set(workflow_sched_key(broken), "not a schedule")
    _before_the_index(registry)
    scratch_failures: list[str] = []

    def close(scratch: TaskRuntime, workflow_id: str) -> bool:
        return _close(scratch, workflow_id)

    original = TaskRuntime._fail_unrestorable_workflow

    async def failing(self: TaskRuntime, workflow_id: str, exc: Exception) -> Any:
        if self._inert:
            scratch_failures.append(workflow_id)
        return await original(self, workflow_id, exc)

    cast(Any, TaskRuntime)._fail_unrestorable_workflow = failing
    try:
        _restarted(registry, close)
    finally:
        cast(Any, TaskRuntime)._fail_unrestorable_workflow = original

    assert scratch_failures == []
    assert broken in _open(registry)
    assert _certified(registry, healthy)


def _freed_once(
    registry: WorkflowRegistry, runtime: TaskRuntime, task_ids: list[str]
) -> None:
    listed = _page(runtime, _query(), 100)
    assert len(listed) == len(set(listed)) and set(listed) == set(task_ids)
    assert registry.history_counts() == (len(task_ids), len(task_ids))
    assert runtime.settled_history.counts_locked() == (len(task_ids), len(task_ids))
    assert runtime.task_status_counts()[-2:] == (len(task_ids), len(task_ids))


def test_a_certification_landing_after_its_answer_was_lost_frees_the_workflow() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    task_ids = _succeed_ready(runtime)
    before = _reads(runtime, task_ids)
    certify = registry.certify_history
    queued: list[tuple[HistoryPublication, int]] = []

    def lost(publication: HistoryPublication, revision: int) -> str:
        queued.append((publication, revision))
        raise RedisConnectionError("reply lost")

    cast(Any, registry).certify_history = lost
    runtime.settled_history.drain()
    assert _resident(runtime, workflow_id)
    assert not runtime.settled_history.sealed_locked(workflow_id)

    assert certify(*queued[0]) == "ok"
    cast(Any, registry).certify_history = certify
    runtime.settled_history.sweep()
    runtime.settled_history.drain()

    assert not _resident(runtime, workflow_id)
    assert workflow_id not in _open(registry)
    _freed_once(registry, runtime, task_ids)
    assert _reads(runtime, task_ids) == before
    restarted, loads = _restarted(registry)
    assert loads.ids == []
    _freed_once(registry, restarted, task_ids)
    assert _reads(restarted, task_ids) == before


def test_a_lost_answer_while_classifying_is_freed_by_the_same_start() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    task_ids = _succeed_ready(runtime)
    before = _reads(runtime, task_ids)
    _before_the_index(registry)
    certify, read_back = registry.certify_history, registry.history_certificate

    def lost(publication: HistoryPublication, revision: int) -> str:
        certify(publication, revision)
        raise RedisConnectionError("reply lost")

    def unreadable(workflow_id: str) -> Any:
        raise RedisConnectionError("store unavailable")

    cast(Any, registry).certify_history = lost
    cast(Any, registry).history_certificate = unreadable
    restarted, loads = _restarted(registry, _close)
    cast(Any, registry).certify_history = certify
    cast(Any, registry).history_certificate = read_back
    assert loads.ids == [workflow_id]
    assert _certified(registry, workflow_id)

    restarted.settled_history.drain()

    assert not _resident(restarted, workflow_id)
    assert workflow_id not in _open(registry)
    _freed_once(registry, restarted, task_ids)
    assert _reads(restarted, task_ids) == before
    again, loads = _restarted(registry)
    assert loads.ids == []
    _freed_once(registry, again, task_ids)


def test_a_workflow_neither_open_nor_freed_is_restored_and_recovers() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    freed = _register(runtime, CHAIN)
    freed_tasks = _succeed_ready(runtime)
    runtime.settled_history.drain()
    assert not _resident(runtime, freed)
    unindexed = _register(runtime, CHAIN)
    task_id = pop_ready(runtime)
    assert task_id is not None
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    runtime.mark_started(task_id, "wkr-1", {}, _TS)
    registry.set_history_index_state(INDEX_COMPLETE)
    raw_control(registry).srem(WORKFLOWS_OPEN_KEY, unindexed)
    before = _reads(runtime, freed_tasks)

    restarted, loads = _restarted(registry, _close)

    assert loads.ids == [unindexed]
    assert _open(registry) == {unindexed}
    assert registry.history_census().consistent
    assert _reads(restarted, freed_tasks)[0] == before[0]
    listed = _page(restarted, _query(), 100)
    assert len(listed) == len(set(listed)) == 6
    info = restarted.describe_task(task_id)
    assert info is not None and info.status == "DISPATCHED"
    payload = result_payload(restarted._results, task_id, {"ok": True}, "org")
    restarted.mark_succeeded(task_id, "wkr-1", payload, _TS, "dsp-1")
    assert pop_ready(restarted) is not None
    scanned = _scans(registry)
    _, loads = _restarted(registry, _close)
    assert scanned == []
    assert loads.ids == [unindexed]


def test_a_deleted_freed_workflow_is_recounted_once() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    deleted, kept = _register(runtime, CHAIN), _register(runtime, CHAIN)
    _succeed_ready(runtime)
    runtime.settled_history.drain()
    registry.set_history_index_state(INDEX_COMPLETE)
    raw_control(registry).srem(WORKFLOWS_SET_KEY, deleted)
    assert not registry.history_census().consistent
    scanned = _scans(registry)

    _, loads = _restarted(registry, _close)

    assert scanned and loads.ids == []
    assert registry.history_census().consistent
    scanned = _scans(registry)
    _restarted(registry, _close)
    assert scanned == []
    assert _certified(registry, kept)


def test_a_freed_workflow_left_open_is_closed_and_not_restored() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    task_ids = _succeed_ready(runtime)
    runtime.settled_history.drain()
    before = _reads(runtime, task_ids)
    registry.set_history_index_state(INDEX_COMPLETE)
    registry.reopen_workflows(workflow_id)

    restarted, loads = _restarted(registry, _close)

    assert loads.ids == []
    assert _open(registry) == set()
    assert registry.history_census().consistent
    assert _reads(restarted, task_ids) == before
