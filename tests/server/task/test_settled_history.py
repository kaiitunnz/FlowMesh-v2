"""A settled workflow's runtime state is freed only behind its certified history."""

import asyncio
import gc
import logging
import threading
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from server.clients.redis import (
    HISTORY_COUNTS_KEY,
    HISTORY_SEQ_KEY,
    TASK_HISTORY_MEMBERS_KEY,
    WORKFLOWS_OPEN_KEY,
    workflow_history_key,
)
from server.config import OrchestrationConfig
from server.registries.history import HistoryPublication
from server.registries.workflow import WorkflowRegistry
from server.task.history import CERTIFICATE_FIELD
from server.task.models import TaskStatus
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from shared.harness.adapter import HarnessResult, HarnessResultKind
from shared.harness.boundary import BoundaryEventKind, BoundaryRequest
from shared.tools.contract import MediatedOperationOutcome
from shared.tools.search.schema import SEARCH_INTERFACE
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.result_store import make_result_reader, result_payload
from tests.server.runtime_helpers import manual_durability_retry
from tests.server.task.test_v2_orchestration import LINEAR, _WorkerRegistryStub
from tests.support.waiting import pop_ready, until

_TS = "2026-06-01T00:00:00Z"

CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: chain
spec:
  graph:
    nodes:
      - {name: a, spec: {taskType: echo}}
      - {name: b, dependsOn: [a], spec: {taskType: echo}}
      - {name: c, dependsOn: [b], spec: {taskType: echo}}
"""


class _Workers(_WorkerRegistryStub):
    """A worker registry whose interrupt delivery can be made to fail."""

    def __init__(self) -> None:
        self.refuse_interrupts = False

    def publish_interrupt(self, *args: Any) -> int:
        if self.refuse_interrupts:
            raise RuntimeError("interrupt not delivered")
        return 0


def _registry() -> WorkflowRegistry:
    return WorkflowRegistry(fake_redis_client(fakeredis.FakeServer()))


def _runtime(registry: WorkflowRegistry, workers: Any = None) -> TaskRuntime:
    runtime = TaskRuntime(
        registry,
        cast(Any, workers or _Workers()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("test.settled_history"),
        credential_vault=InMemoryCredentialVault(),
        redrive=lambda fire, logger: StoreRedriveScheduler(
            fire, logger, run_thread=False
        ),
        durability_retry=manual_durability_retry,
    )
    runtime.settled_history.require(closure=False, archive=False)
    asyncio.run(runtime.start_settled_history())
    return runtime


def _register(runtime: TaskRuntime, text: str) -> str:
    workflow_id, _ = asyncio.run(
        runtime.register("owner", "org", text, format="native")
    )
    return workflow_id


def _succeed_ready(runtime: TaskRuntime) -> list[str]:
    """Dispatch and succeed every task as it becomes ready, each with a stored
    result."""
    settled: list[str] = []
    while (task_id := pop_ready(runtime, timeout=0.05)) is not None:
        record_dispatch(runtime, task_id, "wkr-1", f"dsp-{task_id}")
        runtime.mark_started(task_id, "wkr-1", {}, _TS)
        payload = result_payload(runtime._results, task_id, {"ok": True}, "org")
        runtime.mark_succeeded(task_id, "wkr-1", payload, _TS, f"dsp-{task_id}")
        settled.append(task_id)
    return settled


def _settled(registry: WorkflowRegistry, text: str = CHAIN) -> tuple[TaskRuntime, str]:
    runtime = _runtime(registry)
    workflow_id = _register(runtime, text)
    assert _succeed_ready(runtime)
    return runtime, workflow_id


def _resident(runtime: TaskRuntime, workflow_id: str) -> bool:
    with runtime._lock:
        return runtime._tasks.holds(workflow_id)


def _certified(registry: WorkflowRegistry, workflow_id: str) -> bool:
    raw = raw_control(registry)
    key = workflow_history_key(workflow_id)
    return raw.type(key) == "hash" and bool(raw.hexists(key, CERTIFICATE_FIELD))


@pytest.mark.parametrize("text", [CHAIN, LINEAR])
def test_a_settled_workflow_leaves_memory_behind_its_certified_history(
    text: str,
) -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry, text)
    task_ids = runtime._tasks.ids_of(workflow_id)
    engine = runtime._engines.get(workflow_id)
    engine_ref = weakref.ref(engine) if engine is not None else None
    del engine

    runtime.settled_history.drain()

    assert not _resident(runtime, workflow_id)
    assert _certified(registry, workflow_id)
    assert workflow_id not in registry.open_workflow_ids()
    assert registry.history_counts() == (len(task_ids), len(task_ids))
    assert runtime.settled_history.counts_locked() == (len(task_ids), len(task_ids))
    for each in (
        runtime._original_deps,
        runtime._completed,
        runtime._failed,
        runtime._dag.pending_deps,
        runtime._dag.dependents,
        runtime._ready.merge_key_by_task,
        runtime._epochs.task_epoch_index,
        runtime._fence.returned_dispatches,
        runtime._rehydrated_dispatched,
    ):
        assert not any(task_id in each for task_id in task_ids), each
    assert workflow_id not in runtime._engines
    assert workflow_id not in runtime._occurrences._blueprints
    gc.collect()
    assert engine_ref is None or engine_ref() is None


def test_a_workflow_still_running_stays_resident() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)

    runtime.settled_history.request(workflow_id)
    runtime.settled_history.drain()

    assert _resident(runtime, workflow_id)
    assert not _certified(registry, workflow_id)


def test_a_workflow_is_not_freed_while_an_action_of_it_is_being_delivered() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    during: list[bool] = []

    def settled(settled_id: str) -> None:
        during.append(runtime.settled_history.evaluate(settled_id))

    runtime.set_completion_notifier(settled)
    _succeed_ready(runtime)

    assert during and not any(during)
    assert _resident(runtime, workflow_id)
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


def test_a_workflow_whose_action_failed_to_deliver_stays_until_it_is() -> None:
    registry = _registry()
    workers = _Workers()
    runtime = _runtime(registry, workers)
    workflow_id = _register(runtime, CHAIN)
    first = pop_ready(runtime)
    assert first is not None
    record_dispatch(runtime, first, "wkr-1", "dsp-1")
    workers.refuse_interrupts = True
    runtime.cancel_workflow(workflow_id)
    runtime.mark_cancelled(first, "wkr-1", {}, _TS, "dsp-1")
    assert runtime.workflow_settlement(workflow_id).settled

    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert _resident(runtime, workflow_id)

    workers.refuse_interrupts = False
    runtime._durability.run_due()
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


def _held(runtime: TaskRuntime, workflow_id: str) -> Callable[[], None]:
    with runtime._lock:
        runtime.settled_history.hold_locked(workflow_id)

    def release() -> None:
        with runtime._lock:
            assert runtime.settled_history.release_locked(workflow_id)
        runtime.settled_history.request(workflow_id)

    return release


def _unclosed(runtime: TaskRuntime, workflow_id: str) -> Callable[[], None]:
    runtime.settled_history.require(closure=True, archive=False)
    return lambda: runtime.workflow_closed(workflow_id)


def _unarchived(runtime: TaskRuntime, workflow_id: str) -> Callable[[], None]:
    runtime.settled_history.require(closure=False, archive=True)
    task_ids = runtime._tasks.ids_of(workflow_id)

    def archive() -> None:
        for task_id in task_ids:
            runtime.task_archived(task_id)

    return archive


@pytest.mark.parametrize("keep", [_held, _unclosed, _unarchived])
def test_a_workflow_held_unclosed_or_unarchived_stays_until_released(
    keep: Callable[[TaskRuntime, str], Callable[[], None]],
) -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    release = keep(runtime, workflow_id)

    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert _resident(runtime, workflow_id)

    release()
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


def test_a_workflow_whose_resident_claims_are_open_stays_resident() -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry, LINEAR)
    asked: list[frozenset[str]] = []
    settled = False

    def claims_settled(invocation_ids: Any) -> bool:
        asked.append(frozenset(invocation_ids))
        return settled

    runtime.settled_history.set_invocations_retired(claims_settled)
    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert _resident(runtime, workflow_id)
    assert asked and asked[0] == frozenset(
        runtime._engines[workflow_id].invocation_ids()
    )

    settled = True
    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


def _corrupt(key: Callable[[str], str]) -> Callable[[WorkflowRegistry, str], None]:
    def corrupt(registry: WorkflowRegistry, workflow_id: str) -> None:
        raw_control(registry).set(key(workflow_id), "wrong type")

    return corrupt


def _drop_prepare_reply(registry: WorkflowRegistry, workflow_id: str) -> None:
    def dropped(publication: HistoryPublication) -> None:
        raise RedisConnectionError("reply lost")

    cast(Any, registry).prepare_history = dropped


@pytest.mark.parametrize(
    "fault",
    [
        _corrupt(workflow_history_key),
        _corrupt(lambda workflow_id: TASK_HISTORY_MEMBERS_KEY),
        _drop_prepare_reply,
    ],
    ids=["projection-type", "index-type", "dropped-reply"],
)
def test_a_failed_preparation_leaves_the_workflow_resident_and_uncounted(
    fault: Callable[[WorkflowRegistry, str], None],
) -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    before = {
        task_id: runtime.describe_task(task_id)
        for task_id in runtime._tasks.ids_of(workflow_id)
    }
    fault(registry, workflow_id)

    runtime.settled_history.drain()

    assert _resident(runtime, workflow_id)
    assert not runtime.settled_history.sealed_locked(workflow_id)
    assert not _certified(registry, workflow_id)
    assert workflow_id in registry.open_workflow_ids()
    assert registry.history_counts() == (0, 0)
    assert runtime.settled_history.counts_locked() == (0, 0)
    assert {task_id: runtime.describe_task(task_id) for task_id in before} == before


@pytest.mark.parametrize(
    ("key", "field", "value"),
    [
        (HISTORY_SEQ_KEY, None, "not a number"),
        (HISTORY_COUNTS_KEY, "total", "not a number"),
        (HISTORY_COUNTS_KEY, "done", "1.5"),
    ],
    ids=["sequence", "total", "done"],
)
def test_an_unreadable_counter_refuses_the_certification_before_any_write(
    key: str, field: str | None, value: str
) -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    raw = raw_control(registry)
    if field is None:
        raw.set(key, value)
    else:
        raw.hset(key, field, value)
    before = {name: raw.dump(name) for name in (HISTORY_SEQ_KEY, HISTORY_COUNTS_KEY)}

    runtime.settled_history.drain()

    assert _resident(runtime, workflow_id)
    assert not runtime.settled_history.sealed_locked(workflow_id)
    assert not _certified(registry, workflow_id)
    assert workflow_id in registry.open_workflow_ids()
    assert {name: raw.dump(name) for name in before} == before


def test_a_write_after_the_revision_is_read_refuses_and_a_later_attempt_frees() -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    prepare = registry.prepare_history
    task_id = runtime._tasks.ids_of(workflow_id)[0]

    def prepare_then_write(publication: HistoryPublication) -> bool:
        prepared = prepare(publication)
        stored = registry.load_task_states(workflow_id, task_id)
        registry.save_task_states([state for state in stored if state])
        return prepared

    cast(Any, registry).prepare_history = prepare_then_write
    runtime.settled_history.drain()
    assert _resident(runtime, workflow_id)
    assert not _certified(registry, workflow_id)

    cast(Any, registry).prepare_history = prepare
    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


@pytest.mark.parametrize("landed", [True, False])
def test_a_lost_certification_reply_is_settled_by_reading_the_certificate_back(
    landed: bool,
) -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    certify = registry.certify_history

    def lost(publication: HistoryPublication, revision: int) -> str:
        if landed:
            certify(publication, revision)
        raise RedisConnectionError("reply lost")

    cast(Any, registry).certify_history = lost
    runtime.settled_history.drain()

    assert _resident(runtime, workflow_id) is not landed
    assert _certified(registry, workflow_id) is landed
    assert not runtime.settled_history.sealed_locked(workflow_id)


def test_an_unreadable_certificate_keeps_the_workflow_sealed_until_it_reads() -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    certify = registry.certify_history
    read_back = registry.history_certificate

    def lost(publication: HistoryPublication, revision: int) -> str:
        certify(publication, revision)
        raise RedisConnectionError("reply lost")

    def unreadable(workflow_id: str) -> Any:
        raise RedisConnectionError("store unavailable")

    cast(Any, registry).certify_history = lost
    cast(Any, registry).history_certificate = unreadable
    runtime.settled_history.drain()
    assert _resident(runtime, workflow_id)
    assert runtime.settled_history.sealed_locked(workflow_id)

    cast(Any, registry).history_certificate = read_back
    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


def test_a_late_report_while_sealed_changes_nothing() -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    task_ids = runtime._tasks.ids_of(workflow_id)
    before = {task_id: runtime.describe_task(task_id) for task_id in task_ids}
    revision = registry.workflow_revision(workflow_id)
    prepare = registry.prepare_history

    def report_then_fail(publication: HistoryPublication) -> None:
        for task_id in task_ids:
            runtime.mark_succeeded(task_id, "wkr-1", {}, _TS, f"dsp-{task_id}")
            runtime.mark_updated(task_id, "wkr-1", {"late": True}, f"dsp-{task_id}")
        assert runtime.cancel_workflow(workflow_id) == []
        raise RedisConnectionError("preparation failed")

    cast(Any, registry).prepare_history = report_then_fail
    runtime.settled_history.drain()

    assert _resident(runtime, workflow_id)
    assert not runtime.settled_history.sealed_locked(workflow_id)
    assert registry.workflow_revision(workflow_id) == revision
    assert {task_id: runtime.describe_task(task_id) for task_id in task_ids} == before
    assert not runtime._actions.owes(workflow_id)

    cast(Any, registry).prepare_history = prepare
    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


def test_the_freed_sequence_never_goes_backwards() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    first = _register(runtime, CHAIN)
    second = _register(runtime, CHAIN)
    _succeed_ready(runtime)
    certify = registry.certify_history

    def lost_for_first(publication: HistoryPublication, revision: int) -> str:
        outcome = certify(publication, revision)
        if publication.workflow_id == first:
            raise RedisConnectionError("reply lost")
        return outcome

    def unreadable(workflow_id: str) -> Any:
        raise RedisConnectionError("store unavailable")

    read_back = registry.history_certificate
    cast(Any, registry).certify_history = lost_for_first
    cast(Any, registry).history_certificate = unreadable
    assert not runtime.settled_history.evaluate(first)
    assert runtime.settled_history.evaluate(second)
    assert runtime.settled_history.sealed_locked(first)
    assert not _resident(runtime, second)
    second_seq = runtime.settled_history.published_through
    assert registry.history_seq() == second_seq

    cast(Any, registry).history_certificate = read_back
    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert not _resident(runtime, first)
    certificate = registry.history_certificate(first)
    assert certificate is not None and certificate.seq < second_seq
    assert runtime.settled_history.published_through == second_seq
    assert registry.history_seq() == second_seq


def test_a_cancelled_task_is_counted_freed_but_not_done() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    runtime.cancel_workflow(workflow_id)

    runtime.settled_history.drain()

    assert not _resident(runtime, workflow_id)
    assert registry.history_counts() == (3, 0)
    stored = registry.load_history(
        [
            member.partition(":")[2]
            for member in raw_control(registry).hvals(TASK_HISTORY_MEMBERS_KEY)
        ]
    )
    assert {entry.persisted.record.status for entry in stored if entry} == {
        TaskStatus.CANCELLED
    }
    assert not raw_control(registry).sismember(WORKFLOWS_OPEN_KEY, workflow_id)


def test_the_freeing_thread_frees_a_settled_workflow_and_stops_promptly() -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    stop = threading.Event()
    thread = threading.Thread(target=runtime.settled_history.run, args=(stop,))
    thread.start()
    try:
        asyncio.run(until(lambda: not _resident(runtime, workflow_id)))
    finally:
        stop.set()
        thread.join(timeout=5.0)

    assert not thread.is_alive()


def test_a_tail_during_the_seal_leaves_nothing_behind_the_free() -> None:
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    task_id = runtime._tasks.ids_of(workflow_id)[0]
    prepare = registry.prepare_history

    def tail_while_sealed(publication: HistoryPublication) -> bool:
        assert runtime.settled_history.sealed_locked(workflow_id)
        with runtime.event_tail(task_id):
            pass
        return prepare(publication)

    cast(Any, registry).prepare_history = tail_while_sealed
    runtime.settled_history.drain()
    cast(Any, registry).prepare_history = prepare
    assert not _resident(runtime, workflow_id)

    with runtime.event_tail(task_id):
        pass
    runtime.settled_history.request(workflow_id)
    runtime.settled_history.sweep()
    runtime.settled_history.drain()

    history = runtime.settled_history
    assert workflow_id not in history._tails
    assert workflow_id not in history._candidates
    assert workflow_id not in history._requested


def test_archive_reports_ask_to_free_a_workflow_once_its_last_task_is_archived() -> (
    None
):
    registry = _registry()
    runtime, workflow_id = _settled(registry)
    runtime.settled_history.require(closure=False, archive=True)
    task_ids = runtime._tasks.ids_of(workflow_id)
    requested: list[str] = []
    request = runtime.settled_history.request

    def recorded(workflow_id: str) -> None:
        requested.append(workflow_id)
        request(workflow_id)

    cast(Any, runtime.settled_history).request = recorded
    for task_id in [*task_ids[:-1], task_ids[0]]:
        runtime.task_archived(task_id)
    assert requested == []
    runtime.settled_history.sweep()
    runtime.settled_history.drain()
    assert _resident(runtime, workflow_id)

    runtime.task_archived(task_ids[-1])

    assert requested == [workflow_id]
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)


def _captured_step(call: str, interface: str) -> dict[str, Any]:
    return {
        "agent_episode": HarnessResult(
            kind=HarnessResultKind.BOUNDARY,
            request=BoundaryRequest(
                kind=BoundaryEventKind.INVOCATION,
                call_correlation=call,
                interface=interface,
                request_digest="sha256:0",
            ),
        ).model_dump(mode="json")
    }


class _ReapWorkers(_Workers):
    """A worker registry that records each mediated-op frame it relays."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[tuple[str, dict[str, Any]]] = []

    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(worker_id=worker_id, node_id="node-1")

    def publish_mediated_op(self, worker: Any, message: Any) -> int:
        self.frames.append((message.frame_kind, message.payload))
        return 1


@pytest.mark.parametrize("freed", [True, False], ids=["freed", "sealed"])
def test_a_stale_step_for_a_freed_or_sealed_task_reaps_its_captures(
    freed: bool,
) -> None:
    registry = _registry()
    workers = _ReapWorkers()
    runtime = _runtime(registry, workers)
    workflow_id = _register(runtime, CHAIN)
    task_id = _succeed_ready(runtime)[0]
    prepare = registry.prepare_history

    def report(publication: Any = None) -> None:
        for call, interface in (("c-search", SEARCH_INTERFACE), ("c-model", "model")):
            runtime.mark_succeeded(
                task_id, "wkr-2", _captured_step(call, interface), _TS, "dsp-old"
            )

    if freed:
        runtime.settled_history.drain()
        assert not _resident(runtime, workflow_id)
        report()
    else:

        def report_while_sealed(publication: HistoryPublication) -> bool:
            report()
            return prepare(publication)

        cast(Any, registry).prepare_history = report_while_sealed
        runtime.settled_history.drain()
        assert not _resident(runtime, workflow_id)

    assert sorted(workers.frames, key=str) == sorted(
        [
            ("reap", {"agent_task_id": task_id, "call_correlation": "c-search"}),
            ("reap", {"agent_task_id": task_id, "call_correlation": "c-model"}),
            ("resident_reap", {"task_id": task_id, "call_correlation": "c-model"}),
        ],
        key=str,
    )
    assert not runtime._actions.owes(workflow_id)


def test_a_late_mediated_outcome_for_a_freed_or_sealed_task_reaps_its_worker() -> None:
    registry = _registry()
    workers = _ReapWorkers()
    runtime = _runtime(registry, workers)
    workflow_id = _register(runtime, CHAIN)
    task_ids = _succeed_ready(runtime)
    prepare = registry.prepare_history
    revision = registry.workflow_revision(workflow_id)

    def outcome(task_id: str) -> MediatedOperationOutcome:
        return MediatedOperationOutcome(
            permit_id="mop-1",
            agent_task_id=task_id,
            call_correlation="c-0",
            invocation_id="inv-1",
            idempotency_key=None,
            error="late",
        )

    def late_while_sealed(publication: HistoryPublication) -> bool:
        runtime.settle_mediated_operation(outcome(task_ids[0]), "wkr-2")
        runtime.release_merge(task_ids[0])
        assert registry.workflow_revision(workflow_id) == revision
        return prepare(publication)

    cast(Any, registry).prepare_history = late_while_sealed
    runtime.settled_history.drain()
    assert not _resident(runtime, workflow_id)
    assert not runtime._committer.forget_locked(workflow_id)
    runtime.settle_mediated_operation(outcome(task_ids[1]), "wkr-3")

    assert workers.frames == [
        ("reap", {"agent_task_id": task_ids[0], "call_correlation": "c-0"}),
        ("reap", {"agent_task_id": task_ids[1], "call_correlation": "c-0"}),
    ]


class _CountingLock:
    """A lock that counts how often it is taken."""

    def __init__(self, lock: Any) -> None:
        self._lock = lock
        self.taken = 0

    def __enter__(self) -> Any:
        self.taken += 1
        return self._lock.__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._lock.__exit__(*exc)


def test_a_nested_event_tail_looks_up_its_task_once() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    task_id = runtime._tasks.ids_of(workflow_id)[0]
    counting = _CountingLock(runtime._lock)
    cast(Any, runtime)._lock = counting

    with runtime.event_tail(task_id):
        taken = counting.taken
        with runtime.event_tail(task_id):
            pass
        assert counting.taken == taken
        with runtime.event_tail("tsk-other"):
            pass
        assert counting.taken == taken + 1


def test_a_running_workflow_is_not_asked_to_be_freed() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, CHAIN)
    task_id = pop_ready(runtime)
    assert task_id is not None
    requested: list[str] = []
    cast(Any, runtime.settled_history).request = requested.append

    with runtime.event_tail(task_id):
        record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
        runtime.mark_started(task_id, "wkr-1", {}, _TS)
        payload = result_payload(runtime._results, task_id, {"ok": True}, "org")
        runtime.mark_succeeded(task_id, "wkr-1", payload, _TS, "dsp-1")

    assert requested == []
    assert _resident(runtime, workflow_id)
