"""A task prefers an idle worker already holding a copy of what its upstream tasks
published, and waits briefly for the worker that has just finished producing it."""

import asyncio
import hashlib
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from server.content.directory import ContentHolderRecord
from server.dispatcher import base as dispatcher_base
from server.dispatcher.base import HOLDER_GRACE_SEC, Dispatcher
from server.registries.worker import Worker
from server.task.models import TaskStatus
from server.task.runtime.content_bindings import UnreadableInput
from shared.content import TREE_MEDIA_TYPE, ContentReference, TreeManifest, TreeMember
from shared.private_state import OwnerFence
from shared.schemas.worker import WorkerStatus
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind, final_set
from shared.utils.time import now_iso
from tests.server.dispatch_helpers import record_dispatch
from tests.server.dispatcher.helpers import (
    CapturingDispatcher,
    make_capturing_dispatcher,
)
from tests.server.result_store import result_payload
from tests.server.task.test_v2_orchestration import FakeRegistry, _register
from tests.server.task.test_v2_orchestration import _runtime as _real_runtime
from tests.support.waiting import pop_ready

_TS = "2026-10-14T00:00:00Z"


def _ref(data: bytes, size: int) -> ContentReference:
    return ContentReference(
        authorization_scope="org",
        content_digest=hashlib.sha256(data).hexdigest(),
        size_bytes=size,
    )


_WEIGHTS = _ref(b"weights", 5_000)
_TOKENIZER = _ref(b"tokenizer", 10)


def _tree(name: str, largest: ContentReference, revision: str = "final") -> Any:
    return ArtifactBinding(
        name=name,
        revision=revision,
        kind=ArtifactKind.TREE,
        reference=_ref(name.encode(), 20),
        size_bytes=largest.size_bytes,
        largest=largest,
    )


class _Runtime:
    def __init__(self, scoped: dict[str, Any] | None, depends_on: list[str]) -> None:
        self._scoped = scoped
        self._depends_on = depends_on
        self.artifacts = {
            "tsk-train": {
                ("final", "model"): _tree("model", _WEIGHTS),
                ("final", "tok"): _tree("tok", _TOKENIZER),
                ("step-2", "checkpoints"): _tree(
                    "checkpoints", _ref(b"c", 9_000), "step-2"
                ),
            }
        }

    def scoped_inputs(self, task_id: str) -> dict[str, Any] | None:
        return self._scoped

    def describe_task(self, task_id: str) -> Any:
        return SimpleNamespace(depends_on=self._depends_on)

    def final_artifacts(self, task_id: str) -> Any:
        return final_set(self.artifacts.get(task_id) or {})


class _Holders:
    def __init__(self, by_digest: dict[str, list[ContentHolderRecord]]) -> None:
        self._by_digest = by_digest
        self.reads: list[str] = []

    def holders(self, scope: str, digest: str) -> list[ContentHolderRecord]:
        self.reads.append(digest)
        return self._by_digest.get(digest, [])


def _holder(worker_id: str, generation: int = 1) -> ContentHolderRecord:
    return ContentHolderRecord(worker_id, "nde-1", generation, 2e9)


def _pool(*ids: str) -> list[Any]:
    return [SimpleNamespace(id=worker_id, incarnation=1) for worker_id in ids]


def _dispatcher(holders: _Holders | None, v2: bool = False) -> Any:
    scoped = (
        {"model": SimpleNamespace(task_id="tsk-train", binding=None)} if v2 else None
    )
    return make_capturing_dispatcher(
        runtime=_Runtime(scoped, [] if v2 else ["tsk-train"]),
        content_holders=holders,
    )


@pytest.mark.parametrize("v2", [False, True])
def test_an_idle_holder_of_the_largest_upstream_object_is_preferred(v2: bool) -> None:
    holders = _Holders(
        {
            _WEIGHTS.content_digest: [_holder("wkr-2")],
            _TOKENIZER.content_digest: [_holder("wkr-3")],
        }
    )
    chosen = _dispatcher(holders, v2)._preferred_holder(
        "tsk-1", _pool("wkr-1", "wkr-2", "wkr-3")
    )

    assert chosen is not None and chosen.id == "wkr-2"
    assert holders.reads == [_WEIGHTS.content_digest]


def test_a_holder_of_another_incarnation_is_not_preferred() -> None:
    holders = _Holders({_WEIGHTS.content_digest: [_holder("wkr-2", generation=7)]})

    assert _dispatcher(holders)._preferred_holder("tsk-1", _pool("wkr-2")) is None


def test_a_busy_holder_leaves_the_task_to_normal_selection() -> None:
    holders = _Holders({_WEIGHTS.content_digest: [_holder("wkr-busy")]})
    dispatcher = _dispatcher(holders)

    assert dispatcher._preferred_holder("tsk-1", _pool("wkr-1")) is None
    assert dispatcher.requeued == []


def test_an_upstream_that_cannot_be_read_leaves_no_preference() -> None:
    holders = _Holders({_WEIGHTS.content_digest: [_holder("wkr-2")]})
    dispatcher = _dispatcher(holders)

    def unreadable(task_id: str) -> None:
        raise UnreadableInput("the producer's result is gone")

    dispatcher._runtime.scoped_inputs = unreadable

    assert dispatcher._preferred_holder("tsk-1", _pool("wkr-2")) is None
    assert holders.reads == []


def test_without_a_holder_directory_nothing_is_read() -> None:
    assert _dispatcher(None)._preferred_holder("tsk-1", _pool("wkr-1")) is None


_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: chain}
spec:
  graph:
    nodes:
      - {name: train, spec: {taskType: echo}}
      - {name: use, dependsOn: [train], spec: {taskType: echo}}
"""


def _worker(worker_id: str) -> Worker:
    return Worker(
        id=worker_id,
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_a_dispatch_goes_to_the_holder_only_with_stickiness_on(enabled: bool) -> None:
    holders = _Holders({_WEIGHTS.content_digest: [_holder("wkr-2")]})

    _, registry, _, _ = _dispatch_use(enabled, holders)

    assert registry.publish_task.call_args[0][0].id == ("wkr-2" if enabled else "wkr-1")
    assert holders.reads == ([_WEIGHTS.content_digest] if enabled else [])


def test_a_dispatch_carries_its_upstream_artifacts_and_its_worker_may_read_them() -> (
    None
):
    runtime, registry, ids, model = _dispatch_use(False, None)

    worker, message = registry.publish_task.call_args[0]
    assert message.upstream_artifacts is not None
    assert [b.reference for b in message.upstream_artifacts[ids["train"]]] == [
        model.reference
    ]
    assert runtime.content_binding_authorizes(ids["use"], worker.id, model.reference)
    assert not runtime.content_binding_authorizes(
        ids["use"], worker.id, _ref(b"elsewhere", 3)
    )


class _SettleRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__(None, ["tsk-train"])
        self.records = {
            "tsk-train": SimpleNamespace(
                status=TaskStatus.DONE,
                assigned_worker="wkr-2",
                succeeded_monotonic=100.0,
                dispatch_id="dsp-1",
            ),
            "tsk-1": SimpleNamespace(status=TaskStatus.PENDING),
        }

    def get_record(self, task_id: str) -> Any:
        return self.records.get(task_id)


def _settling(holder: ContentHolderRecord | None = None) -> Any:
    """A dispatcher whose consumer tsk-1's producer just finished on wkr-2, which
    holds the producer's largest object and still reports itself busy on it."""
    holders = _Holders({_WEIGHTS.content_digest: [holder or _holder("wkr-2")]})
    dispatcher = make_capturing_dispatcher(
        runtime=_SettleRuntime(), content_holders=holders
    )
    registry: Any = dispatcher._worker_registry
    registry.get_worker.return_value = _worker("wkr-2").model_copy(
        update={"status": WorkerStatus.BUSY}
    )
    registry.is_worker_stale.return_value = False
    registry.is_cordoned.return_value = False
    registry.reservation.return_value = None
    return dispatcher


def _waits(
    dispatcher: Any,
    now: float,
    consumer: str = "tsk-1",
    waitable: tuple[str, ...] = ("wkr-2",),
) -> bool:
    return dispatcher._holder_settling(consumer, now, lambda: set(waitable))


def test_a_consumer_waits_for_the_busy_holder_that_just_produced_its_input() -> None:
    assert _waits(_settling(), 100.0)


def test_a_consumer_waits_while_its_holder_is_still_reserved_for_the_producer() -> None:
    dispatcher = _settling()
    dispatcher._worker_registry.reservation.return_value = SimpleNamespace(
        dispatch_id="dsp-1"
    )

    assert _waits(dispatcher, 100.0)


def _busy_elsewhere(d: Any) -> None:
    d._worker_registry.reservation.return_value = SimpleNamespace(dispatch_id="dsp-9")


def _cordoned(d: Any) -> None:
    d._worker_registry.is_cordoned.return_value = True


def _stale(d: Any) -> None:
    d._worker_registry.is_worker_stale.return_value = True


def _gone(d: Any) -> None:
    d._worker_registry.get_worker.return_value = None


def _idle(d: Any) -> None:
    d._worker_registry.get_worker.return_value = _worker("wkr-2").model_copy(
        update={"status": WorkerStatus.IDLE}
    )


def _starting(d: Any) -> None:
    d._worker_registry.get_worker.return_value = _worker("wkr-2").model_copy(
        update={"status": WorkerStatus.STARTING}
    )


def test_a_holder_gone_idle_outside_the_pool_is_waited_for_once() -> None:
    dispatcher = _settling()
    _idle(dispatcher)

    assert _waits(dispatcher, 100.0)
    assert not _waits(dispatcher, 101.0)


def _ran_elsewhere(d: Any) -> None:
    d._runtime.records["tsk-train"].assigned_worker = "wkr-3"


def _finished_long_ago(d: Any) -> None:
    d._runtime.records["tsk-train"].succeeded_monotonic = 100.0 - HOLDER_GRACE_SEC


def _not_done(d: Any) -> None:
    d._runtime.records["tsk-train"].status = TaskStatus.DISPATCHED


@pytest.mark.parametrize(
    "fall_through",
    [
        _busy_elsewhere,
        _cordoned,
        _stale,
        _gone,
        _starting,
        _ran_elsewhere,
        _finished_long_ago,
        _not_done,
    ],
)
def test_a_holder_that_cannot_take_the_consumer_soon_is_not_waited_for(
    fall_through: Any,
) -> None:
    dispatcher = _settling()
    fall_through(dispatcher)

    assert not _waits(dispatcher, 100.0)


def test_a_holder_on_another_incarnation_is_not_waited_for() -> None:
    assert not _waits(_settling(_holder("wkr-2", generation=7)), 100.0)


def test_without_a_holder_directory_no_consumer_waits() -> None:
    dispatcher = _settling()
    dispatcher._content_holders = None

    assert not _waits(dispatcher, 100.0)


def test_a_wait_ends_when_its_window_closes() -> None:
    dispatcher = _settling()

    assert _waits(dispatcher, 100.0)
    assert _waits(dispatcher, 100.0 + HOLDER_GRACE_SEC - 0.01)
    assert not _waits(dispatcher, 100.0 + HOLDER_GRACE_SEC)
    assert not _waits(dispatcher, 100.0 + 3 * HOLDER_GRACE_SEC)


def test_a_consumer_never_waits_twice_for_one_producer_settle() -> None:
    dispatcher = _settling()

    assert _waits(dispatcher, 100.0)
    _busy_elsewhere(dispatcher)
    assert not _waits(dispatcher, 101.0)
    dispatcher._worker_registry.reservation.return_value = None
    assert not _waits(dispatcher, 102.0)


def test_a_settled_consumer_drops_its_wait() -> None:
    dispatcher = _settling()
    assert _waits(dispatcher, 100.0)

    dispatcher._runtime.records["tsk-1"].status = TaskStatus.CANCELLED
    _waits(dispatcher, 101.0, "tsk-other")

    assert [key[0] for key in dispatcher._holder_graces] == ["tsk-other"]


def _train_then_ready(finished: str) -> tuple[Any, str, dict[str, str], Any]:
    """Run the chain's train on wkr-1 to success at ``finished``, publishing a model
    tree, and leave use ready."""
    runtime = _real_runtime(FakeRegistry())
    workflow_id, ids = asyncio.run(_register(runtime, _CHAIN))
    train = ids["train"]
    assert pop_ready(runtime) == train
    record_dispatch(runtime, train, "wkr-1", "dsp-1")
    runtime.mark_started(train, "wkr-1", {}, _TS)
    payload = result_payload(runtime._results, train, {"ok": True}, "org")
    manifest = TreeManifest.of(
        [TreeMember("w.bin", _WEIGHTS.content_digest, _WEIGHTS.size_bytes)]
    )
    stored = runtime._results._store.write(
        "org", manifest.encode(), media_type=TREE_MEDIA_TYPE
    )
    model = _tree("model", _WEIGHTS).model_copy(update={"reference": stored})
    payload["artifacts"] = [model.model_dump(mode="json")]
    runtime.mark_succeeded(train, "wkr-1", payload, finished, "dsp-1")
    return runtime, workflow_id, ids, model


def _dispatch_use(enabled: bool, holders: Any) -> tuple[Any, mock.Mock, dict, Any]:
    runtime, _, ids, model = _train_then_ready(_TS)
    use = ids["use"]
    assert pop_ready(runtime) == use
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker("wkr-1"), _worker("wkr-2")]
    registry.satisfying_workers.return_value = [_worker("wkr-1"), _worker("wkr-2")]
    registry.publish_task.return_value = 1
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("holder-preference"),
        enable_stage_weight_stickiness=enabled,
        content_holders=holders,
        worker_selection_strategy="first_fit",
    )
    dispatcher.dispatch_once(use)
    return runtime, registry, ids, model


def _graced_dispatcher(
    finished: str | None = None,
) -> tuple[Any, str, dict[str, str], Dispatcher, mock.Mock]:
    """A real dispatcher for use while train's worker wkr-1, holding train's model,
    still reports itself busy and only wkr-2 is idle; train's worker reported it
    finished at ``finished``, now by default."""
    runtime, workflow_id, ids, _ = _train_then_ready(finished or now_iso())
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker("wkr-2")]
    registry.satisfying_workers.return_value = [_worker("wkr-1"), _worker("wkr-2")]
    registry.publish_task.return_value = 1
    registry.get_worker.return_value = _worker("wkr-1").model_copy(
        update={"status": WorkerStatus.BUSY}
    )
    registry.is_worker_stale.return_value = False
    registry.is_cordoned.return_value = False
    registry.reservation.return_value = None
    holders: Any = _Holders({_WEIGHTS.content_digest: [_holder("wkr-1")]})
    dispatcher = Dispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("holder-preference"),
        enable_stage_weight_stickiness=True,
        content_holders=holders,
        worker_selection_strategy="first_fit",
    )
    return runtime, workflow_id, ids, dispatcher, registry


def test_a_graced_consumer_returns_to_the_queue_without_spending_an_attempt() -> None:
    runtime, _, ids, dispatcher, registry = _graced_dispatcher()
    use = ids["use"]
    assert pop_ready(runtime) == use

    assert dispatcher.dispatch_once(use) is False

    record = runtime.get_record(use)
    assert record is not None
    assert record.status == TaskStatus.PENDING and record.attempts == 0
    registry.publish_task.assert_not_called()
    assert pop_ready(runtime) == use


def test_a_pinned_consumer_is_not_graced() -> None:
    runtime, _, ids, dispatcher, registry = _graced_dispatcher()
    use = ids["use"]
    assert pop_ready(runtime) == use
    record = runtime.get_record(use)
    assert record is not None
    record.selected_worker = ["wkr-2"]

    assert dispatcher.dispatch_once(use) is True

    assert registry.publish_task.call_args[0][0].id == "wkr-2"
    assert dispatcher._holder_graces == {}


def test_a_consumer_cancelled_while_graced_settles_cancelled() -> None:
    runtime, workflow_id, ids, dispatcher, registry = _graced_dispatcher()
    use = ids["use"]
    assert pop_ready(runtime) == use
    assert dispatcher.dispatch_once(use) is False

    runtime.cancel_workflow(workflow_id)

    record = runtime.get_record(use)
    assert record is not None and record.status == TaskStatus.CANCELLED
    registry.publish_task.assert_not_called()


def test_a_graced_consumer_is_retried_at_its_own_pace_without_a_loop_backoff() -> None:
    runtime, _, ids, dispatcher, registry = _graced_dispatcher()
    stop = threading.Event()
    sleeps: list[float] = []
    attempts: list[float] = []
    dispatch_once = dispatcher.dispatch_once

    def attempt(task_id: str) -> bool:
        attempts.append(time.monotonic())
        placed = dispatch_once(task_id)
        if placed or len(attempts) == 3:
            stop.set()
        return placed

    with (
        mock.patch.object(dispatcher_base.time, "sleep", sleeps.append),
        mock.patch.object(dispatcher, "dispatch_once", attempt),
    ):
        dispatcher.dispatch_loop(stop, poll_interval=0.01)

    assert len(attempts) == 3 and sleeps == []
    gaps = [later - earlier for earlier, later in zip(attempts, attempts[1:])]
    assert all(gap >= dispatcher_base._HOLDER_RECHECK_SEC * 0.9 for gap in gaps)
    registry.publish_task.assert_not_called()


_SOLO = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: solo}
spec:
  graph:
    nodes:
      - {name: other, spec: {taskType: echo}}
"""


def test_graced_consumers_never_hold_up_a_placeable_task() -> None:
    runtime = _real_runtime(FakeRegistry())
    chains = [asyncio.run(_register(runtime, _CHAIN))[1] for _ in range(8)]
    assert {pop_ready(runtime) for _ in chains} == {ids["train"] for ids in chains}
    for i, ids in enumerate(chains):
        train = ids["train"]
        record_dispatch(runtime, train, f"wkr-h{i}", f"dsp-{i}")
        runtime.mark_started(train, f"wkr-h{i}", {}, now_iso())
        payload = result_payload(runtime._results, train, {"ok": True}, "org")
        manifest = TreeManifest.of(
            [TreeMember("w.bin", _WEIGHTS.content_digest, _WEIGHTS.size_bytes)]
        )
        stored = runtime._results._store.write(
            "org", manifest.encode(), media_type=TREE_MEDIA_TYPE
        )
        model = _tree("model", _WEIGHTS).model_copy(update={"reference": stored})
        payload["artifacts"] = [model.model_dump(mode="json")]
        runtime.mark_succeeded(train, f"wkr-h{i}", payload, now_iso(), f"dsp-{i}")
    other = asyncio.run(_register(runtime, _SOLO))[1]["other"]
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker("wkr-idle")]
    registry.satisfying_workers.return_value = [_worker("wkr-idle")] + [
        _worker(f"wkr-h{i}") for i in range(8)
    ]
    registry.publish_task.return_value = 1
    registry.get_worker.side_effect = lambda worker_id: _worker(worker_id).model_copy(
        update={"status": WorkerStatus.BUSY}
    )
    registry.is_worker_stale.return_value = False
    registry.is_cordoned.return_value = False
    registry.reservation.return_value = None
    holders: Any = _Holders(
        {_WEIGHTS.content_digest: [_holder(f"wkr-h{i}") for i in range(8)]}
    )
    dispatcher = Dispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("holder-preference"),
        enable_stage_weight_stickiness=True,
        content_holders=holders,
        worker_selection_strategy="first_fit",
    )
    stop = threading.Event()
    sleeps: list[float] = []
    dispatch_once = dispatcher.dispatch_once
    started = time.monotonic()

    def attempt(task_id: str) -> bool:
        placed = dispatch_once(task_id)
        if task_id == other:
            stop.set()
        return placed

    with (
        mock.patch.object(dispatcher_base.time, "sleep", sleeps.append),
        mock.patch.object(dispatcher, "dispatch_once", attempt),
    ):
        dispatcher.dispatch_loop(stop, poll_interval=0.01)

    assert registry.publish_task.call_count == 1
    assert registry.publish_task.call_args[0][0].id == "wkr-idle"
    assert sleeps == []
    assert time.monotonic() - started < dispatcher_base._HOLDER_RECHECK_SEC


def test_a_producer_whose_worker_clock_lags_still_graces_its_consumer() -> None:
    runtime, _, ids, dispatcher, registry = _graced_dispatcher(finished=_TS)
    use = ids["use"]
    assert pop_ready(runtime) == use

    assert dispatcher.dispatch_once(use) is False

    registry.publish_task.assert_not_called()


def _exclude_failed(record: Any, registry: mock.Mock) -> None:
    record.failed_workers = ["wkr-1"]


def _exclude_avoided(record: Any, registry: mock.Mock) -> None:
    record.avoided_worker = "wkr-1"


def _exclude_unfit(record: Any, registry: mock.Mock) -> None:
    registry.satisfying_workers.return_value = [_worker("wkr-2")]


@pytest.mark.parametrize("exclude", [_exclude_failed, _exclude_avoided, _exclude_unfit])
@pytest.mark.parametrize("holder_status", [WorkerStatus.BUSY, WorkerStatus.IDLE])
def test_a_holder_the_consumer_cannot_be_placed_on_is_not_waited_for(
    exclude: Any, holder_status: WorkerStatus
) -> None:
    runtime, _, ids, dispatcher, registry = _graced_dispatcher()
    use = ids["use"]
    assert pop_ready(runtime) == use
    registry.get_worker.return_value = _worker("wkr-1").model_copy(
        update={"status": holder_status}
    )
    record = runtime.get_record(use)
    assert record is not None
    exclude(record, registry)

    assert dispatcher.dispatch_once(use) is True

    assert registry.publish_task.call_args[0][0].id == "wkr-2"


def test_an_owner_bound_consumer_waits_only_for_its_owner() -> None:
    dispatcher = _settling()
    record: Any = SimpleNamespace(avoided_worker=None)
    owner = OwnerFence(worker_id="wkr-3", incarnation=1)

    assert dispatcher._waitable_workers(record, owner, set(), False) == {"wkr-3"}
    assert not _waits(dispatcher, 100.0, waitable=("wkr-3",))


def test_a_holder_of_only_one_of_a_producer_s_artifacts_is_waited_for() -> None:
    dispatcher = _settling()
    dispatcher._content_holders = _Holders(
        {_TOKENIZER.content_digest: [_holder("wkr-2")]}
    )

    assert _waits(dispatcher, 100.0)
    assert _waits(dispatcher, 100.5)
