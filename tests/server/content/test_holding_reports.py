"""How control applies a worker's holding reports: in the order the worker sent them,
each once, asking for the whole holding when one went missing, and sweeping the records
of workers no longer registered."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import fakeredis

from server.clients.redis import WORKERS_SET_KEY, RedisClient
from server.content import (
    ContentHolderDirectory,
    ContentHydrationAuthority,
    ContentTransferSessions,
)
from shared.content import HOLDING_RESYNC_FRAME, HoldingReport
from tests.server.redis_helpers import fake_redis_client

_A = ("org", "a" * 64)
_B = ("org", "b" * 64)


class _Relay:
    def pipeline(self) -> Any:
        raise AssertionError("no transfer is arranged here")


class _Workers:
    """Registered workers; a lookup can be made to take a while, as a Redis read
    can."""

    def __init__(self, redis: RedisClient, *worker_ids: str) -> None:
        self._redis = redis
        self._live = set(worker_ids)
        if worker_ids:
            redis.sync.sadd(WORKERS_SET_KEY, *worker_ids)
        self.delays: list[float] = []
        self.frames: list[tuple[str, str]] = []

    async def get_worker_async(self, worker_id: str) -> Any:
        if self.delays:
            await asyncio.sleep(self.delays.pop(0))
        if worker_id not in self._live:
            return None
        return SimpleNamespace(id=worker_id, node_id="nde-1", incarnation=1)

    async def publish_mediated_op_async(self, worker: Any, payload: Any) -> int:
        self.frames.append((worker.id, payload.frame_kind))
        return 0

    def register(self, worker_id: str) -> None:
        self._live.add(worker_id)
        self._redis.sync.sadd(WORKERS_SET_KEY, worker_id)

    def unregister(self, worker_id: str) -> None:
        self._live.discard(worker_id)
        self._redis.sync.srem(WORKERS_SET_KEY, worker_id)

    def resyncs(self) -> list[str]:
        return [w for w, kind in self.frames if kind == HOLDING_RESYNC_FRAME]


def _setup(
    *worker_ids: str, redis: RedisClient | None = None
) -> tuple[ContentHydrationAuthority, _Workers, ContentHolderDirectory]:
    client = redis or fake_redis_client(fakeredis.FakeServer())
    workers = _Workers(client, *worker_ids)
    directory = ContentHolderDirectory(client)
    authority = ContentHydrationAuthority(
        directory,
        cast(Any, workers),
        authorizes=lambda task_id, worker_id, reference: True,
        grant_ttl_sec=60.0,
        sessions=ContentTransferSessions(
            cast(Any, _Relay()), cast(Any, _Relay()), ttl_sec=600.0
        ),
    )
    return authority, workers, directory


def _held(directory: ContentHolderDirectory, copy: tuple[str, str]) -> list[str]:
    return [record.worker_id for record in directory.holders(*copy)]


def test_reports_apply_in_the_order_the_worker_sent_them() -> None:
    """A report whose worker lookup is slow still applies before the next one."""
    authority, workers, directory = _setup("wkr-1")
    workers.delays = [0.05, 0.0]

    async def run() -> None:
        loop = asyncio.get_running_loop()
        authority.bind_loop(loop)
        authority.on_holding("wkr-1", HoldingReport(1, added=(_A,)))
        authority.on_holding("wkr-1", HoldingReport(2, dropped=(_A,)))
        await asyncio.sleep(0.3)

    asyncio.run(run())

    assert _held(directory, _A) == []
    assert directory.applied_seq("wkr-1") == 2


def test_a_report_already_applied_changes_nothing() -> None:
    authority, workers, directory = _setup("wkr-1")
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(1, added=(_A,))))
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(2, dropped=(_A,))))

    asyncio.run(authority.record_holding("wkr-1", HoldingReport(1, added=(_A,))))

    assert _held(directory, _A) == []
    assert workers.resyncs() == []


def test_a_skipped_report_is_applied_and_the_whole_holding_asked_for() -> None:
    authority, workers, directory = _setup("wkr-1")
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(1, added=(_A,))))

    asyncio.run(authority.record_holding("wkr-1", HoldingReport(3, added=(_B,))))

    assert _held(directory, _B) == ["wkr-1"]
    assert workers.resyncs() == ["wkr-1"]


def test_a_whole_report_replaces_the_records_whatever_its_number() -> None:
    authority, workers, directory = _setup("wkr-1")
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(7, added=(_A,))))

    asyncio.run(
        authority.record_holding("wkr-1", HoldingReport(1, added=(_B,), whole=True))
    )
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(2, dropped=(_B,))))

    assert (_held(directory, _A), _held(directory, _B)) == ([], [])
    assert workers.resyncs() == ["wkr-1"]  # 7 followed nothing applied


def test_a_heartbeat_naming_an_unapplied_report_twice_asks_for_the_whole() -> None:
    authority, workers, _ = _setup("wkr-1")
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(1, added=(_A,))))

    asyncio.run(authority.check_holding("wkr-1", 1))
    asyncio.run(authority.check_holding("wkr-1", 2))
    assert workers.resyncs() == []
    asyncio.run(authority.check_holding("wkr-1", 2))

    assert workers.resyncs() == ["wkr-1"]


def test_a_heartbeat_check_reads_no_redis_once_a_report_was_applied() -> None:
    authority, _, directory = _setup("wkr-1")
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(1, added=(_A,))))

    async def no_read(worker_id: str) -> int:
        raise AssertionError("a heartbeat read the directory")

    cast(Any, directory).applied_seq_async = no_read
    for _ in range(3):
        asyncio.run(authority.check_holding("wkr-1", 1))


def test_a_restarted_root_reads_the_applied_report_once() -> None:
    redis = fake_redis_client(fakeredis.FakeServer())
    before, _, _ = _setup("wkr-1", redis=redis)
    asyncio.run(before.record_holding("wkr-1", HoldingReport(4, added=(_A,))))

    after, workers, directory = _setup("wkr-1", redis=redis)
    reads: list[str] = []
    read = directory.applied_seq_async

    async def counted(worker_id: str) -> int:
        reads.append(worker_id)
        return await read(worker_id)

    cast(Any, directory).applied_seq_async = counted
    for _ in range(3):
        asyncio.run(after.check_holding("wkr-1", 4))
    asyncio.run(after.record_holding("wkr-1", HoldingReport(5, dropped=(_A,))))

    assert reads == ["wkr-1"]
    assert workers.resyncs() == []
    assert _held(directory, _A) == []


def test_the_sweep_removes_every_worker_no_longer_registered() -> None:
    """Reaped, unregistered, or gone while control was down: all leave alike."""
    redis = fake_redis_client(fakeredis.FakeServer())
    before, old, _ = _setup("wkr-1", "wkr-2", "wkr-3", redis=redis)
    for worker_id in ("wkr-1", "wkr-2", "wkr-3"):
        asyncio.run(before.record_holding(worker_id, HoldingReport(1, added=(_A,))))
    old.unregister("wkr-3")  # while control was down

    authority, workers, directory = _setup("wkr-1", "wkr-2", redis=redis)
    workers.unregister("wkr-1")  # reaped
    workers.unregister("wkr-2")  # unregistered
    workers.register("wkr-4")
    asyncio.run(authority.record_holding("wkr-4", HoldingReport(1, added=(_A,))))

    assert asyncio.run(authority.sweep_unregistered()) == 3
    assert _held(directory, _A) == ["wkr-4"]
    assert asyncio.run(authority.sweep_unregistered()) == 0


def test_a_late_report_for_a_swept_worker_writes_nothing() -> None:
    authority, workers, directory = _setup("wkr-1")
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(1, added=(_A,))))
    workers.unregister("wkr-1")
    asyncio.run(authority.sweep_unregistered())
    # The registry read raced the removal: the lookup still finds the worker.
    workers._live.add("wkr-1")

    asyncio.run(authority.record_holding("wkr-1", HoldingReport(2, added=(_B,))))

    assert _held(directory, _B) == []
    assert directory.unregistered() == []


def test_binding_the_loop_sweeps_at_once() -> None:
    authority, workers, directory = _setup("wkr-1")
    asyncio.run(authority.record_holding("wkr-1", HoldingReport(1, added=(_A,))))
    workers.unregister("wkr-1")

    async def run() -> None:
        authority.bind_loop(asyncio.get_running_loop())
        for _ in range(100):
            if not _held(directory, _A):
                return
            await asyncio.sleep(0.01)

    asyncio.run(run())

    assert _held(directory, _A) == []
