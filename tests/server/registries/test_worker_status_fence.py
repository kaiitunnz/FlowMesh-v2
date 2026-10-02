"""The worker status scripts, run by a real Redis.

The dispatcher's reservation fences the status a worker reports: an IDLE applies only
when it names the dispatch the worker is reserved for. These run each script against
a live server, since the fence is the scripts' own logic. Setting
``FLOWMESH_TEST_REDIS_URL`` points them at a Redis whose worker keys they own; without
it they skip, and the argument-shape tests in ``test_worker_registry_guards`` still run.
"""

import os
from collections.abc import Iterator
from typing import Any, cast

import pytest
import redis

from server.clients.redis import WORKERS_SET_KEY, worker_hb_key, worker_key
from server.registries.worker import (
    ReportOutcome,
    Reservation,
    StatusReport,
    WorkerRegistry,
)
from shared.schemas.worker import WorkerStatus

_LIVE_URL = os.getenv("FLOWMESH_TEST_REDIS_URL")

_live = pytest.mark.skipif(not _LIVE_URL, reason="FLOWMESH_TEST_REDIS_URL is not set")

_WORKER = "wkr-fence"


@pytest.mark.skipif(not os.getenv("CI"), reason="only CI is expected to run Redis")
def test_ci_runs_the_scripts_against_a_real_redis() -> None:
    assert _LIVE_URL, (
        "FLOWMESH_TEST_REDIS_URL is unset in CI: the Redis service is not running, so "
        "nothing here runs the status fence"
    )


class _Sync:
    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    def eval(self, *args: Any) -> Any:
        return self._client.eval(*args)

    def set_members(self, key: str) -> set[str]:
        return cast(set[str], self._client.smembers(key))

    def control_pipeline(self) -> Any:
        return self._client.pipeline()

    def hash_getall(self, key: str) -> dict[str, str]:
        return cast(dict[str, str], self._client.hgetall(key))

    def hash_mget(self, key: str, fields: list[str]) -> list[Any]:
        return cast(list[Any], self._client.hmget(key, fields))

    def publish_control(self, *_args: Any) -> int:
        return 1

    def publish_telemetry(self, *_args: Any) -> None:
        return None


class _Rds:
    def __init__(self, client: redis.Redis) -> None:
        self.sync = _Sync(client)


@pytest.fixture
def client() -> Iterator[redis.Redis]:
    assert _LIVE_URL is not None
    live = redis.Redis.from_url(_LIVE_URL, decode_responses=True)
    keys = (worker_key(_WORKER), worker_hb_key(_WORKER))
    live.delete(*keys)
    live.srem(WORKERS_SET_KEY, _WORKER)
    live.sadd(WORKERS_SET_KEY, _WORKER)
    live.hset(worker_key(_WORKER), mapping={"status": "IDLE"})
    live.setex(worker_hb_key(_WORKER), 120, "ts")
    yield live
    live.delete(*keys)
    live.srem(WORKERS_SET_KEY, _WORKER)
    live.close()


@pytest.fixture
def registry(client: redis.Redis) -> WorkerRegistry:
    return WorkerRegistry(cast(Any, _Rds(client)))


def _status(client: redis.Redis) -> str | None:
    return cast(str | None, client.hget(worker_key(_WORKER), "status"))


def _idle(registry: WorkerRegistry, dispatch_id: str | None) -> StatusReport:
    return registry.update_worker_hb(_WORKER, "ts", 120, WorkerStatus.IDLE, dispatch_id)


_APPLIED = StatusReport(ReportOutcome.APPLIED)


@_live
def test_an_idle_for_an_earlier_dispatch_is_fenced_by_the_reservation(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    assert registry.reserve_worker(_WORKER, "tsk-2", "dsp-2")

    assert _idle(registry, "dsp-1") == StatusReport(
        ReportOutcome.FENCED, "tsk-2", "dsp-2"
    )
    assert _status(client) == "BUSY"
    # The heartbeat still counts as liveness.
    assert cast(int, client.ttl(worker_hb_key(_WORKER))) > 0


@_live
def test_the_idle_ending_the_reserved_dispatch_frees_the_worker(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.reserve_worker(_WORKER, "tsk-2", "dsp-2")

    assert _idle(registry, "dsp-2") == _APPLIED
    assert _status(client) == "IDLE"
    assert client.hget(worker_key(_WORKER), "reserved_dispatch") is None


@_live
def test_a_late_busy_never_moves_the_reservation(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.reserve_worker(_WORKER, "tsk-2", "dsp-2")

    busy = registry.set_worker_status(_WORKER, WorkerStatus.BUSY, "ts", None, "dsp-1")

    assert busy == _APPLIED
    assert client.hget(worker_key(_WORKER), "reserved_dispatch") == "dsp-2"
    assert _idle(registry, "dsp-1").outcome is ReportOutcome.FENCED


@_live
def test_a_busy_heartbeat_behind_its_own_idle_heals_on_the_next(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.reserve_worker(_WORKER, "tsk-1", "dsp-1")
    assert _idle(registry, "dsp-1") == _APPLIED
    # A heartbeat built before that IDLE lands after it.
    registry.update_worker_hb(_WORKER, "ts", 120, WorkerStatus.BUSY, "dsp-1")
    assert _status(client) == "BUSY"

    assert _idle(registry, "dsp-1") == _APPLIED
    assert _status(client) == "IDLE"


@_live
def test_a_heartbeat_without_a_status_changes_none(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.reserve_worker(_WORKER, "tsk-1", "dsp-1")

    assert registry.update_worker_hb(_WORKER, "ts", 120) == _APPLIED
    assert _status(client) == "BUSY"


@_live
def test_an_idle_naming_no_dispatch_applies_only_unreserved(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.set_worker_status(_WORKER, WorkerStatus.BUSY, "ts", None)
    assert _idle(registry, None) == _APPLIED
    registry.reserve_worker(_WORKER, "tsk-1", "dsp-1")

    assert _idle(registry, None).outcome is ReportOutcome.FENCED
    assert _status(client) == "BUSY"


@_live
def test_a_release_frees_only_its_own_reservation(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.reserve_worker(_WORKER, "tsk-2", "dsp-2")

    assert registry.release_worker(_WORKER, "dsp-1") is False
    assert _status(client) == "BUSY"
    assert registry.release_worker(_WORKER, "dsp-2") is True
    assert _status(client) == "IDLE"
    assert client.hget(worker_key(_WORKER), "reserved_dispatch") is None


@_live
def test_an_unregistered_worker_gets_nothing_written(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    client.srem(WORKERS_SET_KEY, _WORKER)
    client.delete(worker_key(_WORKER))

    assert _idle(registry, "dsp-1").outcome is ReportOutcome.UNKNOWN
    assert registry.reserve_worker(_WORKER, "tsk-1", "dsp-1") is False
    assert registry.release_worker(_WORKER, "dsp-1") is False
    assert client.exists(worker_key(_WORKER)) == 0


@_live
def test_a_release_returns_the_worker_to_the_status_it_last_reported(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.reserve_worker(_WORKER, "tsk-1", "dsp-1")
    # A draining worker reports itself busy with the dispatch it gives up.
    registry.set_worker_status(_WORKER, WorkerStatus.BUSY, "ts", None, "dsp-1")

    assert registry.release_worker(_WORKER, "dsp-1") is True
    assert _status(client) == "BUSY"
    assert client.hget(worker_key(_WORKER), "reserved_dispatch") is None


@_live
def test_a_release_applies_a_fenced_idle(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    registry.reserve_worker(_WORKER, "tsk-1", "dsp-1")
    # A worker that names no dispatch is fenced while reserved.
    assert _idle(registry, None).outcome is ReportOutcome.FENCED

    assert registry.release_worker(_WORKER, "dsp-1") is True
    assert _status(client) == "IDLE"


@_live
def test_reservations_lists_each_reserved_worker(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    assert registry.reservations() == []
    registry.reserve_worker(_WORKER, "tsk-1", "dsp-1")

    assert registry.reservations() == [Reservation(_WORKER, "tsk-1", "dsp-1")]


@_live
def test_reservation_names_a_worker_s_current_dispatch(
    client: redis.Redis, registry: WorkerRegistry
) -> None:
    assert registry.reservation(_WORKER) is None
    registry.reserve_worker(_WORKER, "tsk-1", "dsp-1")
    assert registry.reservation(_WORKER) == Reservation(_WORKER, "tsk-1", "dsp-1")

    registry.release_worker(_WORKER, "dsp-1")

    assert registry.reservation(_WORKER) is None
