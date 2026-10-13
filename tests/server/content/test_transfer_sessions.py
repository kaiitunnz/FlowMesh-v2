"""A transfer's routing record lasts as long as its object can take to relay, and the
store opens one alike from a thread or the loop."""

import asyncio
from typing import Any, cast

import pytest

from server.content.sessions import ContentTransferSessions


class _Writes:
    def __init__(self, redis: "_Redis") -> None:
        self._redis = redis

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._redis.records[key] = mapping

    def expire(self, key: str, ttl_sec: int) -> None:
        self._redis.expiries[key] = ttl_sec


class _Pipeline(_Writes):
    def execute(self) -> None:
        pass


class _AsyncPipeline(_Writes):
    async def execute(self) -> None:
        pass


class _Redis:
    def __init__(self, pipeline: type[_Writes] = _Pipeline) -> None:
        self._pipeline = pipeline
        self.records: dict[str, dict[str, str]] = {}
        self.expiries: dict[str, int] = {}

    def pipeline(self) -> _Writes:
        return self._pipeline(self)


_SESSION = {
    "origin_node": "node-a",
    "target_node": "node-b",
    "origin_worker": "wkr-a",
    "target_worker": "wkr-b",
    "origin_host": "10.0.0.7",
}


def _sessions() -> tuple[ContentTransferSessions, _Redis, _Redis]:
    sync, asynchronous = _Redis(), _Redis(_AsyncPipeline)
    store = ContentTransferSessions(
        cast(Any, sync), cast(Any, asynchronous), ttl_sec=600.0
    )
    return store, sync, asynchronous


def test_a_large_object_s_record_outlives_its_slowest_relay() -> None:
    sessions, _, redis = _sessions()

    for session_id, size_bytes in (("rly-small", 1024), ("rly-large", 5 << 30)):
        asyncio.run(sessions.open_async(session_id, size_bytes=size_bytes, **_SESSION))

    small, large = (
        redis.expiries[key]
        for key in sorted(redis.expiries, key=lambda k: "large" in k)
    )
    assert 600 <= small <= 602
    # Five GiB at one MiB a second is 5120 seconds on top of the base lifetime.
    assert large >= 600 + 5120


@pytest.mark.parametrize("size_bytes", [1024, 5 << 30])
def test_a_record_opened_from_a_thread_matches_one_opened_on_the_loop(
    size_bytes: int,
) -> None:
    sessions, sync, asynchronous = _sessions()

    sessions.open("rly-1", size_bytes=size_bytes, **_SESSION)
    asyncio.run(sessions.open_async("rly-1", size_bytes=size_bytes, **_SESSION))

    assert sync.records == asynchronous.records
    assert sync.expiries == asynchronous.expiries


def test_every_public_method_has_its_twin() -> None:
    public = {
        name
        for name in vars(ContentTransferSessions)
        if not name.startswith("_") and callable(getattr(ContentTransferSessions, name))
    }
    sync = {name for name in public if not name.endswith("_async")}

    assert {f"{name}_async" for name in sync} == public - sync
