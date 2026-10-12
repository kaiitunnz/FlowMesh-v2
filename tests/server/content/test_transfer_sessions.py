"""A transfer's routing record lasts as long as its object can take to relay."""

from typing import Any, cast

from server.content.sessions import ContentTransferSessions


class _Pipeline:
    def __init__(self, expiries: dict[str, int]) -> None:
        self._expiries = expiries

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        pass

    def expire(self, key: str, ttl_sec: int) -> None:
        self._expiries[key] = ttl_sec

    def execute(self) -> None:
        pass


class _Redis:
    def __init__(self) -> None:
        self.expiries: dict[str, int] = {}

    def pipeline(self) -> _Pipeline:
        return _Pipeline(self.expiries)


def _open(sessions: ContentTransferSessions, session_id: str, size_bytes: int) -> None:
    sessions.open(
        session_id,
        origin_node="node-a",
        target_node="node-b",
        origin_worker="wkr-a",
        target_worker="wkr-b",
        size_bytes=size_bytes,
    )


def test_a_large_object_s_record_outlives_its_slowest_relay() -> None:
    redis = _Redis()
    sessions = ContentTransferSessions(cast(Any, redis), ttl_sec=600.0)

    _open(sessions, "rly-small", 1024)
    _open(sessions, "rly-large", 5 << 30)

    small, large = (
        redis.expiries[key]
        for key in sorted(redis.expiries, key=lambda k: "large" in k)
    )
    assert 600 <= small <= 602
    # Five GiB at one MiB a second is 5120 seconds on top of the base lifetime.
    assert large >= 600 + 5120
