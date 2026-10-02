"""Redis clients over fakeredis, so a test runs the real Lua scripts."""

import json
import logging
from collections.abc import Callable
from typing import Any, cast

import fakeredis

from server.clients.redis import AsyncRedisClient, RedisClient, SyncRedisClient

_LOGGER = logging.getLogger("test.redis")


def fake_sync_client(server: fakeredis.FakeServer) -> SyncRedisClient:
    client = SyncRedisClient.__new__(SyncRedisClient)
    client._control = fakeredis.FakeRedis(server=server, decode_responses=True)
    client._telemetry = client._control
    client.logger = _LOGGER
    return client


def fake_redis_client(server: fakeredis.FakeServer) -> RedisClient:
    async_client = AsyncRedisClient.__new__(AsyncRedisClient)
    cast(Any, async_client)._control = fakeredis.FakeAsyncRedis(
        server=server, decode_responses=True
    )
    cast(Any, async_client)._telemetry = cast(Any, async_client)._control
    client = RedisClient.__new__(RedisClient)
    client.sync = fake_sync_client(server)
    client.asyncio = async_client
    return client


class RecordingSyncClient(SyncRedisClient):
    """A sync client over fakeredis that records each control publish, handing it to
    ``on_publish`` when set."""

    published: list[dict[str, Any]]
    on_publish: Callable[[dict[str, Any]], None] | None

    def publish_control(self, channel: str, message: str) -> int:
        data = json.loads(message)
        self.published.append(data)
        if self.on_publish is not None:
            self.on_publish(data)
        return 1


def recording_redis_client(
    server: fakeredis.FakeServer,
) -> tuple[RedisClient, RecordingSyncClient]:
    """A Redis client over fakeredis whose control publishes are recorded."""
    client = fake_redis_client(server)
    sync = RecordingSyncClient.__new__(RecordingSyncClient)
    sync._control = fakeredis.FakeRedis(server=server, decode_responses=True)
    sync._telemetry = sync._control
    sync.logger = _LOGGER
    sync.published = []
    sync.on_publish = None
    client.sync = sync
    return client, sync
