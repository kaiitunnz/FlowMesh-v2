"""Redis clients over fakeredis, so a test runs the real Lua scripts."""

import logging
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
