"""The Redis clients' key helpers, run by a real Redis.

Setting ``FLOWMESH_TEST_REDIS_URL`` points them at a Redis whose ``test:redis-client:*``
keys they own; without it they skip.
"""

import asyncio
import logging
import os
from collections.abc import Iterator

import pytest
import redis

from server.clients.redis import AsyncRedisClient, SyncRedisClient

_LIVE_URL = os.getenv("FLOWMESH_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(
    not _LIVE_URL, reason="FLOWMESH_TEST_REDIS_URL is not set"
)

_KEY = "test:redis-client:vault"
_LOGGER = logging.getLogger("redis-client-keys")


@pytest.fixture
def live() -> Iterator[redis.Redis]:
    assert _LIVE_URL is not None
    client = redis.Redis.from_url(_LIVE_URL, decode_responses=True)
    client.delete(_KEY)
    client.hset(_KEY, mapping={"old": "1"})
    client.expire(_KEY, 600)
    yield client
    client.delete(_KEY)


def test_a_sync_persistent_write_clears_a_leftover_expiry(live: redis.Redis):
    assert _LIVE_URL is not None
    client = SyncRedisClient(_LIVE_URL, _LIVE_URL, _LOGGER)

    client.hash_set_persistent(_KEY, {"new": "2"})

    assert live.ttl(_KEY) == -1
    assert live.hgetall(_KEY) == {"old": "1", "new": "2"}
    assert _KEY in client.scan_keys("test:redis-client:*")
    live.expire(_KEY, 600)
    assert client.persist(_KEY) and live.ttl(_KEY) == -1


def test_an_async_persistent_write_clears_a_leftover_expiry(live: redis.Redis):
    url = _LIVE_URL
    assert url is not None

    async def write() -> list[str]:
        client = AsyncRedisClient(url, url, _LOGGER)
        await client.hash_set_persistent(_KEY, {"new": "2"})
        return await client.scan_keys("test:redis-client:*")

    keys = asyncio.run(write())

    assert live.ttl(_KEY) == -1
    assert live.hgetall(_KEY) == {"old": "1", "new": "2"}
    assert _KEY in keys
