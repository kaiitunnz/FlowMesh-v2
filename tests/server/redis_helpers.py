"""Redis clients over fakeredis, so a test runs the real Lua scripts."""

import json
import logging
from collections.abc import Callable
from typing import Any, cast

import fakeredis

from server.clients.redis import AsyncRedisClient, RedisClient, SyncRedisClient

_LOGGER = logging.getLogger("test.redis")


def fake_sync_client(server: fakeredis.FakeServer) -> SyncRedisClient:
    return _sync_client(SyncRedisClient, server)


def restart_fake_server(server: fakeredis.FakeServer, run_id: str) -> None:
    """Give ``server`` the run id a restarted Redis would report."""
    cast(Any, server).fake_run_id = run_id


def _info(server: fakeredis.FakeServer) -> Callable[..., dict[str, str]]:
    """``INFO``, which fakeredis lacks, answering the server's run id."""

    def info(section: str | None = None, *args: Any, **kwargs: Any) -> dict[str, str]:
        run_id = getattr(server, "fake_run_id", "fake-run")
        return {"run_id": run_id, "master_replid": f"{run_id}-replid"}

    return info


def _sync_client[C: SyncRedisClient](cls: type[C], server: fakeredis.FakeServer) -> C:
    client = object.__new__(cls)
    client._control = fakeredis.FakeRedis(server=server, decode_responses=True)
    cast(Any, client._control).info = _info(server)
    client._telemetry = client._control
    client.logger = _LOGGER
    return client


def raw_control(registry: Any) -> Any:
    """A registry's control Redis connection, for reading and corrupting raw keys."""
    # redis-py types a sync reply as possibly awaitable.
    return registry._rds.sync._control


def fake_redis_client(server: fakeredis.FakeServer) -> RedisClient:
    async_client = AsyncRedisClient.__new__(AsyncRedisClient)
    cast(Any, async_client)._control = fakeredis.FakeAsyncRedis(
        server=server, decode_responses=True
    )
    cast(Any, async_client)._telemetry = cast(Any, async_client)._control
    sync_info = _info(server)

    async def info(*args: Any, **kwargs: Any) -> dict[str, str]:
        return sync_info(*args, **kwargs)

    cast(Any, async_client)._control.info = info
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
    sync = _sync_client(RecordingSyncClient, server)
    sync.published = []
    sync.on_publish = None
    client.sync = sync
    return client, sync
