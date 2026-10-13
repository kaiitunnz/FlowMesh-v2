"""Where a content transfer's two ends are, for the relay that bridges them.

One record per transfer: the nodes and workers the frames travel between. The root
bridge and each node's attachment read it to route an opaque frame onward, so it lives
in the relay's Redis beside the streams it routes. It names no object and authorizes
nothing, so losing it costs a transfer rather than an object. The record expires on its
own, well after the grant that created it and after the slowest transfer of its object
the relay still counts as moving.

A transfer dialed to the holder's node is bridged by the same record, which also names
the requester node's host, so that node admits only the connection control routed.
"""

import redis
import redis.asyncio as async_redis

from ..clients.redis import content_relay_session_key

# The slowest a relayed transfer is expected to move. Nothing refreshes a routing record
# while frames flow, so its lifetime has to cover the whole of its object at this rate.
_SLOWEST_TRANSFER_BYTES_PER_SEC = 1 << 20


class ContentTransferSessions:
    """The routing record each content transfer is bridged by."""

    def __init__(
        self,
        relay_redis: redis.Redis,
        relay_redis_async: async_redis.Redis,
        *,
        ttl_sec: float,
    ) -> None:
        self._rds = relay_redis
        self._rds_async = relay_redis_async
        self._ttl = ttl_sec

    def _record(
        self,
        session_id: str,
        origin_node: str,
        target_node: str,
        origin_worker: str,
        target_worker: str,
        size_bytes: int,
        origin_host: str,
    ) -> tuple[str, dict[str, str], int]:
        """The record's key, its fields, and how long it lives."""
        ttl = self._ttl + size_bytes / _SLOWEST_TRANSFER_BYTES_PER_SEC
        return (
            content_relay_session_key(session_id),
            {
                "origin_node": origin_node,
                "target_node": target_node,
                "origin_worker": origin_worker,
                "target_worker": target_worker,
                "origin_host": origin_host,
            },
            int(ttl) + 1,
        )

    def open(
        self,
        session_id: str,
        *,
        origin_node: str,
        target_node: str,
        origin_worker: str,
        target_worker: str,
        size_bytes: int,
        origin_host: str = "",
    ) -> None:
        key, mapping, ttl = self._record(
            session_id,
            origin_node,
            target_node,
            origin_worker,
            target_worker,
            size_bytes,
            origin_host,
        )
        pipeline = self._rds.pipeline()
        pipeline.hset(key, mapping=mapping)
        pipeline.expire(key, ttl)
        pipeline.execute()

    async def open_async(
        self,
        session_id: str,
        *,
        origin_node: str,
        target_node: str,
        origin_worker: str,
        target_worker: str,
        size_bytes: int,
        origin_host: str = "",
    ) -> None:
        key, mapping, ttl = self._record(
            session_id,
            origin_node,
            target_node,
            origin_worker,
            target_worker,
            size_bytes,
            origin_host,
        )
        pipeline = self._rds_async.pipeline()
        pipeline.hset(key, mapping=mapping)
        pipeline.expire(key, ttl)
        await pipeline.execute()
