"""Where a content transfer's two ends are, for the relay that bridges them.

One record per transfer: the nodes and workers the frames travel between. The root
bridge and each node's attachment read it to route an opaque frame onward, so it lives
in the relay's Redis beside the streams it routes. It names no object and authorizes
nothing, so losing it costs a transfer rather than an object. The record expires on its
own, well after the grant that created it.
"""

import redis

from ..clients.redis import content_relay_session_key


class ContentTransferSessions:
    """The routing record each content transfer is bridged by."""

    def __init__(self, relay_redis: redis.Redis, *, ttl_sec: float) -> None:
        self._rds = relay_redis
        self._ttl = ttl_sec

    def open(
        self,
        session_id: str,
        *,
        origin_node: str,
        target_node: str,
        origin_worker: str,
        target_worker: str,
    ) -> None:
        key = content_relay_session_key(session_id)
        pipeline = self._rds.pipeline()
        pipeline.hset(
            key,
            mapping={
                "origin_node": origin_node,
                "target_node": target_node,
                "origin_worker": origin_worker,
                "target_worker": target_worker,
            },
        )
        pipeline.expire(key, int(self._ttl) + 1)
        pipeline.execute()
