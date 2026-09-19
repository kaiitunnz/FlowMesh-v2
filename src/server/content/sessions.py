"""Where a content transfer's two ends are, for the relay that bridges them.

One record per transfer: the nodes and workers the frames travel between. The root
bridge and each node's attachment read it to route an opaque frame onward; it names no
object and authorizes nothing, so losing it costs a transfer rather than an object. The
record expires on its own, well after the grant that created it.
"""

from ..clients.redis import RedisClient, content_relay_session_key


class ContentTransferSessions:
    """The routing record each content transfer is bridged by."""

    def __init__(self, redis: RedisClient, *, ttl_sec: float) -> None:
        self._rds = redis
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
        self._rds.sync.hash_set(
            key,
            {
                "origin_node": origin_node,
                "target_node": target_node,
                "origin_worker": origin_worker,
                "target_worker": target_worker,
            },
        )
        self._rds.sync.expire(key, int(self._ttl) + 1)
