"""The per-node reverse-relay attachment that runs on the supervisor.

A node attaches outward to the root once — a standing consumer of its own ``:down``
stream behind an ownership lease and a durable cursor — and multiplexes every resident
relay session over it. The attachment neither dials a peer nor accepts an inbound
connection: it reads frames the root bridged to this node and dispatches each to a local
delivery handler (the co-located sidecar for a session this node targets, or the waiting
deputy for one this node originated), and it publishes response frames to its ``:up``
stream for the root to bridge onward. Recovery is the durable cursor and lease, not a
node command; a restart reclaims the lease and resumes from the stored position.
"""

import asyncio
import logging
from typing import Protocol

from ...network.reverse_relay import (
    BinaryRedis,
    RelayDirection,
    RelayFrame,
    RelayLease,
    RelayStreamStore,
)


class LocalDelivery(Protocol):
    """Handles a down frame the root bridged to this node."""

    async def on_frame(self, frame: RelayFrame) -> None: ...


class _DownCursor:
    """The attachment's durable read position for its own down stream."""

    def __init__(self, redis: BinaryRedis, node_id: str) -> None:
        self._redis = redis
        self._key = f"rr:node:{node_id}:down_cursor"

    async def get(self) -> str:
        raw = await self._redis.hgetall(self._key)
        value = raw.get(b"id")
        return value.decode() if value else "0"

    async def set(self, entry_id: str) -> None:
        await self._redis.hset(self._key, mapping={"id": entry_id})


class ResidentRelayAttachment:
    """A node's standing outbound attachment consumer over its reverse-relay streams."""

    _LEG = "down"

    def __init__(
        self,
        redis: BinaryRedis,
        node_id: str,
        delivery: LocalDelivery,
        *,
        owner: str,
        batch: int = 64,
        poll_ms: int = 1000,
        lease_ttl_ms: int = 15000,
        logger: logging.Logger | None = None,
    ) -> None:
        self._node_id = node_id
        self._delivery = delivery
        self._owner = owner
        self._batch = batch
        self._poll_ms = poll_ms
        self._streams = RelayStreamStore(redis)
        self._lease = RelayLease(redis, ttl_ms=lease_ttl_ms)
        self._cursor = _DownCursor(redis, node_id)
        self._logger = logger or logging.getLogger("resident-relay-attachment")
        self._task: asyncio.Task[None] | None = None

    async def send_up(self, frame: RelayFrame) -> None:
        """Publish a response frame to this node's up stream for the root to bridge."""
        await self._streams.publish_up(self._node_id, frame)

    async def pump_once(self) -> int:
        """Consume one bounded batch of down frames while this node owns the lease.

        Returns the number of frames dispatched, or ``-1`` when the lease is not held so
        a caller can tell "quiet" from "fenced". The read is a bounded long-poll so an
        idle node still loops back to refresh its ownership lease and cut idle latency.
        """
        if not await self._lease.acquire(self._node_id, self._LEG, self._owner):
            if not await self._lease.refresh(self._node_id, self._LEG, self._owner):
                return -1
        after = await self._cursor.get()
        entries, last_id = await self._streams.read_down(
            self._node_id, after, count=self._batch, block_ms=self._poll_ms
        )
        if last_id is None:
            return 0
        # Fence dispatch on continued ownership: if the lease lapsed during the read, a
        # successor owns the leg, so drop this batch undelivered and unadvanced rather
        # than deliver under the successor — it re-reads from the unchanged cursor.
        if not await self._lease.owns(self._node_id, self._LEG, self._owner):
            return -1
        for entry in entries:
            try:
                await self._delivery.on_frame(entry.frame)
            except Exception:
                # A frame that cannot be delivered (its sidecar is gone, a bridge writer
                # is closed) must never stall the node's stream or re-deliver the batch;
                # advance past it and let its session recover on the re-drive path.
                self._logger.exception(
                    "relay frame delivery failed; skipping session=%s",
                    entry.frame.session_id,
                )
        await self._cursor.set(last_id)
        await self._streams.trim_up_to(
            self._node_id, RelayDirection.TARGET_TO_ORIGIN, last_id
        )
        return len(entries)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._task is None:
            self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self._lease.release(self._node_id, self._LEG, self._owner)

    async def _run(self) -> None:
        try:
            while True:
                try:
                    count = await self.pump_once()
                except Exception:
                    self._logger.exception("resident relay attachment pump failed")
                    count = -1
                # A quiet or busy pump paces itself on the long-poll read; only a fenced
                # pump (the lease is held elsewhere) backs off before retrying.
                if count < 0:
                    await asyncio.sleep(self._poll_ms / 1000.0)
        except asyncio.CancelledError:
            return


__all__ = ["LocalDelivery", "ResidentRelayAttachment"]
