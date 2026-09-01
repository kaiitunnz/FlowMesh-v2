"""Bounded relay-session mechanics for the network plane.

A ``RelaySession`` bridges two stream pairs (a caller side and a target side) with a
bounded in-flight buffer, so a slow consumer backpressures a fast producer rather than
letting the relay grow without limit. It is cancellable and self-cleaning: cancelling or
completing a session closes both writers. It is a dedicated network-plane mechanism used
by the node-relay uplink and the control-relay fallback — not the event/log relay — and
carries only the bytes handed to it, never a resident engine request.
"""

import asyncio
import logging

from shared.utils.time import now_iso

_READ_CHUNK = 16384


class RelaySession:
    """A bounded, cancellable bidirectional byte relay between two stream pairs."""

    def __init__(
        self,
        session_id: str,
        *,
        buffer_bytes: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.session_id = session_id
        self._max_chunks = max(1, buffer_bytes // _READ_CHUNK)
        self._logger = logger
        self._started_at = now_iso()
        self._tasks: list[asyncio.Task[None]] = []
        self._cancelled = False

    async def relay(
        self,
        caller_reader: asyncio.StreamReader,
        caller_writer: asyncio.StreamWriter,
        target_reader: asyncio.StreamReader,
        target_writer: asyncio.StreamWriter,
    ) -> None:
        """Pump both directions until either side closes, then clean up."""
        self._tasks = [
            asyncio.create_task(self._pump(caller_reader, target_writer)),
            asyncio.create_task(self._pump(target_reader, caller_writer)),
        ]
        try:
            _, pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            await self._close(caller_writer)
            await self._close(target_writer)

    def cancel(self) -> None:
        """Stop the session; the writers are closed by ``relay``'s cleanup."""
        self._cancelled = True
        for task in self._tasks:
            task.cancel()

    async def _pump(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """One direction, bounded by ``_max_chunks`` in-flight reads (backpressure)."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=self._max_chunks)

        async def read_side() -> None:
            while True:
                chunk = await reader.read(_READ_CHUNK)
                await queue.put(chunk or None)
                if not chunk:
                    break

        async def write_side() -> None:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                writer.write(chunk)
                await writer.drain()

        await asyncio.gather(read_side(), write_side())

    async def _close(self, writer: asyncio.StreamWriter) -> None:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            if self._logger is not None:
                self._logger.debug(
                    "Relay session %s writer close failed: %s", self.session_id, exc
                )
