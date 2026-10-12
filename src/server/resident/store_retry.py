"""Retries of the resident writes a store failure interrupted."""

import asyncio
import logging
from collections.abc import Callable

DEFAULT_BASE_DELAY_SEC = 1.0
DEFAULT_MAX_DELAY_SEC = 30.0

type Attempt = Callable[[], None]

# The kind of a gated serve settlement entry; retirements use their durable kinds.
SETTLEMENT = "settlement"


class StoreRetry:
    """Retries each write a store failure interrupted on the control loop, with capped
    backoff, until it lands.

    An entry is a kind and key with the attempt that lands it. An attempt is idempotent,
    so a retry after a lost reply converges on the same result; it fails by raising or
    by handing its entry back. Pending entries run together after ``base_delay_sec``,
    the delay doubling up to ``max_delay_sec`` while any fails. Use from the loop's own
    thread.
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        base_delay_sec: float = DEFAULT_BASE_DELAY_SEC,
        max_delay_sec: float = DEFAULT_MAX_DELAY_SEC,
    ) -> None:
        self._logger = logger
        self._base = base_delay_sec
        self._max = max_delay_sec
        self._delay = base_delay_sec
        self._pending: dict[tuple[str, str], Attempt] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handle: asyncio.TimerHandle | None = None

    @property
    def pending(self) -> list[tuple[str, str]]:
        return list(self._pending)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._arm()

    def add(self, kind: str, key: str, attempt: Attempt) -> None:
        """Retry an interrupted write after the current backoff."""
        self._pending[(kind, key)] = attempt
        self._arm()

    def run(self) -> None:
        """Retry every pending write immediately."""
        self.cancel()
        failed = False
        for entry, attempt in list(self._pending.items()):
            del self._pending[entry]
            try:
                attempt()
            except Exception:
                self._logger.warning(
                    "retrying resident %s %s failed", *entry, exc_info=True
                )
                self._pending.setdefault(entry, attempt)
            failed = failed or entry in self._pending
        self._delay = min(self._max, self._delay * 2) if failed else self._base
        self.cancel()
        self._arm()

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _arm(self) -> None:
        if self._loop is None or self._handle is not None or not self._pending:
            return
        self._handle = self._loop.call_later(self._delay, self.run)
