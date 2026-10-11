"""Retries of the retirements a store failure interrupted."""

import asyncio
import logging
from collections.abc import Callable

DEFAULT_BASE_DELAY_SEC = 1.0
DEFAULT_MAX_DELAY_SEC = 30.0


class RetirementRetry:
    """Retries each failed retirement on the control loop, with capped backoff.

    ``retire`` takes a retirement's kind and key and raises when the store fails again;
    it is idempotent, so a retry after a reply was lost converges on the same result.
    Pending retirements run together after ``base_delay_sec``, the delay doubling up to
    ``max_delay_sec`` while a retry fails. Use from the loop's own thread.
    """

    def __init__(
        self,
        retire: Callable[[str, str], None],
        logger: logging.Logger,
        *,
        base_delay_sec: float = DEFAULT_BASE_DELAY_SEC,
        max_delay_sec: float = DEFAULT_MAX_DELAY_SEC,
    ) -> None:
        self._retire = retire
        self._logger = logger
        self._base = base_delay_sec
        self._max = max_delay_sec
        self._delay = base_delay_sec
        self._pending: dict[tuple[str, str], None] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handle: asyncio.TimerHandle | None = None

    @property
    def pending(self) -> list[tuple[str, str]]:
        return list(self._pending)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._arm()

    def add(self, kind: str, key: str) -> None:
        """Retry a retirement after the current backoff."""
        self._pending[(kind, key)] = None
        self._arm()

    def run(self) -> None:
        """Retry every pending retirement immediately."""
        self.cancel()
        failed = False
        for kind, key in list(self._pending):
            try:
                self._retire(kind, key)
            except Exception:
                self._logger.warning(
                    "retrying the retirement of resident %s %s failed",
                    kind,
                    key,
                    exc_info=True,
                )
                failed = True
                continue
            self._pending.pop((kind, key), None)
        self._delay = min(self._max, self._delay * 2) if failed else self._base
        self._arm()

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _arm(self) -> None:
        if self._loop is None or self._handle is not None or not self._pending:
            return
        self._handle = self._loop.call_later(self._delay, self.run)
