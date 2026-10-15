"""Reports the changes to this worker's cache to control, in the order they happened.

The cache notes each copy it keeps or drops under the lock it hands the reporter, which
it also holds while it removes a copy from disk, and notes a kept copy only while it is
still on disk, so the notes follow the disk. Notes collect for a moment and leave as one
numbered report; a copy kept and then dropped before the report leaves is named once, as
dropped. A whole report — at registration, or when control asks for one because a report
went missing — names every copy and supersedes anything still collecting.
"""

import logging
import threading
from collections.abc import Callable, Sequence

from shared.content import HeldCopy, HoldingReport

# How long notes collect before they leave as one report.
_FLUSH_AFTER_SEC = 0.1
# A report leaves at once when this many copies have changed.
_MAX_CHANGES = 1000

type AnnounceHolding = Callable[[HoldingReport], None]


class HoldingReporter:
    """Batches the changes to this worker's cache into numbered holding reports."""

    def __init__(
        self,
        announce: AnnounceHolding,
        lock: threading.RLock,
        *,
        flush_after_sec: float = _FLUSH_AFTER_SEC,
        max_changes: int = _MAX_CHANGES,
        logger: logging.Logger | None = None,
    ) -> None:
        self._announce = announce
        self._flush_after_sec = flush_after_sec
        self._max_changes = max_changes
        self._logger = logger or logging.getLogger("content-holding")
        self._changed = threading.Condition(lock)
        # Each changed copy, and whether it is held now, in the order it changed.
        self._pending: dict[HeldCopy, bool] = {}
        self._seq = 0
        self._stopped = False
        self._thread: threading.Thread | None = None

    @property
    def seq(self) -> int:
        """The number of the last report under this registration, sent or not."""
        return self._seq

    def start(self) -> None:
        """Send what collects on a cadence from here on."""
        self._thread = threading.Thread(
            target=self._run, name="flowmesh-content-holding", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Send what is still collecting, and stop sending."""
        with self._changed:
            self._flush()
            self._stopped = True
            self._changed.notify()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def kept(self, copy: HeldCopy) -> None:
        self._note(copy, True)

    def dropped(self, copy: HeldCopy) -> None:
        self._note(copy, False)

    def whole(self, held: Sequence[HeldCopy], *, restart: bool = False) -> None:
        """Report every copy held, superseding what is still collecting; with
        ``restart``, number it as the first report of a new registration."""
        with self._changed:
            self._pending.clear()
            if restart:
                self._seq = 0
            self._send(HoldingReport(seq=self._seq + 1, added=tuple(held), whole=True))

    def flush(self) -> None:
        """Send what is collecting now."""
        with self._changed:
            self._flush()

    def _note(self, copy: HeldCopy, held: bool) -> None:
        with self._changed:
            first = not self._pending
            self._pending.pop(copy, None)
            self._pending[copy] = held
            if len(self._pending) >= self._max_changes:
                self._flush()
            elif first:
                self._changed.notify()

    def _run(self) -> None:
        with self._changed:
            while not self._stopped:
                if not self._pending:
                    self._changed.wait()
                    continue
                self._changed.wait(self._flush_after_sec)
                self._flush()

    def _flush(self) -> None:
        if self._stopped or not self._pending:
            return
        added = tuple(copy for copy, held in self._pending.items() if held)
        dropped = tuple(copy for copy, held in self._pending.items() if not held)
        self._pending.clear()
        self._send(HoldingReport(seq=self._seq + 1, added=added, dropped=dropped))

    def _send(self, report: HoldingReport) -> None:
        # Numbered even when it cannot be sent, so control sees the gap and asks for
        # the whole holding.
        self._seq = report.seq
        try:
            self._announce(report)
        except Exception as exc:
            self._logger.warning(
                "could not report content holding %d: %s", report.seq, exc
            )
