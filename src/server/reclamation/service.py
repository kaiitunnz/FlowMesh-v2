"""The root's content audit: mark what the control store binds, then list the store.

It runs on its own thread, off the event path, reading the control store and the shared
store and writing only its audit stream. Each sweep checks what the store lets it list,
marks every binding in one pass, reads each bound tree not read before, and lists the
store against the mark.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterable
from typing import Self

from opentelemetry.metrics import Meter

from shared.content import ContentReference, ObjectStoreConfig

from ..clients.redis import RedisClient
from .audit import AuditTrail
from .closures import READS_PER_PAUSE, Expansion, OpenContent, TreeClosures
from .collector import SWEEP_INTERVAL_SEC, AuditCollector, SweepAudit
from .marking import Marker
from .pacing import DUTY, MIN_PAUSE_SEC, Pacer
from .store_admin import Eligible, Ineligible, StoreAdmin, uploads_listing_complete


class ContentAuditService:
    """Audits the store on an interval."""

    def __init__(
        self,
        marker: Marker,
        closures: TreeClosures,
        collector: AuditCollector,
        logger: logging.Logger,
        *,
        interval_sec: float = SWEEP_INTERVAL_SEC,
        clock: Callable[[], float] = time.time,
        reads_per_pause: int = READS_PER_PAUSE,
        duty: float = DUTY,
        min_pause_sec: float = MIN_PAUSE_SEC,
    ) -> None:
        self._marker = marker
        self._closures = closures
        self._collector = collector
        self._logger = logger
        self._interval = interval_sec
        self._clock = clock
        self._reads_per_pause = reads_per_pause
        self._duty = duty
        self._min_pause = min_pause_sec
        self._capacities: dict[str, int] = {}

    @classmethod
    def build(
        cls,
        rds: RedisClient,
        store: ObjectStoreConfig,
        open_content: OpenContent,
        logger: logging.Logger,
        meter: Meter | None = None,
    ) -> Self:
        return cls(
            Marker(rds, logger),
            TreeClosures(open_content, logger),
            AuditCollector(
                StoreAdmin.from_config(store),
                logger,
                AuditTrail(rds),
                meter,
                uploads_complete=uploads_listing_complete(store),
            ),
            logger,
        )

    def run(self, stop: threading.Event) -> None:
        """Sweep until stopped."""
        while not stop.is_set():
            try:
                self.sweep(stop)
            except Exception:
                self._logger.warning(
                    "Content audit failed; trying again next interval", exc_info=True
                )
            if stop.wait(self._interval):
                return

    def sweep(self, stop: threading.Event) -> SweepAudit:
        """Run one sweep."""
        started = time.monotonic()
        match self._collector.eligibility():
            case Ineligible() as ineligible:
                return self._collector.report_ineligible(ineligible, self._clock())
            case Eligible() as eligible:
                marked = self._marker.run(stop, self._capacities)
                if marked.complete:
                    self._capacities = marked.mark.counts()
                tree_gaps = self._expand(marked.tree_references(), stop)
                return self._collector.sweep(
                    eligible, marked, self._closures, tree_gaps, started, stop
                )

    def _expand(
        self, trees: Iterable[ContentReference], stop: threading.Event
    ) -> dict[str, str]:
        """Read each bound tree not yet read; returns why each scope's trees are not
        all known."""
        pacer = Pacer(
            stop,
            self._reads_per_pause,
            duty=self._duty,
            min_pause_sec=self._min_pause,
        )
        gaps: dict[str, str] = {}
        for tree in trees:
            if pacer.tick():
                gaps.setdefault(tree.authorization_scope, "the sweep stopped")
                continue
            match self._closures.expand(tree, pacer):
                case Expansion.EXPANDED:
                    continue
                case Expansion.MALFORMED:
                    why = f"bound tree {tree.content_digest} cannot be read as a tree"
                case Expansion.UNAVAILABLE:
                    why = f"bound tree {tree.content_digest} is unavailable now"
                case Expansion.STOPPED:
                    why = "the sweep stopped"
            gaps.setdefault(tree.authorization_scope, why)
        self._closures.compact()
        return gaps


def start_content_audit(
    rds: RedisClient,
    store: ObjectStoreConfig,
    open_content: OpenContent,
    logger: logging.Logger,
    stop: threading.Event,
    meter: Meter | None = None,
) -> threading.Thread:
    """Start the content audit thread; returns it."""
    service = ContentAuditService.build(rds, store, open_content, logger, meter)
    thread = threading.Thread(
        target=service.run, args=(stop,), name="content-audit", daemon=True
    )
    thread.start()
    return thread


__all__ = ["ContentAuditService", "start_content_audit"]
