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
from .closures import Expansion, OpenContent, TreeClosures
from .collector import SWEEP_INTERVAL_SEC, AuditCollector, SweepAudit
from .mark import Mark
from .marking import Marker, MarkPass
from .store_admin import StoreAdmin, uploads_listing_complete


class ReclamationService:
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
    ) -> None:
        self._marker = marker
        self._closures = closures
        self._collector = collector
        self._logger = logger
        self._interval = interval_sec
        self._clock = clock
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
        eligibility = self._collector.eligibility()
        if not eligibility.eligible:
            return self._collector.sweep(
                eligibility, MarkPass(Mark(), self._clock()), self._closures, {}
            )
        marked = self._marker.run(stop, self._capacities)
        if marked.complete:
            self._capacities = marked.mark.counts()
        tree_gaps = self._expand(marked.trees.values(), stop)
        return self._collector.sweep(
            eligibility, marked, self._closures, tree_gaps, stop
        )

    def _expand(
        self, trees: Iterable[ContentReference], stop: threading.Event
    ) -> dict[str, str]:
        """Read each bound tree not yet read; returns why each scope's trees are not
        all known."""
        gaps: dict[str, str] = {}
        for tree in trees:
            if stop.is_set():
                gaps.setdefault(tree.authorization_scope, "the sweep stopped")
                continue
            match self._closures.expand(tree, stop):
                case Expansion.EXPANDED:
                    continue
                case Expansion.MALFORMED:
                    why = f"bound tree {tree.content_digest} cannot be read as a tree"
                case Expansion.UNAVAILABLE:
                    why = f"bound tree {tree.content_digest} is unavailable now"
                case Expansion.STOPPED:
                    why = "the sweep stopped"
            gaps.setdefault(tree.authorization_scope, why)
        return gaps


def start_reclamation(
    rds: RedisClient,
    store: ObjectStoreConfig,
    open_content: OpenContent,
    logger: logging.Logger,
    stop: threading.Event,
    meter: Meter | None = None,
) -> threading.Thread:
    """Start the content audit thread; returns it."""
    service = ReclamationService.build(rds, store, open_content, logger, meter)
    thread = threading.Thread(
        target=service.run, args=(stop,), name="content-audit", daemon=True
    )
    thread.start()
    return thread


__all__ = ["ReclamationService", "start_reclamation"]
