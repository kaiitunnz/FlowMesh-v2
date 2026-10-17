"""The root's content reclamation service: backfill, then periodic audit sweeps.

It runs on its own thread, reading and writing only the control store's reclamation
keys and the bindings it backfills, and listing the shared store; it takes no runtime
lock and has no part in the event path. It starts once the root holds its epoch, marks
the protection hooks live from that epoch, backfills what was bound before, and then
sweeps on an interval. A later root moving the generation on ends it.
"""

import logging
import threading
from collections.abc import Callable

from opentelemetry.metrics import Meter

from shared.content import ObjectStoreConfig

from ..clients.redis import RedisClient
from .audit import AuditTrail
from .backfill import FAMILIES, Backfill
from .certification import CoverageStore
from .collector import SWEEP_INTERVAL_SEC, AuditCollector
from .store_admin import StoreAdmin


class ReclamationService:
    """Backfills protection, then audits the store on an interval."""

    def __init__(
        self,
        epoch: Callable[[], int | None],
        backfill: Backfill,
        collector: AuditCollector,
        coverage: CoverageStore,
        logger: logging.Logger,
        *,
        interval_sec: float = SWEEP_INTERVAL_SEC,
    ) -> None:
        self._epoch = epoch
        self._backfill = backfill
        self._collector = collector
        self._coverage = coverage
        self._logger = logger
        self._interval = interval_sec

    @classmethod
    def build(
        cls,
        rds: RedisClient,
        store: ObjectStoreConfig,
        epoch: Callable[[], int | None],
        logger: logging.Logger,
        meter: Meter | None = None,
    ) -> "ReclamationService":
        coverage = CoverageStore(rds, FAMILIES)
        audit = AuditTrail(rds)
        return cls(
            epoch,
            Backfill(rds, coverage, logger, audit),
            AuditCollector(
                rds, coverage, StoreAdmin.from_config(store), logger, audit, meter
            ),
            coverage,
            logger,
        )

    def run(self, stop: threading.Event) -> None:
        """Mark the hooks live, backfill, then sweep until stopped or superseded."""
        epoch = self._epoch()
        if epoch is None:
            self._logger.error(
                "Content reclamation not started: the root holds no epoch"
            )
            return
        try:
            generation = self._coverage.mark_hooks_on(epoch)
        except Exception:
            self._logger.exception("Content reclamation could not mark its hooks live")
            return
        if generation != epoch:
            self._logger.info(
                "Content reclamation not started: a later root holds generation %d",
                generation,
            )
            return
        if self._backfill.run(generation, stop) is None:
            return
        while not stop.is_set():
            if self._coverage.generation() != generation:
                self._logger.info(
                    "Content reclamation stopped: a later root took the generation"
                )
                return
            try:
                self._collector.sweep(stop)
            except Exception:
                self._logger.warning(
                    "Content reclamation sweep failed; trying again next interval",
                    exc_info=True,
                )
            if stop.wait(self._interval):
                return


def start_reclamation(
    enabled: bool,
    rds: RedisClient,
    store: ObjectStoreConfig,
    epoch: Callable[[], int | None],
    logger: logging.Logger,
    stop: threading.Event,
    meter: Meter | None = None,
) -> threading.Thread | None:
    """Start the reclamation service's thread when ``enabled``; returns it, or None."""
    if not enabled:
        return None
    service = ReclamationService.build(rds, store, epoch, logger, meter)
    thread = threading.Thread(
        target=service.run, args=(stop,), name="content-reclamation", daemon=True
    )
    thread.start()
    return thread


__all__ = ["ReclamationService", "start_reclamation"]
