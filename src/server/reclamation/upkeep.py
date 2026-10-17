"""Upkeep of the multipart registration records: each leaves once its upload closes.

Registration records every upload a writer starts, so the records need upkeep wherever
registration runs, whether or not the store is audited. A pass on the root's own thread
reads each scope's records a page at a time and asks control, never the store, whether
the owner each names still lives: the runtime for a dispatch or the upload obligation
its success left, the finalization index for a finalization. Only a record whose owner
has ended is checked against the store, by its exact UploadId, and only once while it
stays unchanged: a closed upload's record is dropped, and an open one, an upload no
live owner will finish, is remembered and left alone. Listings of open uploads decide
nothing here, since a store may list only some of what it holds open.
"""

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shared.content import ObjectStoreConfig

from ..clients.redis import (
    RECLAIM_ROOT_EPOCH_KEY,
    RECLAIM_SCOPES_KEY,
    RedisClient,
    content_finalization_key,
    content_finalization_scope_key,
)
from .multipart import ClosedUpload, MultipartStore, OwnerKind, OwnerResolver
from .store_admin import StoreAdmin

# How often a pass runs, the most store checks one pass makes, and the pause between
# pages of records.
UPKEEP_INTERVAL_SEC = 300.0
MAX_CHECKS = 500
PAGE_PAUSE_SEC = 0.05


@dataclass
class UpkeepPass:
    """What one pass found."""

    records: int = 0
    ended: int = 0
    """Records whose owner has ended."""
    checked: int = 0
    """Store checks the pass made."""
    dropped: int = 0
    open: int = 0
    """Records whose owner has ended and whose upload the store still holds open."""
    deferred: int = 0
    """Records whose owner has ended that the pass left unchecked for want of budget,
    a key or an answer."""
    complete: bool = False


@dataclass(frozen=True)
class _Ended:
    upload_id: str
    raw: str
    key: str
    dispatch_id: str


class MultipartUpkeep:
    """Drops the records of closed uploads whose owners have ended."""

    def __init__(
        self,
        rds: RedisClient,
        store: MultipartStore,
        admin: StoreAdmin,
        owner_of: OwnerResolver,
        logger: logging.Logger,
        *,
        max_checks: int = MAX_CHECKS,
        page_pause_sec: float = PAGE_PAUSE_SEC,
    ) -> None:
        self._rds = rds
        self._store = store
        self._admin = admin
        self._owner_of = owner_of
        self._logger = logger
        self._max_checks = max_checks
        self._page_pause = page_pause_sec
        self._known_open: dict[str, str] = {}

    def run(
        self,
        epoch: Callable[[], int | None],
        stop: threading.Event,
        interval_sec: float = UPKEEP_INTERVAL_SEC,
    ) -> None:
        """Run a pass on every interval until stopped or a later root starts."""
        if (held := epoch()) is None:
            self._logger.error("Multipart upkeep not started: the root holds no epoch")
            return
        while not stop.is_set():
            try:
                if self._superseded(held):
                    self._logger.info(
                        "Multipart upkeep stopped: a root later than %d started", held
                    )
                    return
                result = self.run_pass(stop)
                if result.dropped or result.open or result.deferred:
                    self._logger.info(
                        "Multipart upkeep: %d records, %d with ended owners, %d "
                        "dropped, %d still open, %d left for a later pass",
                        result.records,
                        result.ended,
                        result.dropped,
                        result.open,
                        result.deferred,
                    )
            except Exception:
                self._logger.warning(
                    "Multipart upkeep pass failed; trying again next interval",
                    exc_info=True,
                )
            if stop.wait(interval_sec):
                return

    def run_pass(self, stop: threading.Event | None = None) -> UpkeepPass:
        """Check the records of ended owners once each, dropping those of closed
        uploads; a set ``stop`` ends the pass after the page in hand."""
        stop = stop or threading.Event()
        result = UpkeepPass()
        still_open: dict[str, str] = {}
        complete = True
        for scope in sorted(self._rds.sync.set_members(RECLAIM_SCOPES_KEY)):
            for page in self._store.records(scope):
                result.records += len(page)
                ended = self._ended(scope, page)
                result.ended += len(ended)
                closed: list[ClosedUpload] = []
                for record in ended:
                    if self._known_open.get(record.upload_id) == record.raw:
                        still_open[record.upload_id] = record.raw
                        result.open += 1
                        continue
                    if not record.key or not complete:
                        result.deferred += 1
                        continue
                    if result.checked >= self._max_checks:
                        result.deferred += 1
                        complete = False
                        continue
                    result.checked += 1
                    try:
                        is_open = self._admin.upload_open(record.key, record.upload_id)
                    except Exception:
                        self._logger.warning(
                            "Multipart upkeep could not ask the store about upload %s",
                            record.upload_id,
                            exc_info=True,
                        )
                        result.deferred += 1
                        complete = False
                        continue
                    if is_open:
                        still_open[record.upload_id] = record.raw
                        result.open += 1
                    else:
                        closed.append(
                            ClosedUpload(
                                record.upload_id, record.raw, record.dispatch_id
                            )
                        )
                result.dropped += self._store.drop(scope, closed)
                if stop.wait(self._page_pause):
                    self._known_open.update(still_open)
                    return result
        if complete:
            self._known_open = still_open
        else:
            self._known_open.update(still_open)
        result.complete = complete
        return result

    def _ended(self, scope: str, page: dict[str, str]) -> list[_Ended]:
        """The records of ``page`` whose owner has ended."""
        ended: list[_Ended] = []
        finalizations: list[tuple[_Ended, str]] = []
        for upload_id, raw in page.items():
            try:
                record: dict[str, Any] = json.loads(raw)
                owner = record.get("owner") or {}
                kind = str(owner.get("kind") or "")
                key = str(record.get("key") or "")
                dispatch_id = str(owner.get("dispatch_id") or "")
            except (ValueError, TypeError, AttributeError):
                continue
            entry = _Ended(upload_id, raw, key, dispatch_id)
            match kind:
                case OwnerKind.DISPATCH | OwnerKind.UPLOAD:
                    task_id = str(owner.get("task_id") or "")
                    registration = str(record.get("registration") or "")
                    resolved = (
                        self._owner_of(task_id, registration, dispatch_id)
                        if task_id and registration and dispatch_id
                        else None
                    )
                    if resolved is None or resolved[0] != scope:
                        ended.append(entry)
                case OwnerKind.FINALIZATION:
                    finalizations.append((entry, str(owner.get("idm") or "")))
                case _:
                    ended.append(entry)
        if finalizations:
            with self._rds.sync.control_pipeline(transaction=False) as pipe:
                for _, idempotency_key in finalizations:
                    pipe.get(content_finalization_scope_key(idempotency_key))
                    pipe.exists(content_finalization_key(scope, idempotency_key))
                replies = pipe.execute()
            for i, (entry, _) in enumerate(finalizations):
                assigned, recorded = replies[2 * i], replies[2 * i + 1]
                if assigned != scope or recorded:
                    ended.append(entry)
        return ended

    def _superseded(self, epoch: int) -> bool:
        current = self._rds.sync.get(RECLAIM_ROOT_EPOCH_KEY)
        return current is not None and int(current) > epoch


def start_multipart_upkeep(
    rds: RedisClient,
    store: ObjectStoreConfig,
    owner_of: OwnerResolver,
    epoch: Callable[[], int | None],
    logger: logging.Logger,
    stop: threading.Event,
) -> threading.Thread | None:
    """Start the upkeep's thread on an S3-compatible store; returns it, or None on a
    store that takes no multipart uploads."""
    if (admin := StoreAdmin.from_config(store)) is None:
        return None
    upkeep = MultipartUpkeep(rds, MultipartStore(rds), admin, owner_of, logger)
    thread = threading.Thread(
        target=upkeep.run,
        args=(epoch, stop),
        name="multipart-upkeep",
        daemon=True,
    )
    thread.start()
    return thread


__all__ = [
    "MAX_CHECKS",
    "UPKEEP_INTERVAL_SEC",
    "MultipartUpkeep",
    "UpkeepPass",
    "start_multipart_upkeep",
]
