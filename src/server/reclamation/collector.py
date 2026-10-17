"""The audit: what the shared store holds that no binding names at scan.

A sweep lists every object, or every object version on a bucket that keeps versions,
of each scope, a page at a time, and tests each content object against the scope's
mark: the bindings a marking pass read, and the members of every bound tree read so
far. An object the mark does not hold is unbound at scan. The mark was taken before
the listing, so an object written and bound since can appear unbound; one modified
within ``RECENT_SEC`` before the mark started, or after, is counted apart as recent. A
scope whose coverage is incomplete reports what it lists and no unbound count. The
sweep also reports the multipart uploads the store holds open, as unknown unless the
store's listing of them is complete. It records what it found in the audit stream, its
log and its metrics, and changes nothing in the store.
"""

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from opentelemetry.metrics import Meter, NoOpMeter

from .audit import AuditTrail
from .closures import TreeClosures
from .marking import MarkPass
from .store_admin import Eligibility, Ineligibility, StoreAdmin, StoredVersion

# How often a sweep runs, how long before the mark an object counts as recent, and the
# pause between listed pages.
SWEEP_INTERVAL_SEC = 3600.0
RECENT_SEC = 3600.0
PAGE_PAUSE_SEC = 0.05
# The most open uploads one sweep sizes by listing their parts.
MAX_SIZED_UPLOADS = 1000


@dataclass
class ScopeAudit:
    """What one sweep found in one scope."""

    scope: str
    complete: bool = False
    gap: str = ""
    """Why the scope's coverage is incomplete, when it is."""
    objects: int = 0
    bytes: int = 0
    bound: int = 0
    """Content objects the mark holds."""
    unbound: int = 0
    """Content objects the mark does not hold, older than recent."""
    unbound_bytes: int = 0
    recent: int = 0
    """Content objects the mark does not hold, modified recently."""
    recent_bytes: int = 0
    foreign: int = 0
    """Keys under the scope that name no content object."""
    stopped: str = ""
    """Why the scope's listing stopped before its last page, when it did."""


@dataclass
class UploadAudit:
    """The multipart uploads a listing of the store names open."""

    known: bool = False
    """Whether the listing names every open upload, so its counts hold."""
    uploads: int = 0
    parts: int = 0
    bytes: int = 0
    unsized: int = 0
    by_scope: dict[str, int] = field(default_factory=dict)


@dataclass
class SweepAudit:
    """What one sweep found."""

    sweep: str
    eligibility: Eligibility
    marked_at: float = 0.0
    """The wall-clock time the sweep's mark started."""
    scopes: list[ScopeAudit] = field(default_factory=list)
    uploads: UploadAudit | None = None
    mark_bytes: int = 0
    seconds: float = 0.0

    @property
    def unbound(self) -> int:
        return sum(scope.unbound for scope in self.scopes)

    @property
    def unbound_bytes(self) -> int:
        return sum(scope.unbound_bytes for scope in self.scopes)

    @property
    def recent(self) -> int:
        return sum(scope.recent for scope in self.scopes)

    @property
    def recent_bytes(self) -> int:
        return sum(scope.recent_bytes for scope in self.scopes)

    @property
    def incomplete(self) -> int:
        return sum(1 for scope in self.scopes if not scope.complete)


class AuditCollector:
    """Lists an eligible store against a mark and reports what it found."""

    def __init__(
        self,
        admin: StoreAdmin | None,
        logger: logging.Logger,
        audit: AuditTrail | None = None,
        meter: Meter | None = None,
        *,
        uploads_complete: bool = False,
        recent_sec: float = RECENT_SEC,
        page_pause_sec: float = PAGE_PAUSE_SEC,
    ) -> None:
        self._admin = admin
        self._logger = logger
        self._audit = audit
        self._uploads_complete = uploads_complete
        self._recent = recent_sec
        self._page_pause = page_pause_sec
        self._last_eligibility: Eligibility | None = None
        meter = meter or NoOpMeter("flowmesh")
        self._unbound = meter.create_gauge(
            "flowmesh.content.reclamation.unbound",
            unit="{object}",
            description="Stored content objects no binding named at the last audit, "
            "past the recent window.",
        )
        self._unbound_bytes = meter.create_gauge(
            "flowmesh.content.reclamation.unbound_bytes",
            unit="By",
            description="Bytes of stored content objects no binding named at the last "
            "audit, past the recent window.",
        )
        self._recent_unbound = meter.create_gauge(
            "flowmesh.content.reclamation.recent_unbound",
            unit="{object}",
            description="Stored content objects no binding named at the last audit, "
            "modified within the hour before its mark.",
        )
        self._recent_unbound_bytes = meter.create_gauge(
            "flowmesh.content.reclamation.recent_unbound_bytes",
            unit="By",
            description="Bytes of stored content objects no binding named at the last "
            "audit, modified within the hour before its mark.",
        )
        self._open_uploads = meter.create_gauge(
            "flowmesh.content.reclamation.open_uploads",
            unit="{upload}",
            description="Multipart uploads the store held open at the last audit, "
            "where its listing of them is complete.",
        )
        self._open_upload_bytes = meter.create_gauge(
            "flowmesh.content.reclamation.open_upload_bytes",
            unit="By",
            description="Bytes of parts open multipart uploads held at the last "
            "audit, where the store's listing of them is complete.",
        )
        self._uploads_known = meter.create_gauge(
            "flowmesh.content.reclamation.open_uploads_known",
            unit="1",
            description="Whether the store's listing of open multipart uploads is "
            "complete, so the open-upload gauges are set.",
        )
        self._incomplete = meter.create_gauge(
            "flowmesh.content.reclamation.incomplete_scopes",
            unit="{scope}",
            description="Scopes whose coverage the last audit could not complete, so "
            "it reported no unbound count for them.",
        )
        self._eligible = meter.create_gauge(
            "flowmesh.content.reclamation.eligible",
            unit="1",
            description="Whether the store could be audited at the last audit.",
        )
        self._mark_bytes = meter.create_gauge(
            "flowmesh.content.reclamation.mark_bytes",
            unit="By",
            description="Bytes the last audit's mark held, bound trees' members "
            "included.",
        )
        self._sweep_seconds = meter.create_gauge(
            "flowmesh.content.reclamation.sweep_seconds",
            unit="s",
            description="How long the last audit took, marking included.",
        )

    def eligibility(self) -> Eligibility:
        """What the store lets an audit list, logged when it changes."""
        eligibility = (
            self._admin.eligibility()
            if self._admin is not None
            else Eligibility(Ineligibility.FILESYSTEM)
        )
        if eligibility != self._last_eligibility:
            self._log_eligibility(eligibility)
            self._last_eligibility = eligibility
        self._eligible.set(1 if eligibility.eligible else 0)
        return eligibility

    def _log_eligibility(self, eligibility: Eligibility) -> None:
        if not eligibility.eligible:
            assert eligibility.reason is not None
            self._logger.warning(
                "Content audit cannot list the store (%s%s)",
                eligibility.reason.value,
                f": {eligibility.detail}" if eligibility.detail else "",
            )
            return
        assert eligibility.listing is not None
        self._logger.info(
            "Content audit lists the store's %s (versioning %s); its open-upload "
            "listing is %s",
            eligibility.listing.value,
            eligibility.versioning,
            "complete" if self._uploads_complete else "incomplete",
        )

    def sweep(
        self,
        eligibility: Eligibility,
        marked: MarkPass,
        closures: TreeClosures,
        tree_gaps: Mapping[str, str],
        stop: threading.Event | None = None,
    ) -> SweepAudit:
        """List every scope of an eligible store against ``marked`` and the bound
        trees in ``closures``; ``tree_gaps`` names the scopes whose trees could not
        all be read. A set ``stop`` ends the sweep after the page in hand."""
        stop = stop or threading.Event()
        started = time.monotonic()
        sweep_id = f"{int(marked.started)}"
        result = SweepAudit(sweep_id, eligibility, marked_at=marked.started)
        result.mark_bytes = marked.mark.nbytes + closures.mark.nbytes
        if self._audit is not None and not eligibility.eligible:
            assert eligibility.reason is not None
            self._audit.record(
                "ineligible",
                sweep=sweep_id,
                reason=eligibility.reason.value,
                detail=eligibility.detail,
            )
        if eligibility.eligible:
            assert self._admin is not None
            scopes = set(self._admin.scopes()) | marked.mark.scopes()
            for scope in sorted(scopes):
                if stop.is_set():
                    break
                audit = self._scope(
                    eligibility, scope, marked, closures, tree_gaps, stop
                )
                result.scopes.append(audit)
                if self._audit is not None:
                    self._audit.record(
                        "scope",
                        sweep=sweep_id,
                        recent_before=_recent_boundary(marked.started, self._recent),
                        **asdict(audit),
                    )
            if not stop.is_set():
                self._audit_uploads(result, sweep_id)
        result.seconds = marked.seconds + time.monotonic() - started
        self._publish(result)
        return result

    def _scope(
        self,
        eligibility: Eligibility,
        scope: str,
        marked: MarkPass,
        closures: TreeClosures,
        tree_gaps: Mapping[str, str],
        stop: threading.Event,
    ) -> ScopeAudit:
        audit = ScopeAudit(scope)
        audit.gap = marked.gap(scope) or tree_gaps.get(scope, "")
        audit.complete = not audit.gap
        assert self._admin is not None and eligibility.listing is not None
        recent_after = marked.started - self._recent
        for page in self._admin.listed(eligibility.listing, scope):
            self._page(scope, page, marked, closures, recent_after, audit)
            if stop.wait(self._page_pause):
                audit.stopped = "stopped"
                break
        if not audit.complete:
            audit.unbound = audit.unbound_bytes = audit.recent = audit.recent_bytes = 0
        return audit

    def _page(
        self,
        scope: str,
        page: list[StoredVersion],
        marked: MarkPass,
        closures: TreeClosures,
        recent_after: float,
        audit: ScopeAudit,
    ) -> None:
        for version in page:
            audit.objects += 1
            audit.bytes += version.size_bytes
            if (digest := version.digest) is None:
                audit.foreign += 1
            elif marked.mark.binds(scope, digest) or closures.mark.binds(scope, digest):
                audit.bound += 1
            elif version.last_modified >= recent_after:
                audit.recent += 1
                audit.recent_bytes += version.size_bytes
            else:
                audit.unbound += 1
                audit.unbound_bytes += version.size_bytes

    def _audit_uploads(self, result: SweepAudit, sweep_id: str) -> None:
        result.uploads = self._uploads()
        if self._audit is not None:
            self._audit.record("uploads", sweep=sweep_id, **asdict(result.uploads))

    def _uploads(self) -> UploadAudit:
        assert self._admin is not None
        audit = UploadAudit(known=self._uploads_complete)
        by_scope: dict[str, int] = defaultdict(int)
        for page in self._admin.uploads():
            for upload in page:
                audit.uploads += 1
                by_scope[upload.scope or ""] += 1
                if audit.uploads > MAX_SIZED_UPLOADS:
                    audit.unsized += 1
                    continue
                try:
                    parts, size = self._admin.upload_bytes(upload)
                except Exception:
                    audit.unsized += 1
                    continue
                audit.parts += parts
                audit.bytes += size
        audit.by_scope = dict(by_scope)
        return audit

    def _publish(self, result: SweepAudit) -> None:
        self._unbound.set(result.unbound)
        self._unbound_bytes.set(result.unbound_bytes)
        self._recent_unbound.set(result.recent)
        self._recent_unbound_bytes.set(result.recent_bytes)
        self._incomplete.set(result.incomplete)
        self._mark_bytes.set(result.mark_bytes)
        self._sweep_seconds.set(result.seconds)
        uploads = result.uploads or UploadAudit()
        self._uploads_known.set(1 if uploads.known else 0)
        if uploads.known:
            self._open_uploads.set(uploads.uploads)
            self._open_upload_bytes.set(uploads.bytes)
        self._logger.info(
            "Content audit %s: %d scopes (%d incomplete), %d objects unbound at scan "
            "(%d bytes), %d more modified within %.0fs before the mark (%d bytes), "
            "%s open uploads (%d listed, %d bytes), in %.1fs",
            result.sweep,
            len(result.scopes),
            result.incomplete,
            result.unbound,
            result.unbound_bytes,
            result.recent,
            self._recent,
            result.recent_bytes,
            uploads.uploads if uploads.known else "unknown",
            uploads.uploads,
            uploads.bytes,
            result.seconds,
        )
        if self._audit is not None:
            self._audit.record(
                "sweep",
                sweep=result.sweep,
                recent_before=_recent_boundary(result.marked_at, self._recent),
                scopes=len(result.scopes),
                incomplete=result.incomplete,
                unbound=result.unbound,
                unbound_bytes=result.unbound_bytes,
                recent=result.recent,
                recent_bytes=result.recent_bytes,
                uploads=uploads.uploads,
                upload_bytes=uploads.bytes,
                uploads_known=uploads.known,
                mark_bytes=result.mark_bytes,
                seconds=round(result.seconds, 3),
            )


def _recent_boundary(marked_at: float, recent_sec: float) -> str:
    """The audit-stream statement of what "recent" covers."""
    return (
        f"modified at or after {marked_at - recent_sec:.0f}, {recent_sec:.0f}s before "
        f"the mark started at {marked_at:.0f}"
    )


__all__ = [
    "RECENT_SEC",
    "SWEEP_INTERVAL_SEC",
    "AuditCollector",
    "ScopeAudit",
    "SweepAudit",
    "UploadAudit",
]
