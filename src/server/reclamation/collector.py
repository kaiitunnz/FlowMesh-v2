"""The audit: what the shared store holds that no binding names at scan.

A sweep lists every object of each scope, or every object version on a bucket that
keeps or kept versions, a page at a time, and tests each content object against the
scope's mark: the bindings a marking pass read, and the members of every bound tree
read so far. An object the mark does not hold is unbound at scan. The mark was taken
before the listing, so an object written and bound since can appear unbound; one
modified within ``RECENT_SEC`` before the mark started, or after, is counted apart as
recent. A scope whose coverage or listing is incomplete reports what it lists and no
unbound count. The sweep also reports the multipart uploads the store holds open, as
unknown unless the store's listing of them is complete. It records what it found in
the audit stream, its log and its metrics, and changes nothing in the store.
"""

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from opentelemetry.metrics import Meter, NoOpMeter

from .audit import AuditTrail
from .closures import TreeClosures
from .marking import MarkPass
from .pacing import DUTY, MIN_PAUSE_SEC, Pacer
from .store_admin import (
    Eligibility,
    Eligible,
    Ineligibility,
    Ineligible,
    StoreAdmin,
    StoredVersion,
)

# How often a sweep runs, and how long before the mark an object counts as recent.
SWEEP_INTERVAL_SEC = 3600.0
RECENT_SEC = 3600.0
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
    error: str = ""
    """Why the listing failed, when it did."""


@dataclass
class SweepAudit:
    """What one sweep found."""

    sweep: str
    eligibility: Eligibility
    marked_at: float = 0.0
    """The wall-clock time the sweep's mark started."""
    scopes: list[ScopeAudit] = field(default_factory=list)
    error: str = ""
    """Why the sweep could not find every scope the store holds, when it could not."""
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
        audit: AuditTrail,
        meter: Meter | None = None,
        *,
        uploads_complete: bool = False,
        recent_sec: float = RECENT_SEC,
        duty: float = DUTY,
        min_pause_sec: float = MIN_PAUSE_SEC,
    ) -> None:
        self._admin = admin
        self._logger = logger
        self._audit = audit
        self._uploads_complete = uploads_complete
        self._recent = recent_sec
        self._duty = duty
        self._min_pause = min_pause_sec
        self._last_eligibility: Eligibility | None = None
        meter = meter or NoOpMeter("flowmesh")
        self._unbound = meter.create_gauge(
            "flowmesh.content.audit.unbound",
            unit="{object}",
            description="Stored objects unbound at scan, modified before the recent "
            "window.",
        )
        self._unbound_bytes = meter.create_gauge(
            "flowmesh.content.audit.unbound_bytes",
            unit="By",
            description="Bytes of stored objects unbound at scan, modified before the "
            "recent window.",
        )
        self._recent_unbound = meter.create_gauge(
            "flowmesh.content.audit.recent_unbound",
            unit="{object}",
            description="Stored objects unbound at scan, modified within the recent "
            "window.",
        )
        self._recent_unbound_bytes = meter.create_gauge(
            "flowmesh.content.audit.recent_unbound_bytes",
            unit="By",
            description="Bytes of stored objects unbound at scan, modified within the "
            "recent window.",
        )
        self._open_uploads = meter.create_gauge(
            "flowmesh.content.audit.open_uploads",
            unit="{upload}",
            description="Multipart uploads the store held open at the last audit, "
            "where its listing of them is complete.",
        )
        self._open_upload_bytes = meter.create_gauge(
            "flowmesh.content.audit.open_upload_bytes",
            unit="By",
            description="Bytes of parts open multipart uploads held at the last "
            "audit, where the store's listing of them is complete.",
        )
        self._uploads_known = meter.create_gauge(
            "flowmesh.content.audit.open_uploads_known",
            unit="1",
            description="Whether the store's listing of open multipart uploads is "
            "complete, so the open-upload gauges are set.",
        )
        self._incomplete = meter.create_gauge(
            "flowmesh.content.audit.incomplete_scopes",
            unit="{scope}",
            description="Scopes whose coverage the last audit could not complete, so "
            "it reported no unbound count for them.",
        )
        self._eligible = meter.create_gauge(
            "flowmesh.content.audit.eligible",
            unit="1",
            description="Whether the store could be audited at the last audit.",
        )
        self._mark_bytes = meter.create_gauge(
            "flowmesh.content.audit.mark_bytes",
            unit="By",
            description="Bytes the last audit's mark held, bound trees' members "
            "included.",
        )
        self._sweep_seconds = meter.create_gauge(
            "flowmesh.content.audit.sweep_seconds",
            unit="s",
            description="How long the last audit took, marking included.",
        )

    def eligibility(self) -> Eligibility:
        """What the store lets an audit list, logged when it changes."""
        eligibility = (
            self._admin.eligibility()
            if self._admin is not None
            else Ineligible(Ineligibility.FILESYSTEM)
        )
        if eligibility != self._last_eligibility:
            self._log_eligibility(eligibility)
            self._last_eligibility = eligibility
        return eligibility

    def _log_eligibility(self, eligibility: Eligibility) -> None:
        match eligibility:
            case Ineligible(reason, detail):
                self._logger.warning(
                    "Content audit cannot list the store (%s%s)",
                    reason.value,
                    f": {detail}" if detail else "",
                )
            case Eligible(listing, versioning):
                self._logger.info(
                    "Content audit lists the store's %s (versioning %s); its "
                    "open-upload listing is %s",
                    listing.value,
                    versioning,
                    "complete" if self._uploads_complete else "incomplete",
                )

    def report_ineligible(self, eligibility: Ineligible, at: float) -> SweepAudit:
        """Report a sweep at wall-clock ``at`` that could not list the store, leaving
        the gauges of what an audit finds as the last eligible sweep set them."""
        result = SweepAudit(f"{int(at)}", eligibility)
        self._eligible.set(0)
        self._audit.record(
            "ineligible",
            sweep=result.sweep,
            reason=eligibility.reason.value,
            detail=eligibility.detail,
        )
        return result

    def sweep(
        self,
        eligibility: Eligible,
        marked: MarkPass,
        closures: TreeClosures,
        tree_gaps: Mapping[str, str],
        started: float,
        stop: threading.Event | None = None,
    ) -> SweepAudit:
        """List every scope of an eligible store against ``marked`` and the bound
        trees in ``closures``; ``tree_gaps`` names the scopes whose trees could not
        all be read, and ``started`` is the monotonic time the sweep began. A set
        ``stop`` ends the sweep after the page in hand."""
        if (admin := self._admin) is None:
            raise ValueError("an eligible store needs a store admin to list it")
        stop = stop or threading.Event()
        pacer = Pacer(stop, 1, duty=self._duty, min_pause_sec=self._min_pause)
        sweep_id = f"{int(marked.started)}"
        result = SweepAudit(sweep_id, eligibility, marked_at=marked.started)
        result.mark_bytes = marked.mark.nbytes + marked.trees_nbytes + closures.nbytes
        scopes = set(marked.mark.scopes())
        try:
            for page in admin.scopes(eligibility.listing):
                scopes.update(page)
                if pacer.tick():
                    break
        except Exception as exc:
            result.error = f"the store's scopes could not be listed: {exc}"
            self._logger.warning("Content audit %s", result.error)
        for scope in sorted(scopes):
            if stop.is_set():
                break
            audit = self._scope(
                admin, eligibility, scope, marked, closures, tree_gaps, pacer
            )
            result.scopes.append(audit)
            self._audit.record(
                "scope",
                sweep=sweep_id,
                recent_before=_recent_boundary(marked.started, self._recent),
                **asdict(audit),
            )
        if not stop.is_set():
            result.uploads = self._uploads(admin, pacer)
            self._audit.record("uploads", sweep=sweep_id, **asdict(result.uploads))
        result.seconds = time.monotonic() - started
        self._publish(result)
        return result

    def _scope(
        self,
        admin: StoreAdmin,
        eligibility: Eligible,
        scope: str,
        marked: MarkPass,
        closures: TreeClosures,
        tree_gaps: Mapping[str, str],
        pacer: Pacer,
    ) -> ScopeAudit:
        audit = ScopeAudit(scope)
        audit.gap = marked.gap(scope) or tree_gaps.get(scope, "")
        recent_after = marked.started - self._recent
        try:
            for page in admin.listed(eligibility.listing, scope):
                self._page(scope, page, marked, closures, recent_after, audit)
                if pacer.tick():
                    audit.stopped = "the audit stopped"
                    audit.gap = audit.gap or audit.stopped
                    break
        except Exception as exc:
            audit.stopped = f"the listing failed: {exc}"
            audit.gap = audit.gap or audit.stopped
            self._logger.warning(
                "Content audit could not list scope %s: %s", scope, exc
            )
        audit.complete = not audit.gap
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

    def _uploads(self, admin: StoreAdmin, pacer: Pacer) -> UploadAudit:
        audit = UploadAudit(known=self._uploads_complete)
        try:
            stopped = self._list_uploads(admin, audit, pacer)
        except Exception as exc:
            stopped = f"the listing failed: {exc}"
            self._logger.warning("Content audit could not list open uploads: %s", exc)
        if stopped:
            audit.known = False
            audit.error = stopped
        return audit

    def _list_uploads(self, admin: StoreAdmin, audit: UploadAudit, pacer: Pacer) -> str:
        """Count every open upload into ``audit``, sizing the first few; returns why
        the listing stopped early, or "" when it did not."""
        for page in admin.uploads():
            for upload in page:
                audit.uploads += 1
                scope = upload.scope or ""
                audit.by_scope[scope] = audit.by_scope.get(scope, 0) + 1
                if audit.uploads > MAX_SIZED_UPLOADS:
                    audit.unsized += 1
                    continue
                try:
                    parts, size = admin.upload_bytes(upload)
                except Exception as exc:
                    self._logger.debug(
                        "Content audit could not size upload %s: %s",
                        upload.upload_id,
                        exc,
                    )
                    audit.unsized += 1
                else:
                    audit.parts += parts
                    audit.bytes += size
                if pacer.tick():
                    return "stopped"
            if pacer.tick():
                return "stopped"
        return ""

    def _publish(self, result: SweepAudit) -> None:
        self._eligible.set(1)
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
            error=result.error,
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
