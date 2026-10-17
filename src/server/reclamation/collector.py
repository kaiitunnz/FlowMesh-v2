"""The audit collector: what the shared store holds that no binding names.

A sweep lists every object version of each certified scope of an eligible store, a page
at a time, and condemns each content object that is neither protected, held by an
assertion in flight, nor a bound tree awaiting expansion. Condemnation is one script
that re-checks all of that, and the scope's certification, atomically, so an object a
binding names is never condemned. A condemned object stays cancellable: an assertion
that names it removes it. The collector also reports the multipart uploads the store
holds open, and drops the registration of each upload a listing no longer names. It
deletes nothing and aborts nothing in the store; it records what it found in the audit
stream, its log and its metrics.

An object younger than ``MIN_AGE_SEC`` is passed over, which paces condemnation behind
writers still about to bind what they wrote; no decision rests on that age.
"""

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from opentelemetry.metrics import Meter, NoOpMeter

from ..clients.redis import (
    RECLAIM_HOOKS_ON_KEY,
    RECLAIM_SCOPES_KEY,
    RedisClient,
    reclaim_cert_key,
    reclaim_condemned_key,
    reclaim_deleting_key,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_uncertified_key,
    reclaim_unexpanded_key,
)
from .audit import AuditTrail
from .certification import CoverageStore
from .multipart import MultipartStore
from .store_admin import Eligibility, Ineligibility, StoreAdmin, StoredVersion

# How often a sweep runs, how old an object must be before it is condemned, and the
# pause between listed pages.
SWEEP_INTERVAL_SEC = 3600.0
MIN_AGE_SEC = 3600.0
PAGE_PAUSE_SEC = 0.05
# The most open uploads one sweep sizes by listing their parts.
MAX_SIZED_UPLOADS = 1000

# KEYS: hooks-on, cert, protected, pins, unexpanded, uncertified, deleting, condemned.
# ARGV: sweep, seen_at, n, d1, v1, ..., dn, vn.
_CONDEMN = """
local types = {[3] = 'set', [4] = 'hash', [5] = 'set', [7] = 'hash', [8] = 'hash'}
for i, expected in pairs(types) do
  local found = redis.call('TYPE', KEYS[i]).ok
  if found ~= 'none' and found ~= expected then
    return redis.error_reply('WRONGTYPE ' .. KEYS[i])
  end
end
local generation = redis.call('GET', KEYS[1])
if not generation or redis.call('GET', KEYS[2]) ~= generation
  or redis.call('EXISTS', KEYS[6]) == 1 then
  return {'uncertified'}
end
if redis.call('SCARD', KEYS[5]) > 0 then
  return {'unexpanded'}
end
local out = {'ok'}
for i = 1, tonumber(ARGV[3]) do
  local digest, version = ARGV[2 + 2 * i], ARGV[3 + 2 * i]
  if redis.call('SISMEMBER', KEYS[3], digest) == 1
    or redis.call('HEXISTS', KEYS[4], digest) == 1
    or redis.call('HEXISTS', KEYS[7], digest) == 1 then
    out[#out + 1] = 'K'
  else
    local seen = ARGV[2]
    local existing = redis.call('HGET', KEYS[8], digest)
    if existing then
      local ok, entry = pcall(cjson.decode, existing)
      if ok and type(entry) == 'table' and entry.seen_at then
        seen = entry.seen_at
      end
    end
    redis.call('HSET', KEYS[8], digest,
      cjson.encode({version_id = version, sweep = ARGV[1], seen_at = seen}))
    out[#out + 1] = 'C'
  end
end
return out
"""


# The reasons a bucket is not collected that still let its open uploads be listed.
_UPLOADS_LISTABLE = frozenset(
    {Ineligibility.UNVERSIONED, Ineligibility.VERSIONING_UNKNOWN}
)


@dataclass
class ScopeAudit:
    """What one sweep found in one scope."""

    scope: str
    certified: bool = False
    reason: str = ""
    versions: int = 0
    bytes: int = 0
    kept: int = 0
    """Content versions a binding or an assertion in flight holds."""
    young: int = 0
    """Content versions nothing binds or holds, too recent to condemn."""
    young_bytes: int = 0
    orphans: int = 0
    """Content versions condemned: stored, unbound and old enough."""
    orphan_bytes: int = 0
    foreign: int = 0
    """Keys under the scope that name no content object, never condemned."""
    stopped: str = ""
    """Why the scope's sweep stopped before its last page, when it did."""


@dataclass
class UploadAudit:
    """The multipart uploads the store holds open, by scope."""

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
    scopes: list[ScopeAudit] = field(default_factory=list)
    uploads: UploadAudit | None = None
    seconds: float = 0.0

    @property
    def orphans(self) -> int:
        return sum(scope.orphans for scope in self.scopes)

    @property
    def orphan_bytes(self) -> int:
        return sum(scope.orphan_bytes for scope in self.scopes)

    @property
    def young(self) -> int:
        return sum(scope.young for scope in self.scopes)

    @property
    def young_bytes(self) -> int:
        return sum(scope.young_bytes for scope in self.scopes)

    @property
    def uncertified(self) -> int:
        return sum(1 for scope in self.scopes if not scope.certified)


class AuditCollector:
    """Sweeps an eligible store for unbound objects, condemning and reporting them."""

    def __init__(
        self,
        rds: RedisClient,
        coverage: CoverageStore,
        admin: StoreAdmin | None,
        logger: logging.Logger,
        audit: AuditTrail | None = None,
        meter: Meter | None = None,
        multipart: MultipartStore | None = None,
        *,
        min_age_sec: float = MIN_AGE_SEC,
        page_pause_sec: float = PAGE_PAUSE_SEC,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._rds = rds
        self._multipart = multipart
        self._coverage = coverage
        self._admin = admin
        self._logger = logger
        self._audit = audit
        self._min_age = min_age_sec
        self._page_pause = page_pause_sec
        self._clock = clock
        self._last_ineligible: tuple[Ineligibility, str] | None = None
        meter = meter or NoOpMeter("flowmesh")
        self._orphans = meter.create_gauge(
            "flowmesh.content.reclamation.orphans",
            unit="{object}",
            description="Stored content versions no binding names, as of the last "
            "sweep.",
        )
        self._orphan_bytes = meter.create_gauge(
            "flowmesh.content.reclamation.orphan_bytes",
            unit="By",
            description="Bytes of stored content versions no binding names, as of the "
            "last sweep.",
        )
        self._young = meter.create_gauge(
            "flowmesh.content.reclamation.young",
            unit="{object}",
            description="Stored content versions no binding names that were too "
            "recent to condemn, as of the last sweep.",
        )
        self._young_bytes = meter.create_gauge(
            "flowmesh.content.reclamation.young_bytes",
            unit="By",
            description="Bytes of stored content versions no binding names that were "
            "too recent to condemn, as of the last sweep.",
        )
        self._open_uploads = meter.create_gauge(
            "flowmesh.content.reclamation.open_uploads",
            unit="{upload}",
            description="Multipart uploads the store holds open, as of the last sweep.",
        )
        self._open_upload_bytes = meter.create_gauge(
            "flowmesh.content.reclamation.open_upload_bytes",
            unit="By",
            description="Bytes of parts open multipart uploads hold, as of the last "
            "sweep.",
        )
        self._uncertified = meter.create_gauge(
            "flowmesh.content.reclamation.uncertified_scopes",
            unit="{scope}",
            description="Scopes the last sweep could not collect for want of "
            "certified coverage.",
        )
        self._eligible = meter.create_gauge(
            "flowmesh.content.reclamation.eligible",
            unit="1",
            description="Whether the store was eligible for collection at the last "
            "sweep.",
        )

    def sweep(self, stop: threading.Event | None = None) -> SweepAudit:
        """Run one sweep over every scope; a set ``stop`` ends it after the page in
        hand."""
        stop = stop or threading.Event()
        started = time.monotonic()
        sweep_id = f"{self._coverage.generation() or 0}:{int(self._clock())}"
        eligibility = (
            self._admin.eligibility()
            if self._admin is not None
            else Eligibility(Ineligibility.FILESYSTEM)
        )
        result = SweepAudit(sweep_id, eligibility)
        self._eligible.set(1 if eligibility.eligible else 0)
        if not eligibility.eligible:
            self._report_ineligible(eligibility, sweep_id)
            # Open uploads list without versioning, so their records stay bounded on
            # any bucket the root can list.
            if eligibility.reason in _UPLOADS_LISTABLE and not stop.is_set():
                self._audit_uploads(result, sweep_id)
                self._publish(result)
            return result
        self._last_ineligible = None
        assert self._admin is not None
        scopes = set(self._rds.sync.set_members(RECLAIM_SCOPES_KEY))
        scopes.update(self._admin.scopes())
        for scope in sorted(scopes):
            if stop.is_set():
                break
            audit = self._scope(scope, sweep_id, stop)
            result.scopes.append(audit)
            if self._audit is not None:
                self._audit.record("scope", sweep=sweep_id, **asdict(audit))
        if not stop.is_set():
            self._audit_uploads(result, sweep_id)
        result.seconds = time.monotonic() - started
        self._publish(result)
        return result

    def _report_ineligible(self, eligibility: Eligibility, sweep_id: str) -> None:
        assert eligibility.reason is not None
        state = (eligibility.reason, eligibility.detail)
        if state != self._last_ineligible:
            self._logger.warning(
                "Content reclamation keeps every stored object: the store is not "
                "eligible (%s%s)",
                eligibility.reason.value,
                f": {eligibility.detail}" if eligibility.detail else "",
            )
            self._last_ineligible = state
        if self._audit is not None:
            self._audit.record(
                "ineligible",
                sweep=sweep_id,
                reason=eligibility.reason.value,
                detail=eligibility.detail,
            )

    def _scope(self, scope: str, sweep_id: str, stop: threading.Event) -> ScopeAudit:
        audit = ScopeAudit(scope)
        certification = self._coverage.certify(scope)
        if not certification.certified:
            audit.reason = certification.status.value
            if certification.detail:
                audit.reason += f": {certification.detail}"
            return audit
        audit.certified = True
        assert self._admin is not None
        for page in self._admin.versions(scope):
            if stopped := self._page(scope, sweep_id, page, audit):
                audit.stopped = stopped
                break
            if stop.wait(self._page_pause):
                audit.stopped = "stopped"
                break
        return audit

    def _page(
        self, scope: str, sweep_id: str, page: list[StoredVersion], audit: ScopeAudit
    ) -> str:
        """Condemn the page's unbound objects; returns why the scope's sweep stops,
        or "" to go on."""
        now = self._clock()
        by_digest: dict[str, list[StoredVersion]] = defaultdict(list)
        for version in page:
            audit.versions += 1
            audit.bytes += version.size_bytes
            if version.digest is None:
                audit.foreign += 1
            else:
                by_digest[version.digest].append(version)
        if not by_digest:
            return ""
        digests = list(by_digest)
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            pipe.smismember(reclaim_protected_key(scope), digests)
            pipe.hmget(reclaim_pins_key(scope), digests)
            pipe.smismember(reclaim_unexpanded_key(scope), digests)
            protected, pinned, unexpanded = pipe.execute()
        candidates: list[tuple[str, StoredVersion]] = []
        for digest, bound, pin, tree in zip(
            digests, protected, pinned, unexpanded, strict=True
        ):
            versions = by_digest[digest]
            if bound or pin is not None or tree:
                audit.kept += len(versions)
            elif any(now - v.last_modified < self._min_age for v in versions):
                audit.young += len(versions)
                audit.young_bytes += sum(v.size_bytes for v in versions)
            else:
                newest = max(versions, key=lambda v: (v.latest, v.last_modified))
                candidates.append((digest, newest))
        if not candidates:
            return ""
        keys = [
            RECLAIM_HOOKS_ON_KEY,
            reclaim_cert_key(scope),
            reclaim_protected_key(scope),
            reclaim_pins_key(scope),
            reclaim_unexpanded_key(scope),
            reclaim_uncertified_key(scope),
            reclaim_deleting_key(scope),
            reclaim_condemned_key(scope),
        ]
        args = [sweep_id, f"{now:.3f}", str(len(candidates))]
        for digest, version in candidates:
            args.extend((digest, version.version_id))
        reply = list(self._rds.sync.eval(_CONDEMN, len(keys), *keys, *args))
        if reply[0] != "ok":
            return str(reply[0])
        for (digest, _), verdict in zip(candidates, reply[1:], strict=True):
            versions = by_digest[digest]
            if verdict == "C":
                audit.orphans += len(versions)
                audit.orphan_bytes += sum(v.size_bytes for v in versions)
            else:
                audit.kept += len(versions)
        return ""

    def _audit_uploads(self, result: SweepAudit, sweep_id: str) -> None:
        result.uploads = self._uploads()
        if self._audit is not None:
            self._audit.record("uploads", sweep=sweep_id, **asdict(result.uploads))

    def _uploads(self) -> UploadAudit:
        assert self._admin is not None
        audit = UploadAudit()
        by_scope: dict[str, int] = defaultdict(int)
        listed_at = self._clock()
        open_ids: set[str] = set()
        for page in self._admin.uploads():
            for upload in page:
                open_ids.add(upload.upload_id)
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
        if self._multipart is not None:
            for scope in sorted(self._rds.sync.set_members(RECLAIM_SCOPES_KEY)):
                self._multipart.prune(scope, open_ids, listed_at)
        return audit

    def _publish(self, result: SweepAudit) -> None:
        self._orphans.set(result.orphans)
        self._orphan_bytes.set(result.orphan_bytes)
        self._young.set(result.young)
        self._young_bytes.set(result.young_bytes)
        self._uncertified.set(result.uncertified)
        uploads = result.uploads or UploadAudit()
        self._open_uploads.set(uploads.uploads)
        self._open_upload_bytes.set(uploads.bytes)
        self._logger.info(
            "Content reclamation sweep %s: %d scopes (%d uncertified), %d orphaned "
            "versions (%d bytes) condemned, %d unbound versions (%d bytes) too recent "
            "to condemn, %d open uploads (%d bytes); nothing deleted, in %.1fs",
            result.sweep,
            len(result.scopes),
            result.uncertified,
            result.orphans,
            result.orphan_bytes,
            result.young,
            result.young_bytes,
            uploads.uploads,
            uploads.bytes,
            result.seconds,
        )
        if self._audit is not None:
            self._audit.record(
                "sweep",
                sweep=result.sweep,
                scopes=len(result.scopes),
                uncertified=result.uncertified,
                orphans=result.orphans,
                orphan_bytes=result.orphan_bytes,
                young=result.young,
                young_bytes=result.young_bytes,
                uploads=uploads.uploads,
                upload_bytes=uploads.bytes,
                seconds=round(result.seconds, 3),
            )


__all__ = [
    "MIN_AGE_SEC",
    "SWEEP_INTERVAL_SEC",
    "AuditCollector",
    "ScopeAudit",
    "SweepAudit",
    "UploadAudit",
]
