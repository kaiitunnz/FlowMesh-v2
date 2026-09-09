"""Per-task public port exposure for the root-hosted gated forward serve ingress.

``forward`` addressing is a separate exposure contract, not a route or capacity
contract. The root exposes one public authority with an allowed port range; every live
serve binding pinned to ``forward`` owns one ``ForwardPortExposure`` mapping a public
port on that authority to the binding.

The exposure is pure address-state: it mints no claim, reserves no capacity, and is
never a replica endpoint. Its identity is ``(serve_task_id, binding_generation)`` and
its locator is ``http://<authority>:<public_port>/`` — the deployment's own front proxy
terminates TLS and forwards plain HTTP to the root. A request arriving on that port
resolves its serve task from the live exposure, never from a client-supplied path: the
port is the whole address. Publication is two-phase — the root reserves a port, binds a
local listener on it, and only then does the exposure go ``LIVE`` and its url reach the
task. A drained
exposure rejects new requests, retires, and quarantines its port before a later exposure
reuses the number.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..utils.time import now_iso


class ForwardExposureStatus(StrEnum):
    """The lifecycle of one binding's public port exposure."""

    RESERVED = "reserved"
    BINDING = "binding"
    LIVE = "live"
    DRAINING = "draining"
    RETIRED = "retired"


class ForwardPortExposure(BaseModel):
    """How one live serve binding is reached over a public root forward port.

    Keyed by ``(serve_task_id, binding_generation)``; ``exposure_generation`` fences one
    reserve-bind-commit round so a stale bind or commit for a superseded reservation is
    refused. ``listener_generation`` is the root's ready evidence, matched at commit: a
    reserved port is not enough to go ``LIVE`` until its listener is bound.
    """

    model_config = ConfigDict(frozen=True)

    serve_task_id: str
    binding_generation: int
    authority: str
    public_port: int
    exposure_generation: int
    listener_generation: int = 0
    status: ForwardExposureStatus = ForwardExposureStatus.RESERVED
    created_at: str = Field(default_factory=now_iso)

    @property
    def public_url(self) -> str:
        """The base a client reaches this exposure at, host authority and port only."""
        return f"http://{self.authority}:{self.public_port}"

    @property
    def live(self) -> bool:
        return self.status is ForwardExposureStatus.LIVE


class ForwardExposureSnapshot(BaseModel):
    """The persisted forward exposures, one per live forward binding."""

    exposures: list[ForwardPortExposure] = Field(default_factory=list)


class ForwardIngressDirectory:
    """The root's public forward authority, port range, and live per-task exposures.

    Each forward binding owns a port on the root's authority, and control resolves the
    serve task from that authoritative exposure rather than from a task-qualified path.
    A retired exposure's port is quarantined until a later exposure reserves it afresh,
    so a reused number never carries a stale generation's traffic.
    """

    def __init__(self, authority: str, port_low: int, port_high: int) -> None:
        self._authority = authority
        self._port_low = port_low
        self._port_high = port_high
        self._exposures: dict[str, ForwardPortExposure] = {}
        self._quarantined: set[int] = set()
        self._generation_high: dict[str, int] = {}

    @property
    def configured(self) -> bool:
        """Whether the root has a usable forward authority and port range."""
        return bool(self._authority) and 1 <= self._port_low <= self._port_high <= 65535

    def reserve(
        self,
        *,
        serve_task_id: str,
        binding_generation: int,
        requested_port: int | None,
    ) -> ForwardPortExposure | None:
        """Reserve a port for one binding on the root authority, or None when it cannot.

        Fails closed when the root has no configured authority/range, a requested port
        is outside the range or already in use, or the range has no free port left.
        """
        if not self.configured:
            return None
        port = self._allocate(requested_port)
        if port is None:
            return None
        # Generations advance monotonically per task and never reset, even after a
        # retire drops the exposure, so a late bind or commit for a superseded
        # reservation can never match a reused generation number.
        generation = self._generation_high.get(serve_task_id, -1) + 1
        self._generation_high[serve_task_id] = generation
        exposure = ForwardPortExposure(
            serve_task_id=serve_task_id,
            binding_generation=binding_generation,
            authority=self._authority,
            public_port=port,
            exposure_generation=generation,
            status=ForwardExposureStatus.RESERVED,
        )
        self._exposures[serve_task_id] = exposure
        return exposure

    def mark_binding(self, serve_task_id: str, exposure_generation: int) -> None:
        """Move a reserved exposure to BINDING while the root binds its listener."""
        exposure = self._match(serve_task_id, exposure_generation)
        if exposure is not None and exposure.status is ForwardExposureStatus.RESERVED:
            self._exposures[serve_task_id] = exposure.model_copy(
                update={"status": ForwardExposureStatus.BINDING}
            )

    def mark_rebinding(self, serve_task_id: str) -> None:
        """Demote a persisted exposure to BINDING on restart until its listener rebinds.

        A loaded exposure comes back LIVE but holds no bound listener yet; treating it
        as BINDING keeps it out of ``live`` resolution until a fresh bind recommits it.
        """
        exposure = self._exposures.get(serve_task_id)
        if (
            exposure is not None
            and exposure.status is not ForwardExposureStatus.RETIRED
        ):
            self._exposures[serve_task_id] = exposure.model_copy(
                update={"status": ForwardExposureStatus.BINDING}
            )

    def commit(
        self,
        *,
        serve_task_id: str,
        exposure_generation: int,
        listener_generation: int,
    ) -> ForwardPortExposure | None:
        """Commit a bound exposure LIVE from the root's ready evidence, or None.

        A commit for a superseded reservation (a newer ``exposure_generation`` has since
        replaced it) is refused, so a late bind never revives a stale exposure.
        """
        exposure = self._match(serve_task_id, exposure_generation)
        if exposure is None or exposure.status is ForwardExposureStatus.RETIRED:
            return None
        live = exposure.model_copy(
            update={
                "status": ForwardExposureStatus.LIVE,
                "listener_generation": listener_generation,
            }
        )
        self._exposures[serve_task_id] = live
        return live

    def drain(self, serve_task_id: str) -> ForwardPortExposure | None:
        """Mark an exposure draining so it rejects new requests; return the drained."""
        exposure = self._exposures.get(serve_task_id)
        if exposure is None or exposure.status is ForwardExposureStatus.RETIRED:
            return None
        drained = exposure.model_copy(update={"status": ForwardExposureStatus.DRAINING})
        self._exposures[serve_task_id] = drained
        return drained

    def retire(self, serve_task_id: str) -> ForwardPortExposure | None:
        """Retire an exposure and quarantine its port before any later reuse."""
        exposure = self._exposures.pop(serve_task_id, None)
        if exposure is None:
            return None
        self._quarantined.add(exposure.public_port)
        return exposure.model_copy(update={"status": ForwardExposureStatus.RETIRED})

    def retire_all(self) -> None:
        """Retire every exposure, as when forward is disabled at startup.

        A persisted exposure holds no bound listener until forward binds one; when
        forward is off no listener ever will, so drop them all out of live resolution
        rather than resolve a task to a port nothing serves.
        """
        for serve_task_id in list(self._exposures):
            self.retire(serve_task_id)

    def live(self, serve_task_id: str) -> ForwardPortExposure | None:
        """The LIVE exposure for a serve task, or None when none is live."""
        exposure = self._exposures.get(serve_task_id)
        return exposure if exposure is not None and exposure.live else None

    def current(self, serve_task_id: str) -> ForwardPortExposure | None:
        """The exposure in any state for a serve task, or None."""
        return self._exposures.get(serve_task_id)

    def all(self) -> list[ForwardPortExposure]:
        return list(self._exposures.values())

    def to_snapshot(self) -> ForwardExposureSnapshot:
        return ForwardExposureSnapshot(exposures=list(self._exposures.values()))

    def load_snapshot(self, snapshot: ForwardExposureSnapshot) -> None:
        self._exposures = {e.serve_task_id: e for e in snapshot.exposures}
        self._generation_high = {
            e.serve_task_id: e.exposure_generation for e in snapshot.exposures
        }

    def _match(
        self, serve_task_id: str, exposure_generation: int
    ) -> ForwardPortExposure | None:
        exposure = self._exposures.get(serve_task_id)
        if exposure is None or exposure.exposure_generation != exposure_generation:
            return None
        return exposure

    def _allocate(self, requested: int | None) -> int | None:
        in_use = {
            e.public_port
            for e in self._exposures.values()
            if e.status is not ForwardExposureStatus.RETIRED
        }
        if requested is not None:
            if not self._port_low <= requested <= self._port_high:
                return None
            if requested in in_use or requested in self._quarantined:
                return None
            return requested
        for port in range(self._port_low, self._port_high + 1):
            if port not in in_use and port not in self._quarantined:
                return port
        return None
