"""Per-task public port exposure for the gated forward serve ingress.

``forward`` addressing is a separate exposure contract, not a route or capacity
contract. A deployment registers one or more forward ingress hosts — each a public
authority with an allowed port range, an optional TLS profile, and the worker that hosts
its listeners — and every live serve binding pinned to ``forward`` owns one
``ForwardPortExposure`` mapping a public port on such a host to that binding.

The exposure is pure address-state: it mints no claim, reserves no capacity, and
is never a replica endpoint. Its identity is ``(serve_task_id, binding_generation)``
and its locator is ``https://<authority>:<public_port>/`` (``http`` only on an
explicit plaintext local-test host). A request arriving on that port resolves its
serve task from the live exposure, never from a client-supplied path: the port is
the whole address. Publication is two-phase — control reserves a port, the ingress
worker binds a listener and returns ready evidence, and only then does the exposure
go ``LIVE`` and its url reach the task. A drained exposure rejects new requests,
retires, and quarantines its port before a later exposure reuses the number.
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


class ForwardIngressHost(BaseModel):
    """An operator-registered forward ingress host and the ports it may expose.

    ``authority`` is the public host authority a client dials; ``worker_id`` is the
    worker that binds its per-task listeners and ``origin_id`` the node the network
    plane derives the transport origin from. ``port_low``/``port_high`` bound the ports
    it may allocate. ``tls_profile_generation`` is ``0`` for an explicit plaintext
    local-test host and positive for one given a TLS profile; a deployment that requires
    TLS refuses to expose a task on a plaintext host.
    """

    model_config = ConfigDict(frozen=True)

    authority: str
    worker_id: str
    origin_id: str
    port_low: int
    port_high: int
    tls_profile_generation: int = 0
    generation: int = 0

    @property
    def tls(self) -> bool:
        return self.tls_profile_generation > 0


class ForwardPortExposure(BaseModel):
    """How one live serve binding is reached over a public forward port.

    Keyed by ``(serve_task_id, binding_generation)``; ``exposure_generation`` fences one
    reserve-bind-commit round so a stale bind or commit for a superseded reservation is
    refused. ``listener_generation`` and ``attachment_generation`` are the worker's
    ready evidence, matched at commit: a bound socket is not enough to go ``LIVE``.
    """

    model_config = ConfigDict(frozen=True)

    serve_task_id: str
    binding_generation: int
    authority: str
    public_port: int
    worker_id: str
    origin_id: str
    exposure_generation: int
    tls_profile_generation: int = 0
    listener_generation: int = 0
    attachment_generation: int = 0
    status: ForwardExposureStatus = ForwardExposureStatus.RESERVED
    created_at: str = Field(default_factory=now_iso)

    @property
    def tls(self) -> bool:
        return self.tls_profile_generation > 0

    @property
    def public_url(self) -> str:
        """The base a client reaches this exposure at, host authority and port only."""
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.authority}:{self.public_port}"

    @property
    def live(self) -> bool:
        return self.status is ForwardExposureStatus.LIVE


class ForwardExposureSnapshot(BaseModel):
    """The persisted forward exposures, one per live forward binding."""

    exposures: list[ForwardPortExposure] = Field(default_factory=list)


class ForwardIngressDirectory:
    """Operator-registered forward ingress hosts and the live per-task port exposures.

    It replaces the single shared forward ingress: rather than one listener addressed by
    a task-qualified path, each forward binding owns a port on a registered host and
    control resolves the serve task from that authoritative exposure. A retired
    exposure's port is quarantined until a later exposure reserves it afresh, so a
    reused number never carries a stale generation's traffic.
    """

    def __init__(self) -> None:
        self._hosts: dict[str, ForwardIngressHost] = {}
        self._exposures: dict[str, ForwardPortExposure] = {}
        self._quarantined: dict[str, set[int]] = {}

    def register_host(self, host: ForwardIngressHost) -> bool:
        """Register (or re-register at a newer generation) one forward ingress host."""
        if (
            host.port_low < 1
            or host.port_high > 65535
            or host.port_low > host.port_high
        ):
            return False
        current = self._hosts.get(host.worker_id)
        if current is not None and host.generation < current.generation:
            return False
        self._hosts[host.worker_id] = host
        return True

    def withdraw_host(self, worker_id: str) -> None:
        """Drop a host and every exposure it carried, so its tasks fail closed again."""
        self._hosts.pop(worker_id, None)
        for task_id, exposure in list(self._exposures.items()):
            if exposure.worker_id == worker_id:
                self._exposures.pop(task_id, None)

    def host(self, worker_id: str) -> ForwardIngressHost | None:
        return self._hosts.get(worker_id)

    def any_host(self) -> ForwardIngressHost | None:
        """Some registered host, or None when a deployment has registered none."""
        return next(iter(self._hosts.values()), None)

    def reserve(
        self,
        *,
        serve_task_id: str,
        binding_generation: int,
        requested_port: int | None,
        require_tls: bool,
    ) -> ForwardPortExposure | None:
        """Reserve a port for one binding on a registered host, or None when it cannot.

        Fails closed when no host is registered, the deployment requires TLS but the
        host has no profile, a requested port is outside the range or already in use, or
        the range has no free port left.
        """
        host = self.any_host()
        if host is None:
            return None
        if require_tls and not host.tls:
            return None
        port = self._allocate(host, requested_port)
        if port is None:
            return None
        prior = self._exposures.get(serve_task_id)
        generation = (prior.exposure_generation + 1) if prior is not None else 0
        exposure = ForwardPortExposure(
            serve_task_id=serve_task_id,
            binding_generation=binding_generation,
            authority=host.authority,
            public_port=port,
            worker_id=host.worker_id,
            origin_id=host.origin_id,
            exposure_generation=generation,
            tls_profile_generation=host.tls_profile_generation,
            status=ForwardExposureStatus.RESERVED,
        )
        self._exposures[serve_task_id] = exposure
        return exposure

    def mark_binding(self, serve_task_id: str, exposure_generation: int) -> None:
        """Move a reserved exposure to BINDING while its worker binds the listener."""
        exposure = self._match(serve_task_id, exposure_generation)
        if exposure is not None and exposure.status is ForwardExposureStatus.RESERVED:
            self._exposures[serve_task_id] = exposure.model_copy(
                update={"status": ForwardExposureStatus.BINDING}
            )

    def commit(
        self,
        *,
        serve_task_id: str,
        exposure_generation: int,
        listener_generation: int,
        attachment_generation: int,
    ) -> ForwardPortExposure | None:
        """Commit a bound exposure LIVE from the worker's ready evidence, or None.

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
                "attachment_generation": attachment_generation,
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
        self._quarantined.setdefault(exposure.worker_id, set()).add(
            exposure.public_port
        )
        return exposure.model_copy(update={"status": ForwardExposureStatus.RETIRED})

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

    def _match(
        self, serve_task_id: str, exposure_generation: int
    ) -> ForwardPortExposure | None:
        exposure = self._exposures.get(serve_task_id)
        if exposure is None or exposure.exposure_generation != exposure_generation:
            return None
        return exposure

    def _allocate(self, host: ForwardIngressHost, requested: int | None) -> int | None:
        in_use = {
            e.public_port
            for e in self._exposures.values()
            if e.worker_id == host.worker_id
            and e.status is not ForwardExposureStatus.RETIRED
        }
        quarantined = self._quarantined.get(host.worker_id, set())
        if requested is not None:
            if not host.port_low <= requested <= host.port_high:
                return None
            if requested in in_use or requested in quarantined:
                return None
            return requested
        for port in range(host.port_low, host.port_high + 1):
            if port not in in_use and port not in quarantined:
                return port
        return None
