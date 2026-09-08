"""The task-scoped standing serve allocation binding.

A public user-declared ``serve`` task is adopted at start as a standing resident
allocation. Its ``ServeTaskResidencyBinding`` is the durable, versioned control relation
from the user-visible task ID to one normalized service profile and its bounded standing
allocation group. It records no engine URL, listener address, credential, or public
alias: task-read access authorizes use of the handle, while each request still enters
normal admission against only this binding's own allocation group.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..resident.state import AdmissionProfile
from ..task.v2.representations.operators import ServiceDependency, ServiceInterface
from ..utils.time import now_iso
from .state import ServeTerminalSnapshot

# The allocation group of an adopted serve task is a family unique to that task, so
# admission for its requests selects only its own replica(s) rather than any compatible
# family replica. The public task id is the group identity.
_SERVE_FAMILY_PREFIX = "serve/"


def serve_family_key(serve_task_id: str) -> str:
    """The per-task service-family key whose replicas are this serve task's group."""
    return f"{_SERVE_FAMILY_PREFIX}{serve_task_id}"


class ServeBindingStatus(StrEnum):
    """A binding is live while its serve task is, then drains before stop."""

    LIVE = "live"
    DRAINING = "draining"


class ServeTaskResidencyBinding(BaseModel):
    """A versioned mapping from one serve task to its standing allocation profile.

    Keyed by ``(serve_task_id, binding_generation)``. It carries the normalized service
    reference, interface, isolation, and adapter that key the reuse domain, the fixed
    request/output bounds the caller cannot widen, and the allowed method/path the gate
    enforces. It never records an engine URL, listener, credential, or public alias.
    """

    model_config = ConfigDict(frozen=True)

    serve_task_id: str
    binding_generation: int
    service_ref: str
    interface: ServiceInterface = ServiceInterface.CHAT
    isolation: str | None = None
    adapter: str | None = None
    adapter_source: str | None = None
    engine_batch_key: str
    max_output_tokens: int | None = None
    allowed_methods: tuple[str, ...] = ("POST",)
    status: ServeBindingStatus = ServeBindingStatus.LIVE
    created_at: str = Field(default_factory=now_iso)

    @property
    def family(self) -> str:
        return serve_family_key(self.serve_task_id)

    @property
    def live(self) -> bool:
        return self.status is ServeBindingStatus.LIVE

    def dependency(self) -> ServiceDependency:
        """The normalized resident dependency a request against this binding
        resolves."""
        return ServiceDependency(
            service_ref=self.service_ref,
            interface=self.interface,
            adapter=self.adapter,
            adapter_source=self.adapter_source,
            isolation=self.isolation,
        )

    def profile(self, *, descriptor_digest: str | None = None) -> AdmissionProfile:
        """The admission profile one request against this binding is admitted under."""
        return AdmissionProfile(
            engine_batch_key=self.engine_batch_key,
            max_output_tokens=self.max_output_tokens,
            adapter_ref=self.adapter,
            adapter_source=self.adapter_source,
            serve_task_id=self.serve_task_id,
            binding_generation=self.binding_generation,
            descriptor_digest=descriptor_digest,
        )


class ServeBindingSnapshot(BaseModel):
    """The persisted serve-binding control facts."""

    bindings: list[ServeTaskResidencyBinding] = Field(default_factory=list)


class ServeSnapshot(BaseModel):
    """The persisted gated-serve control facts: the bindings and status terminals."""

    bindings: ServeBindingSnapshot = Field(default_factory=ServeBindingSnapshot)
    terminals: ServeTerminalSnapshot = Field(default_factory=ServeTerminalSnapshot)


class ServeBindingStore:
    """Durable custody of the live serve-task residency bindings by task ID.

    One live binding per serve task; a re-adoption supersedes the prior generation. A
    stopped task's binding drains (rejecting new calls) before it is removed, so an
    accepted request still reconciles.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, ServeTaskResidencyBinding] = {}

    def adopt(
        self,
        serve_task_id: str,
        *,
        service_ref: str,
        interface: ServiceInterface,
        isolation: str | None,
        adapter: str | None,
        adapter_source: str | None,
        engine_batch_key: str,
        max_output_tokens: int | None,
    ) -> ServeTaskResidencyBinding:
        """Register (or supersede) the live binding for a serve task, bumping its
        generation past any prior binding for the same task."""
        prior = self._bindings.get(serve_task_id)
        generation = (prior.binding_generation + 1) if prior is not None else 0
        binding = ServeTaskResidencyBinding(
            serve_task_id=serve_task_id,
            binding_generation=generation,
            service_ref=service_ref,
            interface=interface,
            isolation=isolation,
            adapter=adapter,
            adapter_source=adapter_source,
            engine_batch_key=engine_batch_key,
            max_output_tokens=max_output_tokens,
        )
        self._bindings[serve_task_id] = binding
        return binding

    def live(self, serve_task_id: str) -> ServeTaskResidencyBinding | None:
        """The live binding for a serve task, or None when none is live."""
        binding = self._bindings.get(serve_task_id)
        return binding if binding is not None and binding.live else None

    def get(self, serve_task_id: str) -> ServeTaskResidencyBinding | None:
        return self._bindings.get(serve_task_id)

    def drain(self, serve_task_id: str) -> ServeTaskResidencyBinding | None:
        """Mark a binding draining so it rejects new calls; return the drained one."""
        binding = self._bindings.get(serve_task_id)
        if binding is None or not binding.live:
            return None
        drained = binding.model_copy(update={"status": ServeBindingStatus.DRAINING})
        self._bindings[serve_task_id] = drained
        return drained

    def remove(self, serve_task_id: str) -> None:
        self._bindings.pop(serve_task_id, None)

    def all(self) -> list[ServeTaskResidencyBinding]:
        return list(self._bindings.values())

    def to_snapshot(self) -> ServeBindingSnapshot:
        return ServeBindingSnapshot(bindings=list(self._bindings.values()))

    def load_snapshot(self, snapshot: ServeBindingSnapshot) -> None:
        self._bindings = {b.serve_task_id: b for b in snapshot.bindings}
