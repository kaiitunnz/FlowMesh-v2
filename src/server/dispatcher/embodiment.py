"""Choosing which embodiment of a menu node the fabric runs.

The selector reads live feasibility and returns one of the embodiments the compiler
already proved equivalent, or defers. It holds no capacity authority: it names no worker
or replica, reserves nothing, mints no claim or route, and cannot turn a required or
self-contained binding into an optional one. A resident selection admits through the
ordinary claim path afterwards, and a local one is placed like any other local work.
"""

from dataclasses import dataclass
from typing import Protocol

from shared.tasks import TaskEnvelope
from shared.tasks.specs import InferenceEmbodimentKind

from ..task.v2.representations.plan import (
    InferenceEmbodimentCandidate,
    InferenceEmbodimentMenu,
)


def relay_placement_task(task: TaskEnvelope) -> TaskEnvelope:
    """The task as a worker that only carries its invocation must satisfy it.

    A resident-served embodiment runs its model on a replica, so the accelerator its
    leaf declares for the self-contained embodiment is not a requirement on the worker
    that relays. Every other declared resource still applies.
    """
    hardware = task.spec.resources.hardware if task.spec.resources else None
    if hardware is None or hardware.gpu is None:
        return task
    relayed = task.model_copy(deep=True)
    resources = relayed.spec.resources
    if resources is not None and resources.hardware is not None:
        resources.hardware.gpu = None
    return relayed


@dataclass(frozen=True)
class EmbodimentSnapshot:
    """Live feasibility evidence for one ready menu node, read-only.

    ``local_capable_workers`` counts the workers that satisfy the task as declared,
    which the embodiment that loads the model needs. ``relay_capable_workers`` counts
    those that satisfy it without its local accelerator, which is what a worker carrying
    an invocation to a replica needs. ``resident_capacity_enabled`` is whether the
    deployment serves resident capacity at all. All three are evidence about
    feasibility, never a reservation of it.
    """

    local_capable_workers: int
    relay_capable_workers: int
    resident_capacity_enabled: bool

    def evidence(self) -> str:
        return (
            f"local_capable_workers={self.local_capable_workers} "
            f"relay_capable_workers={self.relay_capable_workers} "
            f"resident_capacity_enabled={self.resident_capacity_enabled}"
        )


@dataclass(frozen=True)
class EmbodimentDecision:
    """Either the embodiment to run or the reason none can be placed now."""

    alternative_id: str | None = None
    defer_reason: str | None = None

    @classmethod
    def select(cls, alternative_id: str) -> "EmbodimentDecision":
        return cls(alternative_id=alternative_id)

    @classmethod
    def defer(cls, reason: str) -> "EmbodimentDecision":
        return cls(defer_reason=reason)


class EmbodimentSelector(Protocol):
    """Resolves one menu node against live feasibility."""

    name: str

    def __call__(
        self, menu: InferenceEmbodimentMenu, snapshot: EmbodimentSnapshot
    ) -> EmbodimentDecision: ...


def candidate_feasible(
    candidate: InferenceEmbodimentCandidate, snapshot: EmbodimentSnapshot
) -> bool:
    """Whether a candidate's own envelope can be satisfied right now."""
    if candidate.kind is InferenceEmbodimentKind.RESIDENT_SERVED:
        return snapshot.resident_capacity_enabled and snapshot.relay_capable_workers > 0
    return snapshot.local_capable_workers > 0


class PrimaryEmbodimentSelector:
    """Runs the embodiment the submission declared primary, or defers.

    Deferring rather than switching keeps the declared behavior of a workflow stable:
    an embodiment the author did not name is reached only through a selector a
    deployment installs deliberately.
    """

    name = "primary"

    def __call__(
        self, menu: InferenceEmbodimentMenu, snapshot: EmbodimentSnapshot
    ) -> EmbodimentDecision:
        primary = menu.candidate(menu.primary)
        if primary is None:
            return EmbodimentDecision.defer("primary_embodiment_missing")
        if not candidate_feasible(primary, snapshot):
            return EmbodimentDecision.defer(f"{primary.kind.value}_infeasible")
        return EmbodimentDecision.select(primary.alternative_id)
