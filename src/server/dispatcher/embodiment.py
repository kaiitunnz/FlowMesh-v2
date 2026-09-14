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
    deployment serves resident capacity at all, and ``resident_admission_slots`` how
    many concurrent sequences one replica admits, which bounds the batch a resident
    embodiment can ever carry. All are evidence about feasibility, never a reservation
    of it.
    """

    local_capable_workers: int
    relay_capable_workers: int
    resident_capacity_enabled: bool
    resident_admission_slots: int = 0

    def admits_batch(self, batch_size: int) -> bool:
        """Whether a replica's admission bound can ever hold a batch this size.

        A snapshot reporting no bound does not constrain one: admission enforces its own
        capacity, and the scheduler rules a candidate out only on evidence it holds.
        """
        return (
            self.resident_admission_slots <= 0
            or batch_size <= self.resident_admission_slots
        )

    def evidence(self) -> str:
        return (
            f"local_capable_workers={self.local_capable_workers} "
            f"relay_capable_workers={self.relay_capable_workers} "
            f"resident_capacity_enabled={self.resident_capacity_enabled} "
            f"resident_admission_slots={self.resident_admission_slots}"
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
    candidate: InferenceEmbodimentCandidate,
    snapshot: EmbodimentSnapshot,
    batch_size: int = 1,
) -> bool:
    """Whether a candidate's own envelope can be satisfied right now."""
    if candidate.kind is InferenceEmbodimentKind.RESIDENT_SERVED:
        return (
            snapshot.resident_capacity_enabled
            and snapshot.relay_capable_workers > 0
            and snapshot.admits_batch(batch_size)
        )
    return snapshot.local_capable_workers > 0


def candidate_unavailable(
    candidate: InferenceEmbodimentCandidate,
    snapshot: EmbodimentSnapshot,
    batch_size: int = 1,
) -> bool:
    """Whether the deployment's own configuration rules a candidate out entirely.

    This is narrower than infeasibility: a fleet whose workers are momentarily busy
    still admits the candidate once one frees up, but a deployment that serves no
    resident capacity never admits a resident-served one, however long the task waits.
    A batch carrying more conversations than a replica admits at once is the same case:
    it occupies one sequence per conversation, so no amount of waiting frees enough.
    """
    if candidate.kind is not InferenceEmbodimentKind.RESIDENT_SERVED:
        return False
    return not snapshot.resident_capacity_enabled or not snapshot.admits_batch(
        batch_size
    )


class PrimaryEmbodimentSelector:
    """Runs the primary embodiment, falling through only when it is ruled out.

    A primary the fleet cannot place right now defers, so a momentary shortage never
    silently changes how a workflow runs. A primary the deployment rules out entirely
    is a different case: waiting for it would fail the task on a deployment that was
    never going to serve it, so the one remaining feasible embodiment runs instead.
    """

    name = "primary"

    def __call__(
        self, menu: InferenceEmbodimentMenu, snapshot: EmbodimentSnapshot
    ) -> EmbodimentDecision:
        primary = menu.candidate(menu.primary)
        if primary is None:
            return EmbodimentDecision.defer("primary_embodiment_missing")
        if candidate_feasible(primary, snapshot, menu.batch_size):
            return EmbodimentDecision.select(primary.alternative_id)
        if not candidate_unavailable(primary, snapshot, menu.batch_size):
            return EmbodimentDecision.defer(f"{primary.kind.value}_infeasible")
        fallthrough = [
            candidate
            for candidate in menu.candidates
            if candidate.alternative_id != primary.alternative_id
            and candidate_feasible(candidate, snapshot, menu.batch_size)
        ]
        if len(fallthrough) != 1:
            return EmbodimentDecision.defer(
                _unavailable_reason(primary, snapshot, menu.batch_size)
            )
        return EmbodimentDecision.select(fallthrough[0].alternative_id)


def _unavailable_reason(
    primary: InferenceEmbodimentCandidate,
    snapshot: EmbodimentSnapshot,
    batch_size: int,
) -> str:
    """Why no embodiment of a node can run, in terms an operator can act on."""
    base = f"{primary.kind.value}_unavailable"
    if snapshot.admits_batch(batch_size):
        return base
    return (
        f"{base}: its {batch_size} conversations need one admission slot each and a "
        f"replica admits {snapshot.resident_admission_slots} "
        "(RESIDENT_ADMISSION_SLOTS), and no worker can run it self-contained"
    )
