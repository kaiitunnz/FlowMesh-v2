"""Choosing which embodiment of a menu node the fabric runs.

The selector reads live feasibility and returns one of the embodiments the compiler
already proved equivalent, or defers. It holds no capacity authority: it names no worker
or replica, reserves nothing, mints no claim or route, and cannot turn a required or
self-contained binding into an optional one. A resident selection admits through the
ordinary claim path afterwards, and a local one is placed like any other local work.
"""

from dataclasses import dataclass
from typing import Protocol

from shared.tasks.specs import InferenceEmbodimentKind

from ..task.v2.representations.plan import (
    InferenceEmbodimentCandidate,
    InferenceEmbodimentMenu,
)


@dataclass(frozen=True)
class EmbodimentSnapshot:
    """Live feasibility evidence for one ready menu node, read-only.

    ``eligible_workers`` counts the workers that satisfy the task as declared, which
    either embodiment needs: one runs the model, the other carries the invocation.
    ``resident_available`` is whether resident capacity can serve the candidate's family
    at all — evidence about feasibility, never a reservation of it.
    """

    eligible_workers: int
    resident_available: bool

    def evidence(self) -> str:
        return (
            f"eligible_workers={self.eligible_workers} "
            f"resident_available={self.resident_available}"
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
    if snapshot.eligible_workers <= 0:
        return False
    if candidate.kind is InferenceEmbodimentKind.RESIDENT_SERVED:
        return snapshot.resident_available
    return True


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
