"""Replica-selection strategy seam, bound per service family.

Selection is swappable behind the admission layer and configured per approved family,
not per workflow or request. The choice changes only which feasible replica a claim
reserves; the two-level boundary is fixed, and the engine still owns continuous
batching, token scheduling, and KV allocation. A workflow-derived family may trigger
materialization but never selects a strategy or its parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .capacity import residual_after
from .state import ClaimCredit, ReplicaCapacityReport


@dataclass(frozen=True)
class ReplicaCandidate:
    """A feasible replica with the outstanding credit already accounted against it."""

    replica_id: str
    report: ReplicaCapacityReport
    held_slots: int


class SelectionStrategy(ABC):
    """Chooses one replica among the feasible candidates for a reserving claim."""

    name: str

    @abstractmethod
    def select(
        self, candidates: list[ReplicaCandidate], credit: ClaimCredit
    ) -> ReplicaCandidate | None: ...


class BatchAwareBestFit(SelectionStrategy):
    """Fills an efficient batch before spilling: least remaining safe headroom wins.

    Spreading compatible requests thinly suppresses continuous-batching efficiency, so
    this picks the feasible replica left with the least normalized safe headroom under
    the candidate's projected reservation. Replica id breaks ties deterministically.
    """

    name = "batch-aware-best-fit"

    def select(
        self, candidates: list[ReplicaCandidate], credit: ClaimCredit
    ) -> ReplicaCandidate | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (
                residual_after(c.report, c.held_slots, credit),
                c.replica_id,
            ),
        )


class LeastLoad(SelectionStrategy):
    """Spreads load: the replica with the most remaining safe headroom wins."""

    name = "least-load"

    def select(
        self, candidates: list[ReplicaCandidate], credit: ClaimCredit
    ) -> ReplicaCandidate | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (
                -residual_after(c.report, c.held_slots, credit),
                c.replica_id,
            ),
        )


class RoundRobin(SelectionStrategy):
    """Rotates through feasible replicas by stable order."""

    name = "round-robin"

    def __init__(self) -> None:
        self._cursor: str | None = None

    def select(
        self, candidates: list[ReplicaCandidate], credit: ClaimCredit
    ) -> ReplicaCandidate | None:
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda c: c.replica_id)
        nxt = next(
            (c for c in ordered if self._cursor is None or c.replica_id > self._cursor),
            ordered[0],
        )
        self._cursor = nxt.replica_id
        return nxt


_STRATEGIES: dict[str, type[SelectionStrategy]] = {
    BatchAwareBestFit.name: BatchAwareBestFit,
    LeastLoad.name: LeastLoad,
    RoundRobin.name: RoundRobin,
}

DEFAULT_SELECTION_STRATEGY = BatchAwareBestFit.name


def build_selection_strategy(name: str | None) -> SelectionStrategy:
    """Instantiate a named strategy, falling back to the conservative default."""
    return _STRATEGIES.get(name or DEFAULT_SELECTION_STRATEGY, BatchAwareBestFit)()
