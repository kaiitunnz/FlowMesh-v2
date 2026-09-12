"""Advisory control over sealed generations of activation-private state.

A state-control policy reads the sealed generations the orchestration ledger records
and says what should happen to each one. It decides; it never acts. Nothing here
materializes, copies, or deletes state bytes, and no decision creates a holder, an
attachment, or a capacity claim: a sealed generation's identity, its owner fence, and
who may write it stay the ledger's.

Every verb beyond retention is screened against the generation's own evidence before a
policy's answer is honored, so a decision that a component's exportability or a live
continuation forbids settles as retention.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from shared.private_state import BundleProfile, SealedComponent


class StateControlVerb(StrEnum):
    """What a policy asks for a sealed generation."""

    RETAIN = "retain"
    EVICT = "evict"
    REPLICATE = "replicate"
    PLACE = "place"
    PREFETCH = "prefetch"


@dataclass(frozen=True)
class SealedGenerationEvidence:
    """One sealed generation as a policy and an operator see it.

    Carries the generation's identity, the holder fence that can supply it, its
    component contents by digest, and the live facts that bound what may legally happen
    to it: ``attached`` while a holder writes the lineage, ``resumable`` while a
    continuation could still resume on the generation, and ``exportable`` when every
    component may be materialized away from its holder.
    """

    reference_id: str
    instance_id: str
    activation_id: str
    owner_id: str
    org_id: str
    tenant: str | None
    profile: BundleProfile
    generation: int
    owner_worker_id: str
    owner_incarnation: int
    components: tuple[SealedComponent, ...]
    attached: bool
    resumable: bool
    exportable: bool
    sealed_at: str | None = None


@dataclass(frozen=True)
class StateControlDecision:
    """What a policy decided for one sealed generation, and why."""

    verb: StateControlVerb
    reason: str


def eligible_verbs(evidence: SealedGenerationEvidence) -> frozenset[StateControlVerb]:
    """The verbs a sealed generation's own evidence admits.

    Retention is always legal. Eviction takes a generation no holder is writing and no
    continuation can resume on. Replication, placement, and prefetch move a generation
    away from the holder that sealed it, which takes components every one of which is
    exportable.
    """
    verbs = {StateControlVerb.RETAIN}
    if not evidence.attached and not evidence.resumable:
        verbs.add(StateControlVerb.EVICT)
    if evidence.exportable:
        verbs.update(
            (
                StateControlVerb.REPLICATE,
                StateControlVerb.PLACE,
                StateControlVerb.PREFETCH,
            )
        )
    return frozenset(verbs)


def screen(
    evidence: SealedGenerationEvidence, decision: StateControlDecision
) -> StateControlDecision:
    """The decision when the evidence admits its verb, retention otherwise."""
    if decision.verb in eligible_verbs(evidence):
        return decision
    return StateControlDecision(
        verb=StateControlVerb.RETAIN,
        reason=f"{decision.verb.value} is not admissible for this generation",
    )


class StateControlPolicy:
    """The state-control hook, retaining every sealed generation."""

    name = "none"

    def decide(
        self, generations: Sequence[SealedGenerationEvidence]
    ) -> Mapping[str, StateControlDecision]:
        """A decision per generation, keyed by state reference."""
        return {
            evidence.reference_id: StateControlDecision(
                verb=StateControlVerb.RETAIN,
                reason="no state-control policy configured",
            )
            for evidence in generations
        }


class RecencyWarmth(StateControlPolicy):
    """Keeps a fixed number of the most recently sealed generations warm.

    Generations are ranked by seal time, newest first, and the warmest ``warm`` of them
    are retained. The rest are asked to be evicted, which holds for the ones whose
    evidence admits eviction and settles as retention for the ones it does not.
    """

    name = "recency_warmth"

    def __init__(self, warm: int) -> None:
        self._warm = max(0, warm)

    def decide(
        self, generations: Sequence[SealedGenerationEvidence]
    ) -> Mapping[str, StateControlDecision]:
        ranked = sorted(
            generations,
            key=lambda item: (item.sealed_at or "", item.reference_id),
            reverse=True,
        )
        decisions: dict[str, StateControlDecision] = {}
        for position, evidence in enumerate(ranked):
            if position < self._warm:
                decisions[evidence.reference_id] = StateControlDecision(
                    verb=StateControlVerb.RETAIN,
                    reason=f"within the warmest {self._warm} sealed generations",
                )
            else:
                decisions[evidence.reference_id] = StateControlDecision(
                    verb=StateControlVerb.EVICT,
                    reason=f"colder than the warmest {self._warm} sealed generations",
                )
        return decisions
