"""The read projection of sealed activation-private state and its policy decisions.

The inventory reads the sealed generations the live orchestration engines record and
asks the state-control policy what should happen to the eligible ones. Reading decides
nothing physical: no state is materialized, copied, or deleted, no attachment or claim
is minted, and no binding changes. A generation a holder is writing is reported as
evidence and left to the ledger.

The eligibility screen a decision passes is what bounds acting on it: eviction covers
only a generation no holder writes and no continuation can resume on, and a copy verb
only components every one of which is exportable.
"""

from dataclasses import dataclass

from ..policy import (
    PolicySurface,
    SealedGenerationEvidence,
    StateControlDecision,
    screen,
)
from ..task.runtime import TaskRuntime


@dataclass(frozen=True)
class SealedGenerationEntry:
    """One sealed generation and the decision a policy reached for it."""

    evidence: SealedGenerationEvidence
    decision: StateControlDecision | None


def sealed_state_inventory(
    runtime: TaskRuntime, surface: PolicySurface | None
) -> list[SealedGenerationEntry]:
    """Sealed generations across the live engines, newest seal first.

    A policy decides over the generations no holder is currently writing, and every
    answer is screened against the generation's own evidence before it is reported.
    """
    generations = sorted(
        runtime.sealed_private_state(),
        key=lambda item: (item.sealed_at or "", item.reference_id),
        reverse=True,
    )
    if surface is None:
        return [SealedGenerationEntry(evidence, None) for evidence in generations]
    eligible = [evidence for evidence in generations if not evidence.attached]
    decisions = surface.state_control.decide(eligible)
    return [
        SealedGenerationEntry(
            evidence,
            (
                screen(evidence, decision)
                if (decision := decisions.get(evidence.reference_id)) is not None
                else None
            ),
        )
        for evidence in generations
    ]
