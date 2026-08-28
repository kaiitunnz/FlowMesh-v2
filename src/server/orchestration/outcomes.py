"""Invocation/effect outcome semantics for the v2 orchestration path.

Four concerns live here as pure logic: which operators the path admits, whether a
settled work item may be recomputed or must be restored, the durable invocation state
machine that keeps an uncertain effect from silently retrying or reporting success, and
monotone attenuation of an authority face across a spawn site.
"""

from ..task.v2.representations.operators import (
    DeterminismClass,
    EffectClass,
    EffectReplayContract,
    InputProvenanceKind,
    LeafProfile,
    RecoveryClass,
)
from .state import InvocationState, RecoveryDisposition

_DETERMINISTIC = frozenset(
    {DeterminismClass.DETERMINISTIC_BITWISE, DeterminismClass.DETERMINISTIC_SEMANTIC}
)


class AdmissionError(ValueError):
    """Raised when a plan operator is not runnable on the v2 compatibility path."""


def is_replayable(
    effect: EffectClass, replay_contract: EffectReplayContract | None
) -> bool:
    """Whether an uncertain invocation may be reissued through its stable identity.

    A pure operation is idempotent under recompute; an external effect is replayable
    only when it declares a deduplication boundary. A compensable or ambiguity-terminal
    effect is not reissued — its uncertainty is handled without inferring success.
    """
    if effect is EffectClass.PURE:
        return True
    return (
        effect is EffectClass.EXTERNAL_EFFECT
        and replay_contract is EffectReplayContract.REPLAYABLE_DEDUP
    )


def is_compensable(
    effect: EffectClass, replay_contract: EffectReplayContract | None
) -> bool:
    """Whether an uncertain external effect resolves through a compensation contract."""
    return (
        effect is EffectClass.EXTERNAL_EFFECT
        and replay_contract is EffectReplayContract.COMPENSABLE
    )


def attenuate(
    parent_face: tuple[str, ...], *bounds: tuple[str, ...]
) -> tuple[str, ...]:
    """Intersect an authority face with each bound, monotonically shrinking it.

    A child face can only lose interfaces relative to its parent, so a spawn site can
    never widen what an ancestor forbids.
    """
    face = set(parent_face)
    for bound in bounds:
        face &= set(bound)
    return tuple(sorted(face))


def classify_recovery(profile: LeafProfile) -> RecoveryDisposition:
    """Whether a settled operation may be recomputed within one execution.

    Only a pure/hermetic deterministic operation over a complete pinned input cone
    may recompute; sampling, unpinned reads, and effects must restore their recorded
    outcome before later work relies on them.
    """
    if (
        profile.effect is EffectClass.PURE
        and profile.determinism in _DETERMINISTIC
        and profile.input_provenance is InputProvenanceKind.EXTERNAL_PINNED
        and profile.recovery is RecoveryClass.RECOMPUTE
    ):
        return RecoveryDisposition.RECOMPUTE
    return RecoveryDisposition.RESTORE


def check_admissible(
    operator_id: str,
    effect: EffectClass,
    replay_contract: EffectReplayContract | None,
    residency_only: bool,
) -> None:
    """Reject an operator the v2 orchestration path cannot run.

    The path runs effect-free operations and external effects under any declared replay
    contract (replayable-with-dedup, compensable, or ambiguity-terminal), handling each
    uncertainty without inferring success. Private-state operations (sandbox recovery)
    and residency administration are not admitted.
    """
    if residency_only:
        raise AdmissionError(
            f"operator {operator_id!r} administers resident capacity, which the "
            "orchestration path does not run"
        )
    if effect is EffectClass.PRIVATE_STATE:
        raise AdmissionError(
            f"operator {operator_id!r} declares private-state effect, whose sandbox "
            "recovery boundary the orchestration path does not run"
        )


def next_on_acknowledge(state: InvocationState) -> InvocationState:
    """Advance a pending invocation to the generic engine-enqueue acknowledgement."""
    if state in (InvocationState.ISSUED, InvocationState.UNCERTAIN):
        return InvocationState.ACKNOWLEDGED
    return state


def next_on_terminal(state: InvocationState) -> InvocationState:
    """Advance an invocation to its terminal receipt.

    An ambiguity-terminal or compensation-required outcome is never regressed to a
    success receipt by a late completion.
    """
    if state in (
        InvocationState.AMBIGUITY_TERMINAL,
        InvocationState.COMPENSATION_REQUIRED,
    ):
        return state
    return InvocationState.TERMINAL


def next_on_uncertain(
    state: InvocationState, *, replayable: bool, compensable: bool = False
) -> InvocationState:
    """Resolve a lost acknowledgement or route loss, keeping the three contracts apart.

    A replayable invocation becomes ``UNCERTAIN`` and may be reissued through the same
    identity; a compensable one becomes ``COMPENSATION_REQUIRED``; everything else
    becomes ``AMBIGUITY_TERMINAL``. None infers success or retries silently.
    """
    if state is InvocationState.TERMINAL:
        return state
    if replayable:
        return InvocationState.UNCERTAIN
    if compensable:
        return InvocationState.COMPENSATION_REQUIRED
    return InvocationState.AMBIGUITY_TERMINAL


def next_on_reissue(state: InvocationState) -> InvocationState:
    """Reissue a replayable invocation through its stable identity after uncertainty."""
    if state is InvocationState.UNCERTAIN:
        return InvocationState.ISSUED
    return state
