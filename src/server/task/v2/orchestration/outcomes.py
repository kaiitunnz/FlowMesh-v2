"""Invocation/effect outcome semantics for the acyclic compatibility path.

Three concerns live here as pure logic: which operators the initial v2 runtime
admits, whether a settled work item may be recomputed or must be restored, and the
minimal durable invocation state machine that keeps an uncertain non-replayable
effect from silently retrying or reporting success.
"""

from ..representations.operators import (
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
    only when it declares a deduplication boundary. Everything else is non-replayable
    and an uncertain outcome becomes ambiguity-terminal.
    """
    if effect is EffectClass.PURE:
        return True
    return (
        effect is EffectClass.EXTERNAL_EFFECT
        and replay_contract is EffectReplayContract.REPLAYABLE_DEDUP
    )


def classify_recovery(profile: LeafProfile) -> RecoveryDisposition:
    """Whether a settled operation may be recomputed within one execution.

    Only a pure/hermetic deterministic operation over a complete pinned input cone
    may recompute (§5.5.4); sampling, unpinned reads, and effects must restore their
    recorded outcome before later work relies on them.
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
    """Reject an operator the initial v2 runtime cannot honor.

    The compatibility path runs effect-free operations, or external effects declared
    safely replayable with a deduplication boundary. Private-state operations,
    non-replayable external effects, and residency administration wait for later PRs.
    """
    if residency_only:
        raise AdmissionError(
            f"operator {operator_id!r} administers resident capacity, which the v2 "
            "compatibility path does not orchestrate yet"
        )
    if effect is EffectClass.PURE:
        return
    if is_replayable(effect, replay_contract):
        return
    raise AdmissionError(
        f"operator {operator_id!r} declares effect {effect.value!r} with replay "
        f"contract {replay_contract.value if replay_contract else None!r}; the v2 "
        "compatibility path admits only effect-free or replayable-with-dedup operations"
    )


def next_on_acknowledge(state: InvocationState) -> InvocationState:
    """Advance a pending invocation to the generic engine-enqueue acknowledgement."""
    if state in (InvocationState.ISSUED, InvocationState.UNCERTAIN):
        return InvocationState.ACKNOWLEDGED
    return state


def next_on_terminal(state: InvocationState) -> InvocationState:
    """Advance an invocation to its terminal receipt."""
    if state is InvocationState.AMBIGUITY_TERMINAL:
        return state
    return InvocationState.TERMINAL


def next_on_uncertain(state: InvocationState, replayable: bool) -> InvocationState:
    """Resolve a lost acknowledgement or route loss.

    A replayable invocation becomes ``UNCERTAIN`` and may be reissued through the same
    identity; a non-replayable one becomes ``AMBIGUITY_TERMINAL`` and is never silently
    retried or reported as success.
    """
    if state is InvocationState.TERMINAL:
        return state
    return (
        InvocationState.UNCERTAIN if replayable else InvocationState.AMBIGUITY_TERMINAL
    )


def next_on_reissue(state: InvocationState) -> InvocationState:
    """Reissue a replayable invocation through its stable identity after uncertainty."""
    if state is InvocationState.UNCERTAIN:
        return InvocationState.ISSUED
    return state
