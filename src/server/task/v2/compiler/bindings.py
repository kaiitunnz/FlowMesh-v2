from enum import StrEnum

from shared.tasks import TaskType

from ..representations.operators import (
    AuthorityCeiling,
    BindingKey,
    BoundaryEventKind,
    BoundarySignature,
    DeterminismClass,
    EffectClass,
    EqualityRelation,
    EqualityRelationKind,
    InputProvenanceKind,
    LeafProfile,
    RecoveryClass,
)

_TRAINING_TYPES: frozenset[TaskType] = frozenset(
    {
        TaskType.SFT,
        TaskType.LORA_SFT,
        TaskType.PPO,
        TaskType.DPO,
        TaskType.IMAGE_CLASSIFICATION_TRAINING,
    }
)

_SEMANTIC = EqualityRelation(kind=EqualityRelationKind.SEMANTIC)
_BITWISE = EqualityRelation(kind=EqualityRelationKind.BITWISE)


class BindingClass(StrEnum):
    """How the compiler lowers a legacy task type into the operator vocabulary.

    The legacy ``taskType`` set is a binding registry over the small v2 operator
    vocabulary, not a proliferation of logical leaf kinds: most types lower to a
    generic ``Leaf``, ``agent`` lowers to the opaque-body ``Agent``, and ``serve``
    lowers to a residency request surface rather than a result-owning leaf.
    """

    LEAF = "leaf"
    AGENT = "agent"
    RESIDENCY = "residency"


def binding_class(task_type: TaskType) -> BindingClass:
    """Classify how a legacy task type binds into the operator vocabulary."""
    if task_type == TaskType.AGENT:
        return BindingClass.AGENT
    if task_type == TaskType.SERVE:
        return BindingClass.RESIDENCY
    return BindingClass.LEAF


def is_training(task_type: TaskType) -> bool:
    """Whether a task type is a non-pure producer of a new ``ModelRef``."""
    return task_type in _TRAINING_TYPES


def leaf_profile(task_type: TaskType) -> LeafProfile:
    """Return the indicative binding profile for a legacy task type."""
    binding = BindingKey(task_type=task_type)
    det = DeterminismClass.SAMPLED
    effect = EffectClass.PURE
    recovery = RecoveryClass.RECORD
    provenance = InputProvenanceKind.EXTERNAL_PINNED
    equality: EqualityRelation | None = None

    match task_type:
        case TaskType.EMBEDDING:
            det, recovery, equality = (
                DeterminismClass.DETERMINISTIC_SEMANTIC,
                RecoveryClass.RECOMPUTE,
                _SEMANTIC,
            )
        case TaskType.RAG:
            det, recovery, equality, provenance = (
                DeterminismClass.DETERMINISTIC_SEMANTIC,
                RecoveryClass.RECORD,
                _SEMANTIC,
                InputProvenanceKind.LIVE_INPUT,
            )
        case TaskType.DATA_PROFILING:
            det, recovery, equality = (
                DeterminismClass.DETERMINISTIC_SEMANTIC,
                RecoveryClass.RECOMPUTE,
                _SEMANTIC,
            )
        case TaskType.ECHO:
            det, recovery, equality = (
                DeterminismClass.DETERMINISTIC_BITWISE,
                RecoveryClass.RECOMPUTE,
                _BITWISE,
            )
        case TaskType.DATA_RETRIEVAL:
            provenance = InputProvenanceKind.LIVE_INPUT
        case TaskType.API | TaskType.SSH | TaskType.SERVE:
            effect, provenance = (
                EffectClass.EXTERNAL_EFFECT,
                InputProvenanceKind.LIVE_INPUT,
            )
        case (
            TaskType.SFT
            | TaskType.LORA_SFT
            | TaskType.PPO
            | TaskType.DPO
            | TaskType.IMAGE_CLASSIFICATION_TRAINING
        ):
            effect = EffectClass.PRIVATE_STATE
        case _:
            pass

    return LeafProfile(
        determinism=det,
        effect=effect,
        recovery=recovery,
        input_provenance=provenance,
        binding=binding,
        output_equality=equality,
    )


def default_agent_boundary() -> BoundarySignature:
    """Return the finite boundary signature an ``Agent`` leaf exposes."""
    return BoundarySignature(
        events=(
            BoundaryEventKind.INVOCATION,
            BoundaryEventKind.SPAWN,
            BoundaryEventKind.YIELD,
            BoundaryEventKind.EXTERNAL_EFFECT,
            BoundaryEventKind.STATE_ACCESS,
        )
    )


def default_agent_authority() -> AuthorityCeiling:
    """Return the default (empty) authority ceiling for an ``Agent`` leaf."""
    return AuthorityCeiling()
