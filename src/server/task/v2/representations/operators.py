from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.tasks import TaskType


class DeterminismClass(StrEnum):
    """How an operator's output relates to a repeated run over the same inputs."""

    DETERMINISTIC_BITWISE = "deterministic_bitwise"
    DETERMINISTIC_SEMANTIC = "deterministic_semantic"
    SAMPLED = "sampled"


class EffectClass(StrEnum):
    """The externally visible effect an operator may produce."""

    PURE = "pure"
    PRIVATE_STATE = "private_state"
    EXTERNAL_EFFECT = "external_effect"


class RecoveryClass(StrEnum):
    """How an operation may be recovered within one execution."""

    RECOMPUTE = "recompute"
    RECORD = "record"
    REPLAY_WITH_DEDUP = "replay_with_dedup"
    AMBIGUITY_TERMINAL = "ambiguity_terminal"


class InputProvenanceKind(StrEnum):
    """Whether external input is pinned to invariant state or read live."""

    EXTERNAL_PINNED = "external_pinned"
    LIVE_INPUT = "live_input"


class EffectReplayContract(StrEnum):
    """Declared behavior of an external-effect boundary after an uncertain failure."""

    REPLAYABLE_DEDUP = "replayable_dedup"
    COMPENSABLE = "compensable"
    AMBIGUITY_TERMINAL = "ambiguity_terminal"


class BoundaryEventKind(StrEnum):
    """A fabric-relevant event an operator's boundary signature may emit.

    ``SPAWN_SEAL`` closes an agent's child-init producer: a ``SPAWN`` never seals it, so
    an agent yields ``SPAWN_SEAL`` when it will issue no more children.
    """

    INVOCATION = "invocation"
    SPAWN = "spawn"
    SPAWN_SEAL = "spawn_seal"
    YIELD = "yield"
    EXTERNAL_EFFECT = "external_effect"
    STATE_ACCESS = "state_access"


class PortKind(StrEnum):
    """The typed carrier of a port."""

    VALUE = "value"
    STATE_REFERENCE = "state_reference"
    MODEL_REF = "model_ref"


class EqualityRelationKind(StrEnum):
    """The equality a deterministic output declares."""

    BITWISE = "bitwise"
    SEMANTIC = "semantic"


class JoinCompletion(StrEnum):
    """Completion policy of a join region."""

    ALL_SETTLED = "all_settled"
    ALL_SUCCEED = "all_succeed"
    ANY = "any"
    FIRST_K = "first_k"
    PREDICATE = "predicate"


class ResidualPolicy(StrEnum):
    """Fate of a spawn scope's materialized children when an early join releases.

    ``CONTINUE`` leaves the child-init capability open (late children stay legal);
    ``DRAIN`` seals it so materialized children still settle but no new one is created;
    ``CANCEL`` revokes it and cancels every not-yet-settled materialized child.
    """

    CONTINUE = "continue"
    DRAIN = "drain"
    CANCEL = "cancel"


class OperatorKind(StrEnum):
    """Discriminates the logical operator vocabulary."""

    LEAF = "leaf"
    AGENT = "agent"
    BRANCH = "branch"
    MERGE = "merge"
    SPAWN = "spawn"
    JOIN = "join"
    LOOP_CONTEXT = "loop_context"


class ModelRef(BaseModel):
    """A logical, versioned reference to a model identity.

    The architecture is a template-level identity; the version is a runtime
    identity. Changing architecture is a template change; changing version is a
    new ``ModelRef``.
    """

    model_config = ConfigDict(frozen=True)

    architecture: str = Field(description="Logical model architecture identity.")
    version: str | None = Field(default=None, description="Model version identity.")


class StateReference(BaseModel):
    """A typed, logical reference to durable state carried across ports."""

    model_config = ConfigDict(frozen=True)

    ref_kind: str = Field(description="State reference kind (artifact, checkpoint, …).")
    identity: str | None = Field(default=None, description="Logical state identity.")


class Port(BaseModel):
    """A typed input/output port on a logical operator or region."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: PortKind = PortKind.VALUE
    value_type: str | None = None
    model_ref: ModelRef | None = None
    state_ref: StateReference | None = None


class EqualityRelation(BaseModel):
    """The equality relation a deterministic output type declares."""

    model_config = ConfigDict(frozen=True)

    kind: EqualityRelationKind
    relation_id: str | None = Field(
        default=None, description="Portable semantic-equivalence relation identity."
    )


class BindingKey(BaseModel):
    """Symbolic executor-binding identity for a leaf.

    Names compatible implementation semantics — never a worker, replica, episode
    cut, or hardware-feasibility commitment.
    """

    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    backend: str | None = Field(default=None, description="Optional backend hint.")


class AuthorityCeiling(BaseModel):
    """Declared authority bound for an operator.

    ``invoke`` is which service/tool interfaces it may request; ``delegate`` is
    which authority it may pass to a child region. Distinct from progress
    capabilities and route authorization.
    """

    model_config = ConfigDict(frozen=True)

    invoke: tuple[str, ...] = ()
    delegate: tuple[str, ...] = ()


class BoundarySignature(BaseModel):
    """The finite set of fabric-relevant events an operator may emit."""

    model_config = ConfigDict(frozen=True)

    events: tuple[BoundaryEventKind, ...] = ()


class ChildRegionRef(BaseModel):
    """A named reference from an agent's spawn seam to one declared child region.

    ``name`` is the stable role a ``SpawnRequest`` selects; ``spawn_ref`` is the
    operator id of the matched ``Spawn`` region it resolves to. The role name differs
    from the operator id so a request names a role without knowing compiled ids. A
    region's entry target, per-site authority ceiling, and completion/residual contract
    live on the referenced ``Spawn``/``Join`` region, never inline here.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    spawn_ref: str


class EffectBoundary(BaseModel):
    """A source-mapped declared external-effect obligation."""

    model_config = ConfigDict(frozen=True)

    effect_class: EffectClass
    replay_contract: EffectReplayContract | None = None
    source_ref: str | None = None


class ConditionGuard(BaseModel):
    """Recorded conditional-dispatch metadata for an operator.

    Captures a legacy ``condition`` symbolically. It records the branch guard;
    it does not execute it in this representation.
    """

    model_config = ConfigDict(frozen=True)

    node: str
    field: str
    equals: str


class LeafProfile(BaseModel):
    """The typed profile of a generic leaf operator."""

    model_config = ConfigDict(frozen=True)

    determinism: DeterminismClass
    effect: EffectClass
    recovery: RecoveryClass
    input_provenance: InputProvenanceKind
    binding: BindingKey
    output_equality: EqualityRelation | None = None


class _OperatorBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    operator_id: str
    source_ref: str = Field(description="Source-map key back to the frontend source.")
    inputs: tuple[Port, ...] = ()
    outputs: tuple[Port, ...] = ()


class LeafOperator(_OperatorBase):
    """Generic typed declared computation.

    ``residency_only`` marks a binding that administers resident capacity (e.g.
    legacy ``serve``) rather than owning a logical result.
    """

    kind: Literal[OperatorKind.LEAF] = OperatorKind.LEAF
    profile: LeafProfile
    guard: ConditionGuard | None = None
    residency_only: bool = False


class AgentOperator(_OperatorBase):
    """The special opaque-body leaf.

    Its local turns are not authored as a micro-operator graph; it exposes a
    finite boundary signature and an authority ceiling.
    """

    kind: Literal[OperatorKind.AGENT] = OperatorKind.AGENT
    binding: BindingKey
    authority: AuthorityCeiling = AuthorityCeiling()
    boundary: BoundarySignature = BoundarySignature()
    guard: ConditionGuard | None = None
    # The finite, uniquely named set of declared child regions a spawn_agent may select.
    child_region_refs: tuple[ChildRegionRef, ...] = ()
    # Legacy single-target shorthand the compiler normalizes into one declared region;
    # a declaration that sets both this and child_region_refs is rejected.
    child_template_ref: str | None = None


class BranchRegion(_OperatorBase):
    """Typed output-port structure with a selection rule."""

    kind: Literal[OperatorKind.BRANCH] = OperatorKind.BRANCH
    selection: str | None = None


class MergeRegion(_OperatorBase):
    """Typed input-port combination structure."""

    kind: Literal[OperatorKind.MERGE] = OperatorKind.MERGE
    combination: str | None = None


class SpawnRegion(_OperatorBase):
    """A matched child-region boundary for streamed child creation."""

    kind: Literal[OperatorKind.SPAWN] = OperatorKind.SPAWN
    child_template_ref: str | None = None
    authority: AuthorityCeiling = AuthorityCeiling()


class JoinPredicate(BaseModel):
    """A join's declared early-release predicate over its settled qualifiers.

    A count threshold with a monotonicity flag: a monotone predicate releases on the
    first witness, a non-monotone one waits for frontier closure.
    """

    model_config = ConfigDict(frozen=True)

    min_qualifiers: int = 1
    monotone: bool = True


class JoinRegion(_OperatorBase):
    """A child-collection region with a declared completion policy.

    ``first_k`` and ``predicate`` parametrize the early-completion policies; a no-winner
    early join resolves ``EXPLICIT_EMPTY`` unless ``no_winner_failure`` opts into
    ``DECLARED_FAILURE``.
    """

    kind: Literal[OperatorKind.JOIN] = OperatorKind.JOIN
    completion: JoinCompletion
    residual_policy: str | None = None
    first_k: int | None = None
    predicate: JoinPredicate | None = None
    no_winner_failure: bool = False


class LoopContextRegion(_OperatorBase):
    """A structured ingress/feedback/egress region with a loop coordinate."""

    kind: Literal[OperatorKind.LOOP_CONTEXT] = OperatorKind.LOOP_CONTEXT
    loop_coordinate: str
    carried: tuple[Port, ...] = ()


type LogicalOperator = Annotated[
    LeafOperator
    | AgentOperator
    | BranchRegion
    | MergeRegion
    | SpawnRegion
    | JoinRegion
    | LoopContextRegion,
    Field(discriminator="kind"),
]
