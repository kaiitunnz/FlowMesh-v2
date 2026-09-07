from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.harness.boundary import BoundaryEventKind
from shared.tasks import TaskType
from shared.tasks.specs import ModelBindingMode
from shared.tools.facade import FacadeDescriptor as FacadeDescriptor


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


class ServiceInterface(StrEnum):
    """The service interface a resident invocation targets.

    Distinct interfaces never share an engine batch or replica pool: a chat/completion
    runner and an embedding runner are different services even for the same model name.
    """

    CHAT = "chat"
    EMBEDDING = "embedding"


class ServiceDependency(BaseModel):
    """A normalized resident service dependency an invocation must satisfy.

    Names the logical model/service reference, the service interface, and the adapter
    and isolation constraints the invocation must satisfy; it names no replica or
    worker. An Agent's resident model binding and an inference/embedding leaf's resident
    binding both normalize into this one form.

    ``service_family`` and ``engine_batch_key`` key the reuse domain on the base model,
    interface, and isolation domain, so a matching model reference alone does not share
    a service across a differing interface, base model, or isolation domain. An adapter
    co-batches within a compatible base engine through its own slot: it loads into the
    base replica and the request selects it, so it rides ``adapter`` (with its loadable
    ``adapter_source``) rather than the family key.
    """

    model_config = ConfigDict(frozen=True)

    service_ref: str
    interface: ServiceInterface = ServiceInterface.CHAT
    adapter: str | None = None
    adapter_source: str | None = None
    isolation: str | None = None

    @property
    def service_family(self) -> str:
        """The reuse-domain identity: one base model, interface, isolation domain."""
        parts = [self.service_ref.strip(), self.interface.value]
        if self.isolation:
            parts.append(f"iso={self.isolation}")
        return "|".join(parts)

    @property
    def engine_batch_key(self) -> str:
        """The compatible model-runner and config key an admitted batch shares."""
        return "|".join([self.service_ref.strip(), self.interface.value])


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


class BindingProvenance(StrEnum):
    """Which resolution tier supplied an effective binding field at submission."""

    SOURCE = "source"
    DEFAULT = "default"
    FALLBACK = "fallback"


class HarnessBindingProvenance(BaseModel):
    """Per-field provenance of the resolved harness binding."""

    model_config = ConfigDict(frozen=True)

    backend: BindingProvenance
    version: BindingProvenance


class AgentHarnessBinding(BaseModel):
    """The resolved, submission-pinned harness binding for an agent.

    ``params`` is a pinned copy of the source non-secret adapter configuration.
    Kept version-pinned so a later deployment-default change cannot move a live
    activation.
    """

    backend: str
    version: str
    params: dict[str, Any] = {}
    provenance: HarnessBindingProvenance


class ModelBindingProvenance(BaseModel):
    """Per-field provenance of the resolved model-gateway binding."""

    model_config = ConfigDict(frozen=True)

    mode: BindingProvenance
    url: BindingProvenance
    model: BindingProvenance


class AgentModelGatewayBinding(BaseModel):
    """The resolved, submission-pinned managed-model dependency for an agent.

    It names a model dependency, never a credential: ``secret_ref`` is an authorized
    server-side reference, and no secret value is ever stored here. A ``resident``
    binding carries a ``service_model_ref`` and no url/credential.
    """

    model_config = ConfigDict(frozen=True)

    mode: ModelBindingMode
    url: str | None = None
    model: str | None = None
    secret_ref: str | None = None
    service_model_ref: str | None = None
    provenance: ModelBindingProvenance


def agent_service_dependency(
    binding: AgentModelGatewayBinding | None,
) -> ServiceDependency | None:
    """Normalize an agent's resident model binding into a generic service dependency.

    A non-resident or reference-less binding names no resident dependency. An agent's
    managed model boundary is a chat/completion interface.
    """
    if (
        binding is None
        or binding.mode is not ModelBindingMode.RESIDENT
        or not binding.service_model_ref
    ):
        return None
    return ServiceDependency(
        service_ref=binding.service_model_ref, interface=ServiceInterface.CHAT
    )


def operator_service_dependency(
    op: "LogicalOperator | None",
) -> ServiceDependency | None:
    """The normalized resident dependency an operator consumes, agent or leaf."""
    if isinstance(op, AgentOperator):
        return agent_service_dependency(op.model_binding)
    if isinstance(op, LeafOperator):
        return op.service_dependency
    return None


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
    # A resident-required service dependency this leaf consumes (an inference/embedding
    # leaf served by resident capacity). None for an ordinary in-process leaf.
    service_dependency: ServiceDependency | None = None


class AgentOperator(_OperatorBase):
    """The special opaque-body leaf.

    Its local turns are not authored as a micro-operator graph; it exposes a
    finite boundary signature and an authority ceiling.
    """

    kind: Literal[OperatorKind.AGENT] = OperatorKind.AGENT
    binding: BindingKey
    # The resolved, submission-pinned harness and managed-model bindings. Both are set
    # by the compiler for a v2 agent (or a diagnostic is recorded); None only when an
    # operator is constructed outside compilation.
    harness_binding: AgentHarnessBinding | None = None
    model_binding: AgentModelGatewayBinding | None = None
    authority: AuthorityCeiling = AuthorityCeiling()
    boundary: BoundarySignature = BoundarySignature()
    guard: ConditionGuard | None = None
    # The fabric-owned facade tools the gateway injects for this agent, pinned at
    # compile time from its declared tools and child regions.
    facades: tuple[FacadeDescriptor, ...] = ()
    # Explicitly-declared input ports this agent receives on its first turn (opt-in
    # dataflow), distinct from the auto-synthesized ordering-only port. Empty keeps
    # bare dependencies ordering-only.
    declared_input_ports: tuple[str, ...] = ()
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
