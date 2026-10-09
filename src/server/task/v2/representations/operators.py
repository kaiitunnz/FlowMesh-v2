from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.harness.boundary import BoundaryEventKind
from shared.inference import engine_profile_key
from shared.sandbox import SandboxEgressMode, SandboxRuntimeProfile
from shared.tasks import TaskType
from shared.tasks.specs import InferenceEmbodimentKind, ModelBindingMode
from shared.tools.facade import FacadeDescriptor as FacadeDescriptor

from .plan import ServiceFamilyRequirement
from .serving_size import DEFAULT_SERVING_SIZE, ServingSize


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
    # A re-drive may repeat an effect the author already caused, because the surface
    # deliberately carries no per-operation receipt to deduplicate against. It describes
    # what recovery may do, not a delivery the fabric promises.
    AUTHOR_OWNED_AT_LEAST_ONCE = "author_owned_at_least_once"


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


class MergeCombination(StrEnum):
    """How a merge combines the live records reaching it."""

    ONE_LIVE = "one_live"
    """Exactly one input route is live; the merge forwards its binding."""
    CONCAT = "concat"
    """Every live input contributes a member, in declared input order."""


class OperatorKind(StrEnum):
    """Discriminates the logical operator vocabulary."""

    LEAF = "leaf"
    AGENT = "agent"
    BRANCH = "branch"
    MERGE = "merge"
    SPAWN = "spawn"
    JOIN = "join"
    LOOP_CONTEXT = "loop_context"


REGION_OPERATOR_KINDS = frozenset(
    {
        OperatorKind.BRANCH,
        OperatorKind.MERGE,
        OperatorKind.SPAWN,
        OperatorKind.JOIN,
        OperatorKind.LOOP_CONTEXT,
    }
)
"""The operator kinds that settle in the ledger instead of dispatching."""


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
    ``adapter_source``) rather than the family key. ``engine_profile`` is the engine
    configuration that changes what a replica returns; it keys both and the replica
    serves it. ``serving_size`` is the hardware a replica runs at; a non-default size
    keys both.
    """

    model_config = ConfigDict(frozen=True)

    service_ref: str
    interface: ServiceInterface = ServiceInterface.CHAT
    adapter: str | None = None
    adapter_source: str | None = None
    isolation: str | None = None
    engine_profile: str | None = None
    serving_size: ServingSize = DEFAULT_SERVING_SIZE
    # How many conversations one invocation of this leaf carries, when its source names
    # them outright. A leaf resolving its prompts from upstream knows this only once
    # that value is in hand, and carries None until then.
    batch_size: int | None = Field(default=1, ge=1)
    # The most conversations any invocation of this leaf can carry. Each runs as its own
    # engine sequence, so this is what a resident embodiment's feasibility is screened
    # against before a prompt vector exists. A leaf declaring no bound carries None and
    # is screened against the request its preparation produced.
    max_batch_size: int | None = Field(default=1, ge=1)

    @property
    def service_family(self) -> str:
        """The reuse-domain identity: one base model, interface, isolation domain,
        engine profile, and serving size."""
        parts = [self.service_ref.strip(), self.interface.value]
        if self.isolation:
            parts.append(f"iso={self.isolation}")
        return "|".join(parts + self._engine_parts())

    @property
    def engine_batch_key(self) -> str:
        """The compatible model-runner and config key an admitted batch shares."""
        return "|".join(
            [self.service_ref.strip(), self.interface.value, *self._engine_parts()]
        )

    def family_requirement(self) -> ServiceFamilyRequirement:
        """Return the plan requirement naming this dependency's family."""
        return ServiceFamilyRequirement(
            family=self.service_family,
            engine_batch_key=self.engine_batch_key,
            isolation=self.isolation,
            serving_size=self.serving_size,
        )

    def _engine_parts(self) -> list[str]:
        # The default size adds no part, so a dependency restored without a size
        # resolves to the family it was stored under.
        parts = []
        if self.engine_profile is not None:
            parts.append(f"profile={engine_profile_key(self.engine_profile)}")
        if not self.serving_size.is_default:
            parts.append(f"size={self.serving_size.key()}")
        return parts


class InferenceEmbodimentEligibility(StrEnum):
    """Which physical embodiments one pinned inference contract admits."""

    RESIDENT_REQUIRED = "resident_required"
    SELF_CONTAINED_REQUIRED = "self_contained_required"
    LOCAL_ELIGIBLE = "local_eligible"


class InferenceEmbodimentBinding(BaseModel):
    """The submission-pinned embodiment disposition of an inference leaf.

    Part of the leaf's binding and its input cone, not a scheduler hint: a scheduler
    chooses among the embodiments a ``local_eligible`` disposition admits, and never
    reinterprets a required disposition as optional. ``primary`` is set only for
    ``local_eligible`` and names the embodiment the fabric uses when both are placeable.
    """

    model_config = ConfigDict(frozen=True)

    eligibility: InferenceEmbodimentEligibility
    primary: InferenceEmbodimentKind | None = None


class ResolvedEmbodiment(BaseModel):
    """The menu entry a task's dispatch is bound to.

    Routing, placement, and the result projection read it rather than re-deriving
    residence from the leaf's own service binding, which names every embodiment the leaf
    admits. It stays on the control plane: the worker runs the concrete task the
    dispatcher materializes from it, not the choice behind it.
    """

    model_config = ConfigDict(frozen=True)

    alternative_id: str
    kind: InferenceEmbodimentKind


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


class AgentSandboxBinding(BaseModel):
    """The resolved, submission-pinned local sandbox binding for an agent.

    It lowers to bounded local runtime actions inside the agent's own episode, on the
    holder its private state already selected. It names no host, worker, image cache,
    mount, endpoint, service dependency, residency intent, claim, or route: an agent's
    commands are a fenced private-state transition, not a service it invokes.

    ``network_egress`` is the one exception to that framing: a binding pinned to
    ``author_owned_at_least_once`` is an author-owned external-effect surface, declared
    here at binding granularity rather than per command, so no command acquires a
    receipt or an idempotency key of its own.
    """

    model_config = ConfigDict(frozen=True)

    profile: SandboxRuntimeProfile
    provenance: BindingProvenance
    network_egress: SandboxEgressMode = SandboxEgressMode.DENY


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
    # The service dependency this leaf consumes (an inference/embedding leaf its binding
    # admits to resident capacity). None for an ordinary in-process leaf.
    service_dependency: ServiceDependency | None = None
    # Set for every inference/embedding leaf; None for any other binding.
    embodiment: InferenceEmbodimentBinding | None = None


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
    # Set only for an agent whose authority declares ``sandbox.execute``.
    sandbox_binding: AgentSandboxBinding | None = None
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


type SelectorStep = str | int


class SelectionCase(BaseModel):
    """One literal selector value and the output port it selects."""

    model_config = ConfigDict(frozen=True)

    value: str
    port: str


class SelectionRule(BaseModel):
    """How a branch picks one output port from the record on one of its inputs.

    ``field`` walks the accepted input value; the value found must be a string. Without
    ``cases`` that string names an output port; with them it must equal one case value,
    which names the port. No other value selects anything.
    """

    model_config = ConfigDict(frozen=True)

    input: str
    field: tuple[SelectorStep, ...] = ()
    cases: tuple[SelectionCase, ...] | None = None
    version: int = 1


class BranchRegion(_OperatorBase):
    """Routes the record on its input to the one output port its rule selects."""

    kind: Literal[OperatorKind.BRANCH] = OperatorKind.BRANCH
    selection: str | None = None  # an unrunnable pre-rule selection, kept to decode
    rule: SelectionRule | None = None


class MergeRegion(_OperatorBase):
    """Typed input-port combination structure."""

    kind: Literal[OperatorKind.MERGE] = OperatorKind.MERGE
    combination: MergeCombination | None = None

    @field_validator("combination", mode="before")
    @classmethod
    def _tolerate_unknown_combination(cls, value: Any) -> Any:
        # A stored merge may carry any string; one this contract does not name keeps
        # its all-inputs behavior.
        if isinstance(value, str) and value not in MergeCombination:
            return None
        return value


class SpawnRegion(_OperatorBase):
    """A matched child-region boundary for streamed child creation.

    Its child is one operator (``child_template_ref``) or a declared multi-operator
    region definition (``child_definition_ref``).
    """

    kind: Literal[OperatorKind.SPAWN] = OperatorKind.SPAWN
    child_template_ref: str | None = None
    child_definition_ref: str | None = None
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
    """A structured ingress/feedback/egress region with a loop coordinate.

    ``carried`` ports seed time 0 and are replaced by each feedback; ``invariants`` bind
    once at ingress and stay readable at every time. ``body_ref`` names the region
    definition run at each time.
    """

    kind: Literal[OperatorKind.LOOP_CONTEXT] = OperatorKind.LOOP_CONTEXT
    loop_coordinate: str
    carried: tuple[Port, ...] = ()
    invariants: tuple[Port, ...] = ()
    body_ref: str | None = None


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


def spawned_only_region_owners(operators: Iterable[LogicalOperator]) -> dict[str, str]:
    """Map each child-region spawn to its owning agent, where that agent runs only as a
    spawned child.

    An agent a spawn instantiates, other than through its own recursive region, runs
    only as a spawned child, so every scope of its regions is nested and none of its
    joins delivers at the root.
    """
    ops = list(operators)
    region_owner = {
        ref.spawn_ref: op.operator_id
        for op in ops
        if isinstance(op, AgentOperator)
        for ref in op.child_region_refs
    }
    spawned_only = {
        op.child_template_ref
        for op in ops
        if isinstance(op, SpawnRegion)
        and op.child_template_ref
        and region_owner.get(op.operator_id) != op.child_template_ref
    }
    return {
        spawn: owner for spawn, owner in region_owner.items() if owner in spawned_only
    }
