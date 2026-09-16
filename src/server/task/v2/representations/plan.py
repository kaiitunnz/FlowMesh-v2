from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.tasks.specs import InferenceEmbodimentKind

from ..mode import LoweringStrategy
from .versioning import VersionId


class EpisodeBoundaryKind(StrEnum):
    """The boundary that closes one execution episode.

    A run-to-yield cut ends an episode at the first boundary its operators
    reach; the kind names why the episode yields. ``TASK`` is the conservative
    per-operator cut of the transparent lowering.
    """

    TASK = "task"
    SERVICE_ISSUE = "service_issue"
    EFFECT = "effect"
    DURABLE_CHECKPOINT = "durable_checkpoint"
    ISOLATION_CLASS = "isolation_class"
    CONTINUATION = "continuation"
    REGION_BLOCKING = "region_blocking"


class EpisodeSpec(BaseModel):
    """The episode a physical node executes.

    ``boundary`` is the cut that closes the episode. ``fused_refs`` lists the
    additional logical operators executed in the same episode beyond the node's
    ``logical_ref`` anchor. ``resource_class`` names the episode's executor-binding
    family and ``liveness_key`` is a reserved liveness annotation; a scheduler
    feasibility check may read them, and neither names a worker, replica, or capacity
    object.
    """

    model_config = ConfigDict(frozen=True)

    boundary: EpisodeBoundaryKind
    fused_refs: tuple[str, ...] = ()
    resource_class: str | None = None
    liveness_key: str | None = None


class ServiceFamilyRequirement(BaseModel):
    """A plan-time service-family requirement hook.

    Names the service family a node needs. It encodes no admission, transport,
    or allocation policy; resident-capacity control consumes it later.
    """

    model_config = ConfigDict(frozen=True)

    family: str
    engine_batch_key: str | None = None
    isolation: str | None = None


# The one residency warmth preference the fabric expresses: a family carrying it is
# retained longer after its last credit-bearing invocation.
WARM = "warm"

# The policy that answers every lowering hook as the compiler itself would.
CONSERVATIVE_POLICY = "conservative"


class ResidencyIntent(BaseModel):
    """A plan-time residency preference hook.

    Carries warmth/reuse/affinity/preemption intent for a node. It is a
    preference, not a command to pin a process; it encodes no lifecycle policy.
    ``required`` marks a resident dependency the plan cannot run without (a resident
    agent model binding), as opposed to an optional warmth preference; it still
    triggers no allocation here.
    """

    model_config = ConfigDict(frozen=True)

    service_family: str | None = None
    required: bool = False
    warmth: str | None = None
    reuse_domain: str | None = None
    affinity: str | None = None
    preemption: str | None = None
    # Scoped to one embodiment candidate: it registers demand only once that candidate
    # is durably selected, so lifecycle control passes over it while the menu is
    # unresolved. An ordinary non-required warmth preference applies unresolved.
    conditional: bool = False


class LocalExecutionEnvelope(BaseModel):
    """What a self-contained candidate needs on the worker that runs it.

    A resident binding places no worker-local model requirement, so a candidate that
    loads the model locally states its own executor and accelerator requirement rather
    than inheriting the resident leaf's empty one.
    """

    model_config = ConfigDict(frozen=True)

    executor_key: str
    gpu_count: int | None = None


class InferenceEmbodimentCandidate(BaseModel):
    """One physical embodiment of a leaf's pinned inference contract.

    A ``resident_served`` candidate carries the service family it admits to and a
    conditional residency intent; a ``self_contained`` candidate carries the local
    envelope it needs. Both run the same declared contract, so the menu holds their
    shared fingerprint and they share the node's single source map.
    """

    model_config = ConfigDict(frozen=True)

    alternative_id: str
    kind: InferenceEmbodimentKind
    episode: EpisodeSpec
    local: LocalExecutionEnvelope | None = None
    service_family_requirement: ServiceFamilyRequirement | None = None
    residency_intent: ResidencyIntent | None = None

    @model_validator(mode="after")
    def _validate_kind_envelope(self) -> "InferenceEmbodimentCandidate":
        resident = self.kind is InferenceEmbodimentKind.RESIDENT_SERVED
        if resident and (self.service_family_requirement is None or self.local):
            raise ValueError(
                "a resident_served candidate carries a service family requirement "
                "and no local envelope."
            )
        if not resident and (self.local is None or self.service_family_requirement):
            raise ValueError(
                "a self_contained candidate carries a local envelope and no service "
                "family requirement."
            )
        if resident and not (
            self.residency_intent and self.residency_intent.conditional
        ):
            raise ValueError(
                "a resident_served candidate carries a conditional residency intent, "
                "so an unselected menu registers no demand."
            )
        return self


class InferenceEmbodimentMenu(BaseModel):
    """The finite set of contract-equivalent embodiments one inference node admits.

    The compiler proves the entries equivalent before emitting them and records the
    proof as ``contract_fingerprint``, which every entry shares. ``primary`` is the
    submission-pinned entry a conservative scheduler selects; another entry is reachable
    only through a selection the runtime durably records.
    """

    model_config = ConfigDict(frozen=True)

    contract_fingerprint: str
    primary: str
    candidates: tuple[InferenceEmbodimentCandidate, ...]
    # The most conversations one invocation of the node can carry. Every entry runs the
    # same ones, so it belongs to the menu rather than to any single entry. A node whose
    # prompts come from upstream is screened against this bound, not an exact count; one
    # declaring no bound carries None until its preparation reports the exact count.
    max_batch_size: int | None = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_candidates(self) -> "InferenceEmbodimentMenu":
        ids = [candidate.alternative_id for candidate in self.candidates]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError(
                "an embodiment menu holds at least two distinctly identified entries."
            )
        if self.primary not in ids:
            raise ValueError(
                f"primary embodiment {self.primary!r} is not a menu entry."
            )
        return self

    def candidate(self, alternative_id: str) -> InferenceEmbodimentCandidate | None:
        return next(
            (c for c in self.candidates if c.alternative_id == alternative_id), None
        )


class PhysicalNode(BaseModel):
    """One physical realization node of the plan.

    Maps to a logical operator through ``logical_ref`` (``None`` for a residency
    administration node that owns no logical operator). It carries no per-attempt
    placement and no transient endpoint address.

    A node carrying an ``embodiment_menu`` holds its boundary and resident annotations
    per candidate instead of on the node, so nothing reads an unresolved menu as one
    episode or as a registered service dependency.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str
    source_ref: str = Field(description="Source-map key back to the frontend source.")
    logical_ref: str | None = None
    episode: EpisodeSpec | None = None
    service_family_requirement: ServiceFamilyRequirement | None = None
    residency_intent: ResidencyIntent | None = None
    embodiment_menu: InferenceEmbodimentMenu | None = None


class LoweringProvenance(BaseModel):
    """The lowering a plan was produced under.

    Names the episode strategy and the advisory policy effective at each lowering
    hook, so a dry-run inspection and a persisted submission are comparable at the
    decisions that produced them. A hook a deployment selects no policy for records
    ``conservative``, its effective policy.
    """

    model_config = ConfigDict(frozen=True)

    strategy: LoweringStrategy
    fusion: str = CONSERVATIVE_POLICY
    residency: str = CONSERVATIVE_POLICY
    service_family: str = CONSERVATIVE_POLICY


class PhysicalExecutionPlan(BaseModel):
    """A finite, versioned, symbolic menu of legal physical lowerings.

    It records physical nodes and their source maps to the logical template.
    A node may carry an :class:`EpisodeSpec` with its run-to-yield boundary and
    lightweight resource/liveness annotations, or an :class:`InferenceEmbodimentMenu`
    of the contract-equivalent embodiments one inference node admits. It holds no
    phase-by-phase allocation schema.
    """

    model_config = ConfigDict(frozen=True)

    plan_version: VersionId
    template_version: VersionId
    nodes: tuple[PhysicalNode, ...] = ()
    lowering: LoweringProvenance | None = None

    @model_validator(mode="after")
    def _validate_node_ids(self) -> "PhysicalExecutionPlan":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node_id in physical execution plan.")
        return self
