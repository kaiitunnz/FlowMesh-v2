from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ResidencyIntent(BaseModel):
    """A plan-time residency preference hook.

    Carries warmth/reuse/affinity/preemption intent for a node. It is a
    preference, not a command to pin a process; it encodes no lifecycle policy.
    """

    model_config = ConfigDict(frozen=True)

    service_family: str | None = None
    warmth: str | None = None
    reuse_domain: str | None = None
    affinity: str | None = None
    preemption: str | None = None


class PhysicalNode(BaseModel):
    """One physical realization node of the plan.

    Maps to a logical operator through ``logical_ref`` (``None`` for a residency
    administration node that owns no logical operator). It carries no per-attempt
    placement and no transient endpoint address.
    """

    model_config = ConfigDict(frozen=True)

    node_id: str
    source_ref: str = Field(description="Source-map key back to the frontend source.")
    logical_ref: str | None = None
    episode: EpisodeSpec | None = None
    service_family_requirement: ServiceFamilyRequirement | None = None
    residency_intent: ResidencyIntent | None = None


class PhysicalExecutionPlan(BaseModel):
    """A finite, versioned, symbolic menu of legal physical lowerings.

    It records physical nodes and their source maps to the logical template.
    A node may carry an :class:`EpisodeSpec` with its run-to-yield boundary and
    lightweight resource/liveness annotations. Per-operator alternative-lowering
    menus and state-placement schemas remain reserved extension points; no
    phase-by-phase allocation schema is frozen here.
    """

    model_config = ConfigDict(frozen=True)

    plan_version: VersionId
    template_version: VersionId
    nodes: tuple[PhysicalNode, ...] = ()

    @model_validator(mode="after")
    def _validate_node_ids(self) -> "PhysicalExecutionPlan":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node_id in physical execution plan.")
        return self
