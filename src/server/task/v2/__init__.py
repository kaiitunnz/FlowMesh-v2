from pydantic import BaseModel, ConfigDict

from .mode import V2_API_VERSION, ExecutionMode
from .operators import (
    AgentOperator,
    AuthorityCeiling,
    BindingKey,
    BoundaryEventKind,
    BoundarySignature,
    BranchRegion,
    ConditionGuard,
    DeterminismClass,
    EffectBoundary,
    EffectClass,
    EffectReplayContract,
    EqualityRelation,
    EqualityRelationKind,
    InputProvenanceKind,
    JoinCompletion,
    JoinRegion,
    LeafOperator,
    LeafProfile,
    LogicalOperator,
    LoopContextRegion,
    MergeRegion,
    ModelRef,
    OperatorKind,
    Port,
    PortKind,
    RecoveryClass,
    SpawnRegion,
    StateReference,
)
from .plan import (
    PhysicalExecutionPlan,
    PhysicalNode,
    ResidencyIntent,
    ServiceFamilyRequirement,
)
from .project import project_acyclic
from .results import (
    CardinalityKind,
    LegacyLogicalTaskProjection,
    ReleaseConditionKind,
    ResultDeclaration,
    Visibility,
)
from .source import FrontendWorkflowSource
from .template import (
    LogicalWorkflowTemplate,
    ResourceDeclaration,
    SourceMapEntry,
    TemplateEdge,
    ToolDeclaration,
)
from .versioning import VersionId, content_digest


class PersistedV2Workflow(BaseModel):
    """The durable plan-time bundle for a v2 workflow submission.

    Bundles the immutable frontend source with the logical template and physical
    plan. Persisted as one immutable record and never rewritten in place; a
    revision is a compatible successor.
    """

    model_config = ConfigDict(frozen=True)

    source: FrontendWorkflowSource
    template: LogicalWorkflowTemplate
    plan: PhysicalExecutionPlan


__all__ = [
    "AgentOperator",
    "AuthorityCeiling",
    "BindingKey",
    "BoundaryEventKind",
    "BoundarySignature",
    "BranchRegion",
    "CardinalityKind",
    "ConditionGuard",
    "DeterminismClass",
    "EffectBoundary",
    "EffectClass",
    "EffectReplayContract",
    "EqualityRelation",
    "EqualityRelationKind",
    "ExecutionMode",
    "FrontendWorkflowSource",
    "InputProvenanceKind",
    "JoinCompletion",
    "JoinRegion",
    "LeafOperator",
    "LeafProfile",
    "LegacyLogicalTaskProjection",
    "LogicalOperator",
    "LogicalWorkflowTemplate",
    "LoopContextRegion",
    "MergeRegion",
    "ModelRef",
    "OperatorKind",
    "PersistedV2Workflow",
    "PhysicalExecutionPlan",
    "PhysicalNode",
    "Port",
    "PortKind",
    "RecoveryClass",
    "ReleaseConditionKind",
    "ResidencyIntent",
    "ResourceDeclaration",
    "ResultDeclaration",
    "ServiceFamilyRequirement",
    "SourceMapEntry",
    "SpawnRegion",
    "StateReference",
    "TemplateEdge",
    "ToolDeclaration",
    "V2_API_VERSION",
    "VersionId",
    "Visibility",
    "content_digest",
    "project_acyclic",
]
