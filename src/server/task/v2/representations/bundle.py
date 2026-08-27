from pydantic import BaseModel, ConfigDict

from .plan import PhysicalExecutionPlan
from .source import FrontendWorkflowSource
from .template import LogicalWorkflowTemplate


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
