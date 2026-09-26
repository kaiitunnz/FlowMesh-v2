from ._base import StrictBaseModel
from .components import TaskMetadata
from .envelope import TaskSpecStrict
from .result_binding import ResultBinding


class MergedChildTaskStrict(StrictBaseModel):
    task_id: str
    owner_id: str
    workflow_id: str
    spec: TaskSpecStrict
    metadata: TaskMetadata | None = None
    upstream_results: dict[str, ResultBinding] | None = None
