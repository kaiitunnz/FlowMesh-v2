from pydantic import BaseModel, Field

from server.task.v2 import InspectionReport


class WorkflowSubmitTaskEntry(BaseModel):
    task_id: str = Field(description="Task identifier.")
    status: str | None = Field(default=None, description="Current task status.")
    assigned_worker: str | None = Field(
        default=None, description="Worker assigned to the task."
    )
    topic: str | None = Field(default=None, description="Dispatch topic.")
    waiting_on: list[str] | None = Field(
        default=None, description="Pending dependency task IDs."
    )
    depends_on: list[str] | None = Field(
        default=None, description="Dependency task IDs."
    )
    attempts: int | None = Field(default=None, description="Attempt count.")
    max_attempts: int | None = Field(default=None, description="Max retry count.")
    load: int | None = Field(default=None, description="Task load score.")


class WorkflowSubmitResponse(BaseModel):
    ok: bool = Field(description="Whether the submission succeeded.")
    workflow_id: str = Field(description="Workflow identifier.")
    count: int = Field(description="Number of tasks in the workflow.")
    tasks: list[WorkflowSubmitTaskEntry] = Field(description="Submitted task entries.")


class WorkflowValidateTaskEntry(BaseModel):
    task_id: str = Field(
        description="Mock task identifier, not associated with any real task."
    )
    graph_node_name: str | None = Field(description="Original graph node name.")
    depends_on: list[str] = Field(description="Dependency task IDs.")


class WorkflowValidateResponse(BaseModel):
    ok: bool = Field(description="Whether the validation succeeded.")
    count: int = Field(description="Number of validated tasks.")
    tasks: list[WorkflowValidateTaskEntry] = Field(
        description="Validated task entries."
    )
    inspection: InspectionReport | None = Field(
        default=None,
        description="Compiled v2 template/plan inspection (v2 submissions only).",
    )
