"""Workflow-related models."""

from typing import Any

from pydantic import BaseModel

from .common import TaskStatus, WorkflowStatus


class WorkflowSubmitTaskEntry(BaseModel):
    task_id: str
    status: TaskStatus | None = None
    assigned_worker: str | None = None
    topic: str | None = None
    waiting_on: list[str] | None = None
    depends_on: list[str] | None = None
    attempts: int | None = None
    max_attempts: int | None = None
    load: int | None = None


class WorkflowSubmitResponse(BaseModel):
    ok: bool
    workflow_id: str
    count: int
    tasks: list[WorkflowSubmitTaskEntry]


class WorkflowValidateTaskEntry(BaseModel):
    task_id: str
    graph_node_name: str | None = None
    depends_on: list[str]


class SourceLocation(BaseModel):
    source_kind: str
    source_id: str
    detail: str | None = None


class Diagnostic(BaseModel):
    code: str
    message: str
    severity: str
    location: SourceLocation | None = None


class InspectionReport(BaseModel):
    workflow_id: str
    # The compiled logical template and physical plan are provisional internal
    # representations; kept as raw mappings rather than mirroring the full
    # operator tree into the published SDK.
    template: dict[str, Any]
    plan: dict[str, Any]
    diagnostics: list[Diagnostic] = []
    region_bearing: bool = False


class WorkflowValidateResponse(BaseModel):
    ok: bool
    count: int
    tasks: list[WorkflowValidateTaskEntry]
    inspection: InspectionReport | None = None


class Workflow(BaseModel):
    workflow_id: str
    task_ids: list[str]
    submitted_at: str
    updated_at: str
    status: WorkflowStatus
    dispatched_tasks: list[str]
    completed_tasks: list[str]
    failed_tasks: list[str]
    cancelled_tasks: list[str]
