from enum import StrEnum

from pydantic import BaseModel, Field, SerializeAsAny

from shared.schemas.result import AnyExecutorResult


class OutputOutcome(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    EXPLICIT_EMPTY = "explicit_empty"
    DECLARED_FAILURE = "declared_failure"


class WorkflowOutputMember(BaseModel):
    name: str = Field(description="Name of the node the output is declared on.")
    cardinality: str = Field(description="singleton or keyed_collection.")
    value_type: str | None = Field(
        default=None, description="Declared type of the output's value."
    )
    scope: str | None = Field(
        default=None, description="Opaque scope a collection member belongs to."
    )
    key: str | None = Field(
        default=None, description="Collection member key (the child index)."
    )
    sequence: int | None = Field(default=None, description="Member sequence number.")
    outcome: OutputOutcome = Field(description="Where the member stands.")


class WorkflowOutputEntry(WorkflowOutputMember):
    cursor: str = Field(description="Opaque cursor of this member.")


class WorkflowOutputPage(BaseModel):
    entries: list[WorkflowOutputEntry] = Field(description="Members, in cursor order.")
    next_cursor: str | None = Field(
        default=None, description="Cursor of the last entry."
    )
    prev_cursor: str | None = Field(
        default=None, description="Cursor of the first entry."
    )
    open: bool = Field(
        description="Whether the workflow can publish more; a last page is final "
        "only when this is false."
    )


class WorkflowOutputValue(WorkflowOutputMember):
    value: SerializeAsAny[AnyExecutorResult] | None = Field(
        default=None, description="The member's value, when it settled with one."
    )
