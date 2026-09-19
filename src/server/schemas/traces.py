"""Response shapes for the telemetry query surface.

A workflow's spans come back from the store flat; the tree endpoint returns them
assembled into the parent/child hierarchy, and the aggregate endpoint returns one
bucket per distinct value of the grouping attribute.

The ``logical`` and ``physical`` attribute views stay separate all the way out: a
consumer building the logical view of a workflow reads ``logical`` alone.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class TraceSpanNode(BaseModel):
    span_id: str = Field(description="Span identifier (hex).")
    parent_span_id: str | None = Field(
        default=None, description="Parent span identifier, absent at a root."
    )
    name: str = Field(description="Span name.")
    start_time: datetime = Field(description="Span start timestamp.")
    end_time: datetime = Field(description="Span end timestamp.")
    duration_seconds: float = Field(description="Span duration in seconds.")
    status: str = Field(description="Span status code.")
    service_name: str = Field(description="Service that emitted the span.")
    logical: dict[str, str] = Field(
        default_factory=dict, description="Logical-namespace attributes."
    )
    physical: dict[str, str] = Field(
        default_factory=dict, description="Physical-namespace attributes."
    )
    children: list["TraceSpanNode"] = Field(
        default_factory=list, description="Child spans, ordered by start time."
    )


class TraceTree(BaseModel):
    workflow_id: str = Field(description="Workflow identifier.")
    trace_id: str = Field(description="Trace identifier derived from the workflow id.")
    span_count: int = Field(description="Total number of spans in the trace.")
    total_duration_seconds: float = Field(
        description="Wall time from the earliest span start to the latest span end."
    )
    roots: list["TraceSpanNode"] = Field(
        default_factory=list, description="Root spans, ordered by start time."
    )


class TraceAggregateBucket(BaseModel):
    group_value: str = Field(description="Value of the grouping attribute.")
    stat: str = Field(description="Statistic applied to the metric's datapoints.")
    value: float = Field(description="Aggregated value.")
    sample_count: int = Field(description="Datapoints behind the aggregate.")


class TraceAggregate(BaseModel):
    metric: str = Field(description="Aggregated metric name.")
    group_by: str = Field(description="Attribute key the metric is grouped by.")
    stat: str = Field(description="Statistic applied to the metric's datapoints.")
    buckets: list[TraceAggregateBucket] = Field(
        default_factory=list, description="One bucket per distinct grouping value."
    )
