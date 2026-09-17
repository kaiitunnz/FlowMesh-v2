"""Trace response payload types as seen by the SDK.

``ProfileSummary`` and the models under it describe the wire shape returned by
``GET /traces/workflows/analyze/{workflow_id}``; ``TraceTree`` and ``TraceAggregate``
describe the span-tree and metric-aggregate queries over the telemetry store.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class _ProfileBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetSummary(_ProfileBase):
    asset_guid: str
    latest_data_id: str
    latest_version: int
    user_id: str
    versions: int
    created_at: str | None = None


class LineageEdge(_ProfileBase):
    data_id: str
    source_data_id: str
    created_at: str | None = None


class EventSummary(_ProfileBase):
    """Per-event-type duration aggregates as parallel lists."""

    event_type: list[str]
    count: list[int]
    total_seconds: list[float]
    avg_seconds: list[float]
    min_seconds: list[float]
    max_seconds: list[float]


class E2EBreakdown(_ProfileBase):
    hardware_summary: EventSummary
    network_summary: EventSummary
    workflow_duration_seconds: float
    total_network_seconds: float


class ActiveWaitBreakdown(_ProfileBase):
    data_id: list[str]
    active_seconds: list[float]
    wait_seconds: list[float]


class TaskTiming(_ProfileBase):
    data_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    queuing_delay_seconds: float
    parent_data_ids: list[str]
    blocking_parent_data_id: str | None = None


class CriticalPathSummary(_ProfileBase):
    path: list[str]
    critical_path_seconds: float
    active_wait_breakdown: ActiveWaitBreakdown
    hardware_summary: EventSummary
    network_summary: EventSummary
    total_network_seconds: float


class ProfileSummary(_ProfileBase):
    workflow_id: str | None = None
    event_count: int
    data_ids: list[str]
    assets: list[AssetSummary]
    lineage: list[LineageEdge]
    e2e_breakdown: E2EBreakdown
    per_data_id: list[TaskTiming]
    critical_path: CriticalPathSummary | None = None


class TraceSpanNode(BaseModel):
    """One span in a workflow's assembled trace, with its children nested under it."""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    parent_span_id: str | None = None
    name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    status: str
    logical: dict[str, str] = Field(default_factory=dict)
    physical: dict[str, str] = Field(default_factory=dict)
    children: list["TraceSpanNode"] = Field(default_factory=list)


class TraceTree(BaseModel):
    """A workflow's spans assembled into their parent/child hierarchy."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    trace_id: str
    span_count: int
    total_duration_seconds: float
    roots: list[TraceSpanNode] = Field(default_factory=list)


class TraceAggregateBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_value: str
    stat: str
    value: float
    sample_count: int


class TraceAggregate(BaseModel):
    """One telemetry metric aggregated into a bucket per grouping value."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    group_by: str
    stat: str
    workflow_id: str | None = None
    buckets: list[TraceAggregateBucket] = Field(default_factory=list)
