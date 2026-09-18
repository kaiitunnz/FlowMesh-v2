from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.utils.json import normalize_numbers
from shared.utils.time import now_iso

from .worker import WorkerStatus


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(
        ..., description="Event type, expected to be an uppercase enum value."
    )
    ts: str = Field(default_factory=now_iso, description="Event timestamp (ISO8601).")
    traceparent: str | None = Field(
        default=None,
        description="W3C traceparent naming the trace this event belongs to.",
    )

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        value = (value or "").strip().upper()
        if not value:
            raise ValueError("event type must not be empty")
        return value


class TaskEvent(BaseEvent):
    worker_id: str | None = Field(
        default=None, description="Associated worker identifier."
    )
    task_id: str = Field(..., description="Associated task identifier.")
    status: str | None = Field(default=None, description="Task status.")
    error: str | None = Field(default=None, description="Error message if any.")
    retryable: bool | None = Field(
        default=None,
        description=(
            "Whether the failure may be retried on another worker. None to defer the "
            "decision to the server."
        ),
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Additional event payload."
    )

    @field_validator("task_id")
    @classmethod
    def _trim_task_id(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("task_id must not be empty")
        return value


class WorkerEvent(BaseEvent):
    worker_id: str = Field(..., description="Associated worker identifier.")
    status: WorkerStatus | None = Field(
        default=None, description="Worker status (IDLE/RUNNING/etc)."
    )
    tags: list[str] | None = Field(default=None, description="Worker tags.")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Metrics reported in heartbeat."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Additional context."
    )
    actor: dict[str, Any] | None = Field(default=None, description="Actor information")


class NodeEvent(BaseEvent):
    node_id: str = Field(..., description="Associated node identifier.")
    tags: list[str] | None = Field(default=None, description="Node tags.")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Additional context."
    )
    actor: dict[str, Any] | None = Field(default=None, description="Actor information")


Event = TaskEvent | WorkerEvent | NodeEvent


def parse_event(data: dict[str, Any]) -> Event:
    data = normalize_numbers(data)
    event_type = str(data.get("type", "")).upper()
    if event_type.startswith("TASK_"):
        return TaskEvent.model_validate(data)
    if event_type.startswith("SV_"):
        return NodeEvent.model_validate(data)
    return WorkerEvent.model_validate(data)


def serialize_event(event: Event) -> dict[str, Any]:
    """Serialize an event, omitting trace context it does not carry.

    An absent ``traceparent`` is dropped rather than travelling as an explicit null, so
    a deployment with telemetry off puts no extra bytes on any of the paths this
    crosses -- several of which re-serialize an event they relayed.
    """
    payload = event.model_dump(mode="python")
    if payload.get("traceparent") is None:
        payload.pop("traceparent", None)
    return payload


__all__ = [
    "BaseEvent",
    "Event",
    "NodeEvent",
    "TaskEvent",
    "WorkerEvent",
    "parse_event",
    "serialize_event",
]
