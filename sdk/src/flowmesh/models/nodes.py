"""Node-related models."""

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .workers import WorkerHardware


class NodeRole(StrEnum):
    ROOT = "root"
    WORKER = "worker"


class Node(BaseModel):
    id: str
    namespace: str
    cluster: str
    alias: str
    version: str | None = None
    started_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    last_seen: str | None = None
    max_gpu_count: int = 0
    current_gpu_count: int = 0
    network_endpoint: dict[str, Any] | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [tag.strip() for tag in value.split(",") if tag.strip()]
        return list(value)

    @field_validator("network_endpoint", mode="before")
    @classmethod
    def validate_network_endpoint(cls, value: Any) -> Any:
        # The server serializes the advertisement as a JSON string in its node hash.
        if isinstance(value, str):
            text = value.strip()
            return json.loads(text) if text else None
        return value


class NodeRegisterResponse(BaseModel):
    node_id: str


class WorkerRegisterResponse(BaseModel):
    worker_id: str


class NodeWorkerInfo(BaseModel):
    id: str | None = None
    name: str
    namespace: str
    cluster: str
    node_id: str
    node_alias: str
    provider: str
    version: str | None = None
    status: str
    hardware: WorkerHardware | None = None
