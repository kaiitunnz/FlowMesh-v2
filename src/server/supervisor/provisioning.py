"""Durable records of the workers a supervisor provisions."""

import json
import logging
from enum import StrEnum
from typing import Annotated, Any, Literal, cast, get_args
from urllib.parse import quote

import redis
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from ..config import IdentityConfig

logger = logging.getLogger("supervisor")


class RunState(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"


class RecordState(StrEnum):
    PROVISIONING = "provisioning"
    """A launch is in progress and has committed no handle yet."""
    PRESENT = "present"
    REMOVING = "removing"
    """The worker is being destroyed; its handle is removed before the record."""


class Removal(StrEnum):
    REMOVED = "removed"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class DockerHandle(BaseModel):
    """The container a Docker worker runs in."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["docker"] = "docker"
    container_id: str
    container_name: str


class VastHandle(BaseModel):
    """The Vast.ai instance a worker runs on."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["vastai"] = "vastai"
    instance_id: int
    created_instance: bool
    """Whether the instance was rented for the worker: removal destroys a rented
    instance and stops a supplied one."""


ProviderHandle = Annotated[DockerHandle | VastHandle, Field(discriminator="provider")]


class WorkerRecord(BaseModel):
    alias: str
    provider: str
    config: dict[str, Any]
    """The worker's launch settings, without its secret fields."""
    token: SecretStr
    run_state: RunState
    state: RecordState = RecordState.PRESENT
    handle: ProviderHandle | None = None
    worker_id: str | None = None


def recorded_config(config: BaseModel) -> dict[str, Any]:
    """Return ``config``'s settings without its secret fields, which a rebuilt config
    reads from the environment again."""
    secret = {
        name
        for name, field in type(config).model_fields.items()
        if field.annotation is SecretStr or SecretStr in get_args(field.annotation)
    }
    return config.model_dump(mode="json", exclude=secret)


class WorkerProvisioningStore:
    """The records of one node's provisioned workers, keyed by worker alias."""

    def __init__(self, client: redis.Redis, identity: IdentityConfig) -> None:
        self._redis = client
        scope = ":".join(
            quote(part, safe="")
            for part in (identity.namespace, identity.cluster, identity.alias)
        )
        self._key = f"supervisor-state:{scope}:workers"

    def load(self) -> list[WorkerRecord]:
        raw = cast(dict[str, str], self._redis.hgetall(self._key))
        records = []
        for alias, value in raw.items():
            try:
                records.append(WorkerRecord.model_validate_json(value))
            except ValidationError as exc:
                logger.error(
                    "Ignoring unreadable record of worker %s: %s", alias, type(exc)
                )
        return records

    def create(self, record: WorkerRecord) -> bool:
        """Store a new record; return False when its alias already has one."""
        return bool(self._redis.hsetnx(self._key, record.alias, _encode(record)))

    def put(self, record: WorkerRecord) -> None:
        self._redis.hset(self._key, record.alias, _encode(record))

    def delete(self, alias: str) -> None:
        self._redis.hdel(self._key, alias)


def _encode(record: WorkerRecord) -> str:
    data = record.model_dump(mode="json")
    data["token"] = record.token.get_secret_value()
    return json.dumps(data, separators=(",", ":"))
