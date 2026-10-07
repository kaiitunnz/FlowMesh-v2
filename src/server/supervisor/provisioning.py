"""Durable records of the workers a supervisor provisions."""

import asyncio
import json
import logging
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Annotated, Any, Literal, cast, get_args
from urllib.parse import quote

import redis
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from ..config import IdentityConfig

logger = logging.getLogger("supervisor")

_STORE_RETRY_MAX_SEC = 60.0


class RunState(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"


class RecordState(StrEnum):
    PROVISIONING = "provisioning"
    """A launch is in progress and has committed no handle yet."""
    PRESENT = "present"
    REMOVING = "removing"
    """The worker is being destroyed; its handle is removed before the record."""


class Due(StrEnum):
    """What the heartbeat owes a record."""

    REMOVE = "remove"
    """Remove what a removing worker's record names, then the record."""
    EXPIRE = "expire"
    """Remove a restored worker that did not register again within its grace."""
    FINISH_STOP = "finish_stop"
    """Stop a stopped worker whose record still names what it ran on."""


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


class ProvisionedWorkers:
    """The records of a node's provisioned workers and what their recovery still
    owes, by alias."""

    # Called on the supervisor's event loop, except the handle commit from a launch
    # thread; ``_lock`` serializes every record write.

    def __init__(self, store: WorkerProvisioningStore) -> None:
        self._store = store
        # A launching thread commits a handle while the loop writes the rest.
        self._records: dict[str, WorkerRecord] = {}
        self._lock = threading.RLock()
        # Aliases whose latest record has not reached the store.
        self._unsaved: set[str] = set()
        # Restored workers expected to register again within the grace.
        self._awaiting: set[str] = set()
        self._grace_deadline: float | None = None
        # Lifecycle operations under way; the heartbeat leaves their aliases alone.
        self._in_flight: Counter[str] = Counter()

    async def load(self) -> list[WorkerRecord]:
        """Read every record, retrying until the store answers."""
        delay = 1.0
        while True:
            try:
                records = await asyncio.to_thread(self._store.load)
            except Exception as exc:
                logger.error(
                    "Failed to read the supervisor state store, retrying in %.0fs: %s",
                    delay,
                    exc,
                )
            else:
                with self._lock:
                    self._records = {record.alias: record for record in records}
                return records
            await asyncio.sleep(delay)
            delay = min(delay * 2, _STORE_RETRY_MAX_SEC)

    def __contains__(self, alias: str) -> bool:
        return alias in self._records

    def get(self, alias: str) -> WorkerRecord | None:
        return self._records.get(alias)

    def worker_ids(self) -> list[str]:
        """Return the worker ids the recorded workers last registered under."""
        return [r.worker_id for r in self._records.values() if r.worker_id]

    def create(self, record: WorkerRecord) -> None:
        """Store a new record; raise when its alias already has one."""
        with self._lock:
            if record.alias in self._records or not self._store.create(record):
                raise ValueError(f"Worker '{record.alias}' already exists")
            self._records[record.alias] = record

    def update(self, alias: str, strict: bool = True, **changes: Any) -> None:
        """Change ``alias``'s record. When the store write fails, a strict update raises
        and changes nothing; any other keeps the change for ``retry_unsaved``.

        A removing record stays removing until it is forgotten; a change of state is
        dropped from it.
        """
        with self._lock:
            record = self._records.get(alias)
            if record is None:
                return
            if record.state is RecordState.REMOVING:
                changes.pop("state", None)
            updated = record.model_copy(update=changes)
            try:
                self._store.put(updated)
            except Exception:
                if strict:
                    raise
                logger.exception("Failed to save the record of worker %s", alias)
                self._unsaved.add(alias)
            else:
                self._unsaved.discard(alias)
            self._records[alias] = updated

    def handle_committer(self, alias: str) -> Callable[[ProviderHandle | None], None]:
        """Return the callback a worker's adapter reports what it launched to."""

        def commit(handle: ProviderHandle | None) -> None:
            state = {"state": RecordState.PRESENT} if handle is not None else {}
            self.update(alias, strict=False, handle=handle, **state)

        return commit

    def launch_ended(self, alias: str, handle: ProviderHandle | None) -> None:
        """Record that a launch which returned or failed left ``handle``."""
        if handle is None:
            self.update(alias, strict=False, state=RecordState.PRESENT)

    def commit_worker_id(self, alias: str, worker_id: str) -> None:
        """Record the id a worker registered under; raise when the store write
        fails."""
        self.update(alias, worker_id=worker_id)

    def forget(self, alias: str) -> None:
        """Delete the record of a worker that is gone; keep it when the store write
        fails."""
        self._awaiting.discard(alias)
        with self._lock:
            try:
                self._store.delete(alias)
            except Exception as exc:
                logger.warning(
                    "Failed to delete the record of worker %s: %s", alias, exc
                )
                return
            self._records.pop(alias, None)
            self._unsaved.discard(alias)

    def expect(self, alias: str) -> None:
        """Give a restored worker the grace to register again."""
        self._awaiting.add(alias)

    def end_grace(self, alias: str) -> None:
        self._awaiting.discard(alias)

    def awaiting(self, alias: str) -> bool:
        return alias in self._awaiting

    def open_grace(self, until: float) -> None:
        """Expire restored workers that have not registered by ``until``."""
        self._grace_deadline = until

    @contextmanager
    def operating(self, alias: str) -> Iterator[None]:
        """Mark an operator's lifecycle operation on ``alias``, which ends a restored
        worker's grace and keeps the heartbeat off it."""
        self._awaiting.discard(alias)
        self.claim(alias)
        try:
            yield
        finally:
            self.settled(alias)

    def retry_unsaved(self) -> None:
        """Write again the records a lenient update could not save."""
        with self._lock:
            unsaved = list(self._unsaved)
        for alias in unsaved:
            self.update(alias, strict=False)

    def due(self, now: float) -> list[tuple[str, Due]]:
        """Return what each record no operation is under way on is owed."""
        deadline = self._grace_deadline
        expired = deadline is not None and now >= deadline
        due: list[tuple[str, Due]] = []
        for alias, record in list(self._records.items()):
            if alias in self._in_flight:
                continue
            if record.state is RecordState.REMOVING:
                due.append((alias, Due.REMOVE))
            elif (
                alias in self._awaiting
                and expired
                and record.run_state is RunState.RUNNING
            ):
                due.append((alias, Due.EXPIRE))
            elif record.handle is not None and record.run_state is RunState.STOPPED:
                due.append((alias, Due.FINISH_STOP))
        return due

    def claim(self, alias: str) -> None:
        """Keep the heartbeat off ``alias`` until ``settled``."""
        self._in_flight[alias] += 1

    def settled(self, alias: str) -> None:
        """End one operation or claim on ``alias``."""
        self._in_flight[alias] -= 1
        if self._in_flight[alias] <= 0:
            del self._in_flight[alias]


def _encode(record: WorkerRecord) -> str:
    data = record.model_dump(mode="json")
    data["token"] = record.token.get_secret_value()
    return json.dumps(data, separators=(",", ":"))
