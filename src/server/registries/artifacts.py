"""The durable catalog of each task's published artifacts.

A task's artifacts live in one hash, a field per revision and name holding the binding.
Fields are only ever added — by the transition that accepts their publication, in the
same transaction as the task's record — and are never rewritten or removed while the
task's state is kept.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.client import Pipeline

from shared.tasks.artifact_binding import (
    ArtifactBinding,
    ArtifactBindings,
)

from ..clients.redis import RedisClient, task_artifacts_key

_SEPARATOR = "\0"
_logger = logging.getLogger(__name__)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def artifact_field(revision: str, name: str) -> str:
    return f"{revision}{_SEPARATOR}{name}"


def queue_artifact_bindings(
    pipe: Pipeline | AsyncPipeline, task_id: str, bindings: Sequence[ArtifactBinding]
) -> None:
    """Queue the writes adding ``bindings`` to a task's catalog."""
    if bindings:
        pipe.hset(
            task_artifacts_key(task_id),
            mapping={
                artifact_field(
                    binding.revision, binding.name
                ): binding.model_dump_json()
                for binding in bindings
            },
        )


def decode_artifact_binding(
    task_id: str, field: Any, raw: Any
) -> ArtifactBinding | None:
    """Decode one stored binding of a task's catalog, or None when it does not
    parse."""
    try:
        return ArtifactBinding.model_validate_json(raw)
    except ValidationError:
        _logger.error("Task %s has an unreadable artifact binding %r", task_id, field)
        return None


def decode_artifact_bindings(
    task_id: str, raw: Mapping[Any, Any], revision: str | None = None
) -> ArtifactBindings:
    """Decode a task's catalog from its stored hash, or only ``revision`` of it,
    skipping a field that does not parse."""
    bindings: ArtifactBindings = {}
    prefix = None if revision is None else artifact_field(revision, "")
    for field, value in raw.items():
        if prefix is not None and not _text(field).startswith(prefix):
            continue
        if (binding := decode_artifact_binding(task_id, field, value)) is not None:
            bindings[binding.key] = binding
    return bindings


def _keyed(
    binding: ArtifactBinding | None, revision: str, name: str
) -> ArtifactBinding | None:
    return binding if binding is not None and binding.key == (revision, name) else None


class ArtifactCatalogStore:
    """Reads of the artifact catalog."""

    _rds: RedisClient

    def artifact_bindings(
        self, task_id: str, revision: str | None = None
    ) -> ArtifactBindings:
        """Read a task's catalog, or only ``revision`` of it."""
        return decode_artifact_bindings(
            task_id, self._rds.sync.hash_getall(task_artifacts_key(task_id)), revision
        )

    def artifact_binding(
        self, task_id: str, revision: str, name: str
    ) -> ArtifactBinding | None:
        """Read one binding of a task's catalog, or None when it holds none."""
        field = artifact_field(revision, name)
        (raw,) = self._rds.sync.hash_mget(task_artifacts_key(task_id), [field])
        if raw is None:
            return None
        return _keyed(decode_artifact_binding(task_id, field, raw), revision, name)

    async def artifact_binding_async(
        self, task_id: str, revision: str, name: str
    ) -> ArtifactBinding | None:
        field = artifact_field(revision, name)
        (raw,) = await self._rds.asyncio.hash_mget(task_artifacts_key(task_id), [field])
        if raw is None:
            return None
        return _keyed(decode_artifact_binding(task_id, field, raw), revision, name)

    async def artifact_bindings_async(
        self, task_id: str, revision: str | None = None
    ) -> ArtifactBindings:
        return decode_artifact_bindings(
            task_id,
            await self._rds.asyncio.hash_getall(task_artifacts_key(task_id)),
            revision,
        )

    def artifact_bindings_many(
        self, task_ids: Sequence[str], revision: str | None = None
    ) -> dict[str, ArtifactBindings]:
        """Read each task's catalog, or only ``revision`` of it, in one round trip."""
        if not task_ids:
            return {}
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            for task_id in task_ids:
                pipe.hgetall(task_artifacts_key(task_id))
            replies = pipe.execute()
        return {
            task_id: decode_artifact_bindings(task_id, raw, revision)
            for task_id, raw in zip(task_ids, replies, strict=True)
        }

    async def artifact_bindings_many_async(
        self, task_ids: Sequence[str], revision: str | None = None
    ) -> dict[str, ArtifactBindings]:
        if not task_ids:
            return {}
        async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
            for task_id in task_ids:
                pipe.hgetall(task_artifacts_key(task_id))
            replies = await pipe.execute()
        return {
            task_id: decode_artifact_bindings(task_id, raw, revision)
            for task_id, raw in zip(task_ids, replies, strict=True)
        }


__all__ = [
    "ArtifactCatalogStore",
    "artifact_field",
    "decode_artifact_binding",
    "decode_artifact_bindings",
    "queue_artifact_bindings",
]
