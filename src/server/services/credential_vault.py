import json
from collections.abc import Collection, Mapping
from typing import Any

from pydantic import SecretStr

from ..clients.redis import (
    WORKFLOW_CREDENTIAL_KEY_PATTERN,
    RedisClient,
    credential_key_workflow_id,
    workflow_credential_key,
)


def _model_key(value: str) -> str:
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    return decoded if isinstance(decoded, str) else value


class CredentialVault:
    """Durable, workflow-scoped store for the credentials a submission carries inline.

    Each credential is vaulted at submission under its owning workflow's namespace and
    referenced everywhere else by an opaque generated ref. A read resolves a ref only
    within its owning workflow, so a ref minted for one workflow never yields another's
    credential. Each value is stored JSON-encoded; an agent's model key is read with
    ``resolve``, which also reads a key stored as its raw string, and a task-spec value
    with ``resolve_values``.

    A workflow's credentials live until it settles: the terminal transition or a cancel
    purges them, and a restart keeps every live workflow's credentials while dropping
    those of a workflow that settled or never registered.

    Values are stored structured (one Redis hash field per ref) within the Redis
    control store's trust boundary. No credential is encrypted at rest here; the store
    keeps the credential out of every readable surface and scopes it against
    cross-workflow resolution.
    """

    def __init__(self, redis: RedisClient) -> None:
        self._redis = redis

    async def store_values(self, workflow_id: str, values: Mapping[str, Any]) -> None:
        """Vault credential values, keyed by their generated refs."""
        if values:
            await self._redis.asyncio.hash_set_persistent(
                workflow_credential_key(workflow_id),
                {ref: json.dumps(value) for ref, value in values.items()},
            )

    def resolve(self, workflow_id: str, ref: str | None) -> SecretStr | None:
        """The model key for ``ref`` within ``workflow_id``."""
        if not ref:
            return None
        value = self._read(workflow_id, [ref])[0]
        return None if value is None else SecretStr(_model_key(value))

    def resolve_values(self, workflow_id: str, refs: Collection[str]) -> dict[str, Any]:
        """The task-spec values ``refs`` name within ``workflow_id``; a ref that no
        longer resolves is absent from the result."""
        ordered = list(dict.fromkeys(refs))
        if not ordered:
            return {}
        values = self._read(workflow_id, ordered)
        return {
            ref: json.loads(value)
            for ref, value in zip(ordered, values, strict=True)
            if value is not None
        }

    def _read(self, workflow_id: str, refs: list[str]) -> list[str | None]:
        values = self._redis.sync.hash_mget(workflow_credential_key(workflow_id), refs)
        return [
            value if value is None or isinstance(value, str) else value.decode()
            for value in values
        ]

    def purge(self, workflow_id: str) -> None:
        """Drop every vaulted credential for a workflow at its terminal transition."""
        self._redis.sync.delete(workflow_credential_key(workflow_id))

    async def retain_only(self, live_workflow_ids: Collection[str]) -> None:
        """Keep the credentials of ``live_workflow_ids`` without expiry and drop every
        other workflow's."""
        live = set(live_workflow_ids)
        for key in await self._redis.asyncio.scan_keys(WORKFLOW_CREDENTIAL_KEY_PATTERN):
            if credential_key_workflow_id(key) in live:
                await self._redis.asyncio.persist(key)
            else:
                await self._redis.asyncio.delete(key)
