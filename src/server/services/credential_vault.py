import json
import logging
from collections.abc import Collection, Mapping
from typing import Any

from pydantic import SecretStr

from ..clients.redis import RedisClient, workflow_credential_key


class CredentialVault:
    """Durable, workflow-scoped store for the credentials a submission carries inline.

    Each credential is vaulted at submission under its owning workflow's namespace and
    referenced everywhere else by an opaque generated ref. A read resolves a ref only
    within its owning workflow, so a ref minted for one workflow never yields another's
    credential. An agent's model key is stored as its raw string and read with
    ``resolve``; a task-spec value is stored JSON-encoded and read with
    ``resolve_values``. A read refreshes a sliding TTL, so an active workflow keeps its
    credentials while an abandoned, idle submission expires; the primary purge is
    explicit, on the workflow's terminal transition.

    Values are stored structured (one Redis hash field per ref) within the Redis
    control store's trust boundary. No credential is encrypted at rest here; the store
    keeps the credential out of every readable surface and scopes it against
    cross-workflow resolution.
    """

    def __init__(
        self, redis: RedisClient, ttl_sec: int, logger: logging.Logger | None = None
    ) -> None:
        self._redis = redis
        self._ttl_sec = max(1, ttl_sec)
        self._logger = logger or logging.getLogger("credential-vault")

    async def store(self, workflow_id: str, ref: str, secret: SecretStr) -> None:
        """Vault one workflow-scoped model key under its generated ref."""
        await self._store_fields(workflow_id, {ref: secret.get_secret_value()})

    async def store_values(self, workflow_id: str, values: Mapping[str, Any]) -> None:
        """Vault task-spec credential values, keyed by their generated refs."""
        if values:
            await self._store_fields(
                workflow_id, {ref: json.dumps(value) for ref, value in values.items()}
            )

    async def _store_fields(self, workflow_id: str, fields: dict[str, str]) -> None:
        # The write and its TTL commit as one transaction, so a crash never leaves a
        # credential without an expiry backstop.
        key = workflow_credential_key(workflow_id)
        async with self._redis.asyncio.control_pipeline() as pipe:
            pipe.hset(key, mapping=fields)
            pipe.expire(key, self._ttl_sec)
            await pipe.execute()

    def resolve(self, workflow_id: str, ref: str | None) -> SecretStr | None:
        """The model key for ``ref`` within ``workflow_id``, refreshing the TTL."""
        if not ref:
            return None
        value = self._read(workflow_id, [ref])[0]
        return None if value is None else SecretStr(value)

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
        key = workflow_credential_key(workflow_id)
        values = self._redis.sync.hash_mget(key, refs)
        if any(value is not None for value in values):
            self._redis.sync.expire(key, self._ttl_sec)
        return [
            value if value is None or isinstance(value, str) else value.decode()
            for value in values
        ]

    def purge(self, workflow_id: str) -> None:
        """Drop every vaulted credential for a workflow at its terminal transition."""
        self._redis.sync.delete(workflow_credential_key(workflow_id))
