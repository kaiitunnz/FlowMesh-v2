import logging

from pydantic import SecretStr

from ..clients.redis import RedisClient, workflow_model_secret_key


class ModelSecretVault:
    """Durable, workflow-scoped store for user-supplied model credentials.

    A workflow's inline ``api_key`` is vaulted at submission under its owning
    workflow's namespace and referenced everywhere else by an opaque generated ref.
    ``resolve`` reads a secret only within its owning workflow, so a ref minted for one
    workflow never yields another's credential. A read refreshes a sliding TTL, so an
    active workflow keeps its key while an abandoned, idle submission expires; the
    primary purge is explicit, on the workflow's terminal transition.

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
        self._logger = logger or logging.getLogger("model-secret-vault")

    async def store(self, workflow_id: str, ref: str, secret: SecretStr) -> None:
        """Vault one workflow-scoped credential under its generated ref."""
        key = workflow_model_secret_key(workflow_id)
        await self._redis.asyncio.hash_set(key, {ref: secret.get_secret_value()})
        await self._redis.asyncio.expire(key, self._ttl_sec)

    def resolve(self, workflow_id: str, ref: str | None) -> SecretStr | None:
        """The credential for ``ref`` within ``workflow_id``, refreshing the TTL."""
        if not ref:
            return None
        key = workflow_model_secret_key(workflow_id)
        values = self._redis.sync.hash_mget(key, [ref])
        value = values[0] if values else None
        if value is None:
            return None
        self._redis.sync.expire(key, self._ttl_sec)
        return SecretStr(value if isinstance(value, str) else value.decode())

    def purge(self, workflow_id: str) -> None:
        """Drop every vaulted credential for a workflow at its terminal transition."""
        self._redis.sync.delete(workflow_model_secret_key(workflow_id))
