"""An in-memory control store behind the real credential vault, for runtime tests."""

import fnmatch
from typing import Any, cast

from server.services.credential_vault import CredentialVault


class InMemoryVaultRedis:
    """The control-store surface the vault uses, with per-key expiries."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.expiring: set[str] = set()
        self.asyncio = _Async(self)
        self.sync = _Sync(self)


class _Async:
    def __init__(self, redis: InMemoryVaultRedis) -> None:
        self._redis = redis

    async def hash_set_persistent(self, key: str, mapping: dict[str, str]) -> None:
        self._redis.hashes.setdefault(key, {}).update(mapping)
        self._redis.expiring.discard(key)

    async def scan_keys(self, pattern: str) -> list[str]:
        return [k for k in self._redis.hashes if fnmatch.fnmatchcase(k, pattern)]

    async def persist(self, key: str) -> bool:
        self._redis.expiring.discard(key)
        return key in self._redis.hashes

    async def delete(self, key: str) -> None:
        self._redis.hashes.pop(key, None)
        self._redis.expiring.discard(key)


class _Sync:
    def __init__(self, redis: InMemoryVaultRedis) -> None:
        self._redis = redis

    def hash_mget(self, key: str, fields: list[str]) -> list[Any]:
        stored = self._redis.hashes.get(key, {})
        return [stored.get(field) for field in fields]

    def delete(self, key: str) -> None:
        self._redis.hashes.pop(key, None)
        self._redis.expiring.discard(key)


class InMemoryCredentialVault(CredentialVault):
    """The credential vault over an in-memory control store."""

    def __init__(self) -> None:
        self.redis = InMemoryVaultRedis()
        super().__init__(cast(Any, self.redis))
