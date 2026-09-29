from typing import Any, cast

import pytest
from pydantic import SecretStr

from server.services.credential_vault import CredentialVault


class _FakePipe:
    def __init__(
        self, store: dict[str, dict[str, str]], expire_calls: list[tuple[str, int]]
    ) -> None:
        self._store = store
        self._expire_calls = expire_calls
        self._ops: list[tuple] = []

    def hset(self, key: str, mapping: dict[str, str]) -> "_FakePipe":
        self._ops.append(("hset", key, mapping))
        return self

    def expire(self, key: str, ttl_sec: int) -> "_FakePipe":
        self._ops.append(("expire", key, ttl_sec))
        return self

    async def execute(self) -> None:
        for op in self._ops:
            if op[0] == "hset":
                _, key, mapping = op
                self._store.setdefault(key, {}).update(mapping)
            else:
                _, key, ttl = op
                self._expire_calls.append((key, ttl))
        self._ops.clear()

    async def __aenter__(self) -> "_FakePipe":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeAsync:
    def __init__(
        self, store: dict[str, dict[str, str]], expire_calls: list[tuple[str, int]]
    ) -> None:
        self._store = store
        self._expire_calls = expire_calls

    def control_pipeline(self) -> _FakePipe:
        return _FakePipe(self._store, self._expire_calls)


class _FakeSync:
    def __init__(
        self, store: dict[str, dict[str, str]], expire_calls: list[tuple[str, int]]
    ) -> None:
        self._store = store
        self.expire_calls = expire_calls

    def hash_mget(self, key: str, fields: list[str]) -> list[Any]:
        h = self._store.get(key, {})
        return [h.get(f) for f in fields]

    def expire(self, key: str, ttl_sec: int) -> bool:
        self.expire_calls.append((key, ttl_sec))
        return key in self._store

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self.asyncio = _FakeAsync(self._store, self.expire_calls)
        self.sync = _FakeSync(self._store, self.expire_calls)


def _vault(ttl_sec: int = 100) -> tuple[CredentialVault, _FakeRedis]:
    redis = _FakeRedis()
    return CredentialVault(cast(Any, redis), ttl_sec), redis


@pytest.mark.anyio
async def test_store_then_resolve_within_the_same_workflow():
    vault, _ = _vault()
    await vault.store("wfl-1", "msk-a", SecretStr("sk-user"))
    resolved = vault.resolve("wfl-1", "msk-a")
    assert resolved is not None and resolved.get_secret_value() == "sk-user"


@pytest.mark.anyio
async def test_store_commits_the_ttl_atomically():
    vault, redis = _vault(ttl_sec=100)
    await vault.store("wfl-1", "msk-a", SecretStr("sk-user"))
    assert ("workflow:wfl-1:model_secret", 100) in redis.expire_calls


@pytest.mark.anyio
async def test_a_ref_does_not_resolve_under_another_workflow():
    vault, _ = _vault()
    await vault.store("wfl-1", "msk-a", SecretStr("sk-user"))
    assert vault.resolve("wfl-2", "msk-a") is None


def test_missing_ref_and_none_resolve_to_none():
    vault, _ = _vault()
    assert vault.resolve("wfl-1", "msk-missing") is None
    assert vault.resolve("wfl-1", None) is None


@pytest.mark.anyio
async def test_resolve_refreshes_the_sliding_ttl():
    vault, redis = _vault(ttl_sec=100)
    await vault.store("wfl-1", "msk-a", SecretStr("sk-user"))
    vault.resolve("wfl-1", "msk-a")
    assert ("workflow:wfl-1:model_secret", 100) in redis.expire_calls


@pytest.mark.anyio
async def test_purge_drops_the_workflow_credentials():
    vault, _ = _vault()
    await vault.store("wfl-1", "msk-a", SecretStr("sk-user"))
    vault.purge("wfl-1")
    assert vault.resolve("wfl-1", "msk-a") is None


@pytest.mark.anyio
async def test_task_spec_values_round_trip_with_their_json_types():
    vault, _ = _vault()
    await vault.store_values(
        "wfl-1", {"msk-h": "Bearer sk", "msk-k": ["ssh-ed25519 A"], "msk-d": {"a": 1}}
    )
    assert vault.resolve_values("wfl-1", ["msk-h", "msk-k", "msk-d"]) == {
        "msk-h": "Bearer sk",
        "msk-k": ["ssh-ed25519 A"],
        "msk-d": {"a": 1},
    }


@pytest.mark.anyio
async def test_values_resolve_only_within_their_workflow_and_omit_missing_refs():
    vault, _ = _vault()
    await vault.store_values("wfl-1", {"msk-h": "Bearer sk"})
    assert vault.resolve_values("wfl-2", ["msk-h"]) == {}
    assert vault.resolve_values("wfl-1", ["msk-h", "msk-gone"]) == {
        "msk-h": "Bearer sk"
    }
