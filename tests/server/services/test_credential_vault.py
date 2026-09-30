from typing import Any

import pytest

from server.clients.redis import workflow_credential_key
from tests.server.credential_vault_helpers import InMemoryCredentialVault


def _vault() -> tuple[InMemoryCredentialVault, Any]:
    vault = InMemoryCredentialVault()
    return vault, vault.redis


@pytest.mark.anyio
async def test_store_then_resolve_within_the_same_workflow():
    vault, _ = _vault()
    await vault.store_values("wfl-1", {"msk-a": "sk-user"})
    resolved = vault.resolve("wfl-1", "msk-a")
    assert resolved is not None and resolved.get_secret_value() == "sk-user"


@pytest.mark.anyio
async def test_a_ref_does_not_resolve_under_another_workflow():
    vault, _ = _vault()
    await vault.store_values("wfl-1", {"msk-a": "sk-user"})
    assert vault.resolve("wfl-2", "msk-a") is None


def test_missing_ref_and_none_resolve_to_none():
    vault, _ = _vault()
    assert vault.resolve("wfl-1", "msk-missing") is None
    assert vault.resolve("wfl-1", None) is None


@pytest.mark.anyio
async def test_purge_drops_the_workflow_credentials():
    vault, _ = _vault()
    await vault.store_values("wfl-1", {"msk-a": "sk-user"})
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


@pytest.mark.anyio
async def test_retain_only_keeps_live_vaults_without_expiry_and_drops_the_rest():
    vault, redis = _vault()
    for workflow_id in ("wfl-live", "wfl-settled", "wfl-unregistered"):
        await vault.store_values(workflow_id, {"msk-a": "sk"})
    redis.expiring.add(workflow_credential_key("wfl-live"))
    redis.hashes["workflow:wfl-live:other"] = {"x": "y"}

    await vault.retain_only(["wfl-live"])

    assert vault.resolve("wfl-live", "msk-a") is not None
    assert workflow_credential_key("wfl-live") not in redis.expiring
    assert vault.resolve("wfl-settled", "msk-a") is None
    assert vault.resolve("wfl-unregistered", "msk-a") is None
    assert "workflow:wfl-live:other" in redis.hashes


@pytest.mark.anyio
async def test_a_model_key_stored_as_its_raw_string_still_resolves():
    vault, redis = _vault()
    redis.hashes[workflow_credential_key("wfl-1")] = {"msk-a": "sk-raw", "msk-n": "123"}

    assert vault.resolve("wfl-1", "msk-a").get_secret_value() == "sk-raw"
    assert vault.resolve("wfl-1", "msk-n").get_secret_value() == "123"


@pytest.mark.anyio
async def test_a_write_clears_an_expiry_the_key_still_carries():
    vault, redis = _vault()
    key = workflow_credential_key("wfl-1")
    redis.hashes[key] = {"msk-old": '"sk"'}
    redis.expiring.add(key)

    await vault.store_values("wfl-1", {"msk-new": "Bearer sk"})

    assert key not in redis.expiring
