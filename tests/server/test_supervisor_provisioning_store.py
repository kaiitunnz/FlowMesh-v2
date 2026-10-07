from typing import Any, cast

import fakeredis
import pytest
from pydantic import SecretStr

from server.clients import redis as redis_clients
from server.config import IdentityConfig, RedisConfig
from server.supervisor.adapters.docker import DockerWorkerConfig
from server.supervisor.provisioning import (
    DockerHandle,
    ProviderHandle,
    VastHandle,
    WorkerProvisioningStore,
    WorkerRecord,
    recorded_config,
)
from tests.server.supervisor_helpers import worker_record


def _record(
    alias: str, token: str = "tok", handle: ProviderHandle | None = None
) -> WorkerRecord:
    return worker_record(
        alias,
        token=token,
        handle=handle or DockerHandle(container_id="c1", container_name=alias),
    )


@pytest.mark.parametrize(
    "handle",
    [
        DockerHandle(container_id="c1", container_name="w1"),
        VastHandle(instance_id=7, created_instance=True),
        None,
    ],
)
def test_records_round_trip_with_their_token_and_never_print_it(
    handle: ProviderHandle | None,
) -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    store = WorkerProvisioningStore(client, IdentityConfig())
    assert store.create(
        _record("w1", "secret-token").model_copy(update={"handle": handle})
    )

    [loaded] = store.load()
    assert loaded.token.get_secret_value() == "secret-token"
    assert loaded.handle == handle
    assert "secret-token" not in repr(loaded)


def test_an_unreadable_record_is_skipped() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    store = WorkerProvisioningStore(client, IdentityConfig())
    assert store.create(_record("w1"))
    [key] = cast(list[str], client.keys())
    client.hset(key, "w2", "{not json")
    client.hset(key, "w3", '{"alias": "w3"}')
    assert [r.alias for r in store.load()] == ["w1"]


def test_create_refuses_an_alias_that_has_a_record() -> None:
    store = WorkerProvisioningStore(
        fakeredis.FakeRedis(decode_responses=True), IdentityConfig()
    )
    assert store.create(_record("w1", "first"))
    assert not store.create(_record("w1", "second"))
    [loaded] = store.load()
    assert loaded.token.get_secret_value() == "first"


def test_identities_sharing_one_endpoint_keep_their_own_records() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    identities = [
        IdentityConfig(namespace="ns", cluster="c", alias="node"),
        IdentityConfig(namespace="ns", cluster="c", alias="other"),
        IdentityConfig(namespace="ns", cluster="d", alias="node"),
        # Joined without encoding, these two would share one key.
        IdentityConfig(namespace="ns", cluster="c:x", alias="node"),
        IdentityConfig(namespace="ns:c", cluster="x", alias="node"),
    ]
    stores = [WorkerProvisioningStore(client, identity) for identity in identities]
    for i, store in enumerate(stores):
        assert store.create(_record("w1", f"tok-{i}"))

    for i, store in enumerate(stores):
        assert [r.token.get_secret_value() for r in store.load()] == [f"tok-{i}"]
    stores[0].delete("w1")
    assert stores[0].load() == [] and len(stores[1].load()) == 1


def test_recorded_config_leaves_out_secret_fields() -> None:
    config = DockerWorkerConfig(worker_alias="w1", hf_token=SecretStr("hf-secret"))
    recorded = recorded_config(config)
    assert "hf_token" not in recorded and "nebula_api_token" not in recorded
    assert "hf-secret" not in str(recorded)
    assert DockerWorkerConfig.model_validate(recorded).worker_alias == "w1"


def _urls(monkeypatch, module: Any = redis_clients.redis) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def from_url(url: str, **kwargs) -> object:
        calls.append((url, kwargs))
        return object()

    monkeypatch.setattr(module, "from_url", from_url)
    return calls


def test_the_default_store_is_the_control_redis_with_its_auth_and_tls(
    monkeypatch,
) -> None:
    calls = _urls(monkeypatch)
    redis_clients.supervisor_state_sync_client(
        RedisConfig(
            control_url="rediss://control:6379/0",
            acl_enabled=True,
            username="admin",
            password="pw",
            tls_ca_file="/ca.pem",
        )
    )
    [(url, kwargs)] = calls
    assert url == "rediss://admin:pw@control:6379/0"
    assert kwargs["ssl_ca_certs"] == "/ca.pem"
    assert kwargs["socket_timeout"] and kwargs["socket_connect_timeout"]


def test_an_operator_store_connects_with_only_its_own_url(monkeypatch) -> None:
    calls = _urls(monkeypatch)
    redis_clients.supervisor_state_sync_client(
        RedisConfig(
            control_url="redis://control:6379/0",
            supervisor_state_url="rediss://op:oppw@state:6390/2",
            acl_enabled=True,
            username="admin",
            password="pw",
            tls_ca_file="/ca.pem",
        )
    )
    [(url, kwargs)] = calls
    assert url == "rediss://op:oppw@state:6390/2"
    assert "ssl_ca_certs" not in kwargs and "connection_class" not in kwargs
    assert kwargs["socket_timeout"] and kwargs["socket_connect_timeout"]


@pytest.mark.parametrize(
    "cfg",
    [
        RedisConfig(
            control_url="rediss://control:6379/0",
            acl_enabled=True,
            username="admin",
            password="pw",
            tls_ca_file="/ca.pem",
        ),
        RedisConfig(
            control_url="redis://control:6379/0",
            supervisor_state_url="rediss://op:oppw@state:6390/2",
        ),
    ],
    ids=["control", "operator"],
)
def test_the_async_and_blocking_store_clients_connect_alike(
    monkeypatch, cfg: RedisConfig
) -> None:
    sync_calls = _urls(monkeypatch)
    async_calls = _urls(monkeypatch, redis_clients.async_redis)
    redis_clients.supervisor_state_sync_client(cfg)
    redis_clients.supervisor_state_client(cfg)

    [(sync_url, sync_kwargs)] = sync_calls
    [(async_url, async_kwargs)] = async_calls
    assert sync_url == async_url
    sync_class = sync_kwargs.pop("connection_class", None)
    async_class = async_kwargs.pop("connection_class", None)
    assert sync_kwargs == async_kwargs
    assert (sync_class, async_class) in (
        (None, None),
        (redis_clients.SyncSSLConnection, redis_clients.AsyncSSLConnection),
    )
