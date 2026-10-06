import fakeredis
import pytest

from server.clients import redis as redis_clients
from server.config import IdentityConfig, RedisConfig
from server.supervisor.adapters.docker import DockerWorkerConfig
from server.supervisor.provisioning import (
    ProviderHandle,
    RecordState,
    RunState,
    WorkerProvisioningStore,
    WorkerRecord,
    recorded_config,
)


def _record(alias: str, token: str = "tok") -> WorkerRecord:
    return WorkerRecord(
        alias=alias,
        provider="docker",
        config={"worker_alias": alias},
        token=token,  # type: ignore[arg-type]
        run_state=RunState.RUNNING,
        handle=ProviderHandle(container_id="c1", container_name=alias),
    )


def test_records_round_trip_with_their_token_and_never_print_it() -> None:
    store = WorkerProvisioningStore(
        fakeredis.FakeRedis(decode_responses=True), IdentityConfig()
    )
    assert store.create(_record("w1", "secret-token"))

    [loaded] = store.load()
    assert loaded.token.get_secret_value() == "secret-token"
    assert loaded.handle == ProviderHandle(container_id="c1", container_name="w1")
    assert loaded.state is RecordState.PROVISIONING
    assert "secret-token" not in repr(loaded)


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


def test_recorded_config_leaves_out_secret_fields(monkeypatch) -> None:
    config = DockerWorkerConfig(worker_alias="w1", hf_token="hf-secret")  # type: ignore[arg-type]
    recorded = recorded_config(config)
    assert "hf_token" not in recorded and "nebula_api_token" not in recorded
    assert "hf-secret" not in str(recorded)
    assert DockerWorkerConfig.model_validate(recorded).worker_alias == "w1"


def _urls(monkeypatch) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def from_url(url: str, **kwargs) -> object:
        calls.append((url, kwargs))
        return object()

    monkeypatch.setattr(redis_clients.redis, "from_url", from_url)
    return calls


def test_the_default_store_is_the_control_redis_with_its_auth_and_tls(
    monkeypatch,
) -> None:
    calls = _urls(monkeypatch)
    redis_clients.supervisor_state_client(
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


def test_an_operator_store_connects_with_only_its_own_url(monkeypatch) -> None:
    calls = _urls(monkeypatch)
    redis_clients.supervisor_state_client(
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


@pytest.mark.parametrize("value", ["", "  "])
def test_an_unset_store_url_means_the_control_redis(monkeypatch, value) -> None:
    monkeypatch.setenv("REDIS_SUPERVISOR_STATE_URL", value)
    assert RedisConfig.from_env().supervisor_state_url == ""
