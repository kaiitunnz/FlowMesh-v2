"""A worker's environment carries only the credentials the worker reads."""

import pytest

from server import env
from tests.server.test_supervisor_telemetry_env import _docker_worker, _vastai_worker


@pytest.mark.parametrize("make_worker", [_docker_worker, _vastai_worker])
@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "GOOGLE_API_KEY"])
def test_worker_env_omits_unread_provider_keys(make_worker, name: str) -> None:
    assert name not in make_worker()._base_environment()


@pytest.mark.parametrize("make_worker", [_docker_worker, _vastai_worker])
def test_worker_env_carries_the_peer_connect_budget(
    make_worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env, "NETWORK_PLANE_CONNECT_BUDGET_SEC", 2.5)

    environment = make_worker()._base_environment()

    assert float(environment["NETWORK_PLANE_CONNECT_BUDGET_SEC"]) == 2.5
