"""A worker's environment carries only the credentials the worker reads."""

import pytest

from tests.server.test_supervisor_telemetry_env import _docker_worker, _vastai_worker


@pytest.mark.parametrize("make_worker", [_docker_worker, _vastai_worker])
@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "GOOGLE_API_KEY"])
def test_worker_env_omits_unread_provider_keys(make_worker, name: str) -> None:
    assert name not in make_worker()._base_environment()
