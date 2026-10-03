import pytest

from server.supervisor import resource_manager
from server.supervisor.adapters.external import ExternalWorkerFactory


def test_server_tests_never_probe_docker_for_host_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[str] = []

    def record() -> None:
        probes.append("docker")
        raise RuntimeError("no Docker in a unit test")

    monkeypatch.setattr(resource_manager, "get_docker_client", record)

    ExternalWorkerFactory(system_principal=None)  # type: ignore[arg-type]

    assert probes == []
