import pytest

from server.supervisor.resource_manager import MachineEnv, ResourceManager


@pytest.fixture(autouse=True)
def _host_without_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the supervisor a GPU-less host, so no test probes Docker for its GPUs."""
    host = object.__new__(ResourceManager)
    host._env = MachineEnv(
        cpu_count=16, gpu_families={}, available_gpus=set(), gpu_uuids={}
    )
    monkeypatch.setattr(ResourceManager, "_instance", host)
