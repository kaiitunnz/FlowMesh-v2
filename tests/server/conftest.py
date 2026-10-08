from collections.abc import Iterator

import pytest

from server.supervisor.resource_manager import MachineEnv, ResourceManager
from tests.server import lock_contract

lock_contract.install()


@pytest.fixture(autouse=True)
def _host_without_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the supervisor a GPU-less host, so no test probes Docker for its GPUs."""
    host = object.__new__(ResourceManager)
    host._env = MachineEnv(
        cpu_count=16, gpu_families={}, available_gpus=set(), gpu_uuids={}
    )
    monkeypatch.setattr(ResourceManager, "_instance", host)


@pytest.fixture(autouse=True)
def _stop_durability_retries() -> Iterator[None]:
    """Stop the durability retry of every runtime a test built, so none fires in a
    later test."""
    before = set(lock_contract.runtimes())
    yield
    for runtime in set(lock_contract.runtimes()) - before:
        runtime._durability.stop()


@pytest.fixture(autouse=True)
def _runtime_lock_contract() -> Iterator[None]:
    """Fail a test in which runtime source breaks the runtime's lock contract."""
    lock_contract.take_trips()
    yield
    breaches = sorted(
        {
            f"{trip.kind} {trip.name} <- {trip.site}"
            for trip in lock_contract.take_trips()
            if trip.from_source
        }
    )
    assert not breaches, "lock contract broken:\n" + "\n".join(breaches)
