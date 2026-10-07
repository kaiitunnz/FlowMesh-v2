"""A supervisor keeps a record of each worker it provisions and takes its workers back
from those records when it starts again."""

import asyncio
import itertools
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from docker.errors import APIError, NotFound
from pydantic import SecretStr, ValidationError

from server.hooks import PrincipalContext
from server.supervisor import manager as manager_module
from server.supervisor.adapters import docker as docker_adapter
from server.supervisor.adapters import vastai as vastai_adapter
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.external import mint_external_token
from server.supervisor.manager import (
    WORKER_RECONNECT_GRACE_SEC,
    ServerWorkerConfig,
    WorkerInitConfig,
    WorkerManager,
)
from server.supervisor.provisioning import (
    DockerHandle,
    ProviderHandle,
    RecordState,
    Removal,
    RunState,
    VastHandle,
    WorkerRecord,
)
from server.supervisor.registry import WorkerRegistry
from server.supervisor.resource_manager import ResourceManager
from server.supervisor.schemas import WorkerHardware, WorkerStatus
from tests.server.supervisor_helpers import memory_store, worker_record
from tests.server.test_worker_manager_gpu import _resource_manager

_LOGGER = logging.getLogger("test.provisioned")
_PRINCIPAL = PrincipalContext(
    principal_id="system",
    org_id="org",
    external_id="system",
    principal_type="user",
    scopes=[],
)
_DEFAULTS = {"results_dir": "/results", "hf_cache_dir": "/hf", "enable_ssh": False}


class _Container:
    def __init__(self, daemon: "_Daemon", name: str, env: dict[str, str]) -> None:
        self.id = f"cid-{next(daemon.ids)}"
        self.name = name
        self.status = "running"
        self.attrs = {"Config": {"Env": [f"{k}={v}" for k, v in env.items()]}}
        self._daemon = daemon

    def reload(self) -> None:
        pass

    def exec_run(self, *_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(exit_code=0, output=b"")

    def stop(self, timeout: float | None = None) -> None:
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        if self._daemon.refuse_removal:
            raise APIError("removal refused")
        self._daemon.containers.pop(self.id, None)


class _Daemon:
    """A Docker daemon that keeps container names unique, as Docker does."""

    def __init__(self) -> None:
        self.ids = itertools.count()
        self.containers: dict[str, _Container] = {}
        self.runs = 0
        self.refuse_removal = False
        self.refuse_runs = False
        self.unreachable = False
        # When set, a launch waits for it; ``launching`` tells a test it has begun.
        self.gate: threading.Event | None = None
        self.launching = threading.Event()
        self.volumes = MagicMock()
        self.volumes.list.return_value = []
        self.networks = MagicMock()

    def get(self, key: str) -> _Container:
        if self.unreachable:
            raise APIError("daemon unreachable")
        for container in self.containers.values():
            if key in (container.id, container.name):
                return container
        raise NotFound(f"No such container: {key}")

    def run(self, **kwargs: Any) -> Any:
        if kwargs.get("remove"):
            return b""
        self.launching.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.refuse_runs:
            raise APIError("run refused")
        name = kwargs["name"]
        if any(c.name == name for c in self.containers.values()):
            raise APIError(f'The container name "/{name}" is already in use')
        self.runs += 1
        container = _Container(self, name, kwargs["environment"])
        self.containers[container.id] = container
        return container

    def list(self, all: bool = False, filters: dict | None = None) -> list[_Container]:
        prefix = (filters or {}).get("name")
        if not isinstance(prefix, str):
            return []
        return [c for c in self.containers.values() if c.name.startswith(prefix)]

    def add_foreign(self, name: str, token: str = "someone-else") -> _Container:
        container = _Container(self, name, {"WORKER_TOKEN": token})
        self.containers[container.id] = container
        return container


def _client(daemon: _Daemon) -> Any:
    return SimpleNamespace(
        containers=SimpleNamespace(get=daemon.get, run=daemon.run, list=daemon.list),
        volumes=daemon.volumes,
        networks=daemon.networks,
    )


class _Vast:
    """The VastAI client's calls, in the shapes the real client returns."""

    def __init__(self) -> None:
        self.created: list[int] = []
        self.live: set[int] = set()
        self.destroy_result: Any = None
        self.listing: Any = None
        self.on_show_instance: Any = None
        self.gate: threading.Event | None = None
        self.creating = threading.Event()

    def search_offers(self, **_: Any) -> list[dict[str, Any]]:
        return [{"id": 7, "gpu_name": "N/A"}]

    def create_instance(self, **_: Any) -> dict[str, Any]:
        self.creating.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        instance_id = 100 + len(self.created)
        self.created.append(instance_id)
        self.live.add(instance_id)
        return {"success": True, "new_contract": instance_id}

    def show_instance(self, id: int) -> dict[str, Any]:
        if self.on_show_instance is not None:
            self.on_show_instance(id)
        return {"id": id}

    def destroy_instance(self, id: int) -> Any:
        if self.destroy_result is None:
            self.live.discard(id)
        return self.destroy_result

    def show_instances(self) -> Any:
        if self.listing is not None:
            return self.listing
        return [{"id": i} for i in self.live]


class _Node:
    """One node's daemon, store and Vast account, across its supervisor's runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.monkeypatch = monkeypatch
        self.daemon = _Daemon()
        self.store = memory_store()
        self.vast = _Vast()
        self.config_path = tmp_path / "workers.yaml"
        self.vast_keys: list[str] = []
        monkeypatch.setattr(
            docker_adapter, "get_docker_client", lambda: _client(self.daemon)
        )

        def vast(api_key: str, **_: Any) -> _Vast:
            self.vast_keys.append(api_key)
            return self.vast

        monkeypatch.setattr(vastai_adapter, "VastAI", vast)
        self.clock = [1000.0]
        # The manager's clock only: the event loop's timers keep real time.
        monkeypatch.setattr(
            manager_module, "time", SimpleNamespace(monotonic=lambda: self.clock[0])
        )

    def write_config(self, *workers: dict[str, Any]) -> None:
        self.config_path.write_text(
            yaml.safe_dump(
                {"default_worker_config": _DEFAULTS, "workers": list(workers)}
            )
        )

    def supervisor(self, gpus: int = 4) -> WorkerManager:
        """A fresh supervisor run: new registry, new GPU pool, same node."""
        self.rm = _resource_manager(set(range(gpus)))
        self.monkeypatch.setattr(
            ResourceManager, "get_instance", classmethod(lambda cls: self.rm)
        )
        return WorkerManager(
            _PRINCIPAL,
            str(self.config_path),
            WorkerRegistry(),
            _LOGGER,
            self.store,
            vast_api_key=SecretStr("deployment-key"),
        )

    def records(self) -> dict[str, WorkerRecord]:
        return {r.alias: r for r in self.store.load()}

    def expire_grace(self) -> None:
        self.clock[0] += WORKER_RECONNECT_GRACE_SEC + 1


async def _no_sleep(_: float) -> None:
    return None


async def _run(wm: WorkerManager) -> list[str]:
    previous = await wm.restore()
    await wm.start()
    wm.grpc_ready()
    return previous


def _settle(wm: WorkerManager) -> Any:
    wm._settle_records()
    return asyncio.gather(*list(wm._tasks))


def _entry(alias: str, init_on_start: bool = True, **config: Any) -> dict[str, Any]:
    return {
        "provider": "docker",
        "init_on_start": init_on_start,
        "worker_config": {"worker_alias": alias, **config},
    }


def _gpu_request(**config: Any) -> WorkerInitConfig:
    return WorkerInitConfig(worker_config={"worker_type": "gpu", **config})


async def _restarted_removing(node: _Node, alias: str) -> WorkerManager:
    """Mark ``alias`` removing, as a destroy the supervisor crashed in, and restart."""
    record = node.records()[alias]
    node.store.put(record.model_copy(update={"state": RecordState.REMOVING}))
    restarted = node.supervisor()
    await _run(restarted)
    assert not restarted._registry.exists_by_alias(alias)
    return restarted


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Node:
    return _Node(monkeypatch, tmp_path)


# ---------------------------------------------------------------- restart ----


@pytest.mark.asyncio
async def test_a_restart_takes_back_every_provisioned_worker_without_relaunching(
    node: _Node,
) -> None:
    node.write_config(
        _entry("cfg-cpu"),
        _entry("cfg-gpu", worker_type="gpu", cuda_devices=[2]),
        {**_entry("pinned"), "worker_token": "pinned-token"},
    )
    first = node.supervisor()
    await _run(first)
    api = await first.create_worker(_gpu_request(gpu_count=1))
    for worker in first._registry.all_workers():
        first.commit_worker_id(worker, f"wkr-{worker.alias}")
    containers = {c.name: c.id for c in node.daemon.containers.values()}
    tokens = {w.alias: w.token for w in first._registry.all_workers()}
    assert node.daemon.runs == 4

    # The supervisor crashes: its process is gone, its containers run on.
    second = node.supervisor()
    previous = await _run(second)

    assert node.daemon.runs == 4
    assert {c.name: c.id for c in node.daemon.containers.values()} == containers
    restored = {w.alias: w for w in second._registry.all_workers()}
    assert {a: w.token for a, w in restored.items()} == tokens
    assert restored["pinned"].token == "pinned-token"
    assert all(w.holds_worker() for w in restored.values())
    assert sorted(previous) == sorted(f"wkr-{alias}" for alias in tokens)
    held = {2, *(restored[api.alias].get_info().held_gpus or [])}
    assert node.rm._env.available_gpus == {0, 1, 2, 3} - held


@pytest.mark.asyncio
async def test_a_survivors_gpus_are_held_before_any_worker_is_created(
    node: _Node,
) -> None:
    first = node.supervisor(gpus=2)
    await _run(first)
    await first.create_worker(_gpu_request(cuda_devices=[0]))

    node.write_config(_entry("later", worker_type="gpu", gpu_count=1))
    second = node.supervisor(gpus=2)
    await _run(second)

    assert second._registry.get_by_alias("later").get_info().held_gpus == [1]
    assert node.rm.available_gpu_count() == 0


@pytest.mark.asyncio
async def test_changed_or_removed_config_leaves_a_survivor_as_it_was_launched(
    node: _Node,
) -> None:
    node.write_config(_entry("kept", tags="old"), _entry("dropped"))
    await _run(node.supervisor())

    node.write_config(_entry("kept", tags="new"))
    second = node.supervisor()
    await _run(second)

    assert node.daemon.runs == 2
    assert second._registry.get_by_alias("kept").config.tags == "old"
    assert second._registry.exists_by_alias("dropped")
    assert set(node.records()) == {"kept", "dropped"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content", [yaml.safe_dump({"workers": [{"provider": "docker"}]}), "workers: ["]
)
async def test_an_invalid_config_still_restores_every_survivor(
    node: _Node, content: str
) -> None:
    node.write_config(_entry("w1"))
    await _run(node.supervisor())

    node.config_path.write_text(content)
    second = node.supervisor()
    await _run(second)

    assert second._registry.exists_by_alias("w1")
    assert node.daemon.runs == 1


@pytest.mark.asyncio
async def test_an_unreadable_store_holds_startup_until_it_reads(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    node.write_config(_entry("w1"))
    monkeypatch.setattr(manager_module.asyncio, "sleep", _no_sleep)
    load, failures = node.store.load, iter([ConnectionError("down")] * 2)

    def flaky_load() -> list[WorkerRecord]:
        if (error := next(failures, None)) is not None:
            assert node.daemon.runs == 0
            raise error
        return load()

    monkeypatch.setattr(node.store, "load", flaky_load)
    await _run(node.supervisor())

    assert node.daemon.runs == 1


# ----------------------------------------------------------- write points ----


@pytest.mark.asyncio
async def test_a_create_whose_record_cannot_be_written_launches_nothing(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    wm = node.supervisor()
    await _run(wm)
    monkeypatch.setattr(node.store, "create", MagicMock(side_effect=ConnectionError))

    with pytest.raises(ConnectionError):
        await wm.create_worker(_gpu_request(gpu_count=2))

    assert node.daemon.runs == 0
    assert node.rm.available_gpu_count() == 4
    assert wm._registry.all_workers() == []


@pytest.mark.asyncio
async def test_a_launch_commits_its_handle_and_a_failed_commit_is_saved_later(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(WorkerInitConfig())
    [container] = node.daemon.containers.values()
    assert node.records()[info.alias].handle == DockerHandle(
        container_id=container.id, container_name=container.name
    )

    await wm.stop_worker(info.alias)
    put = node.store.put
    failures = iter([ConnectionError("down")])

    def flaky_put(record: WorkerRecord) -> None:
        if record.handle is not None and next(failures, None) is not None:
            raise ConnectionError("down")
        put(record)

    monkeypatch.setattr(node.store, "put", flaky_put)
    assert await wm.start_worker(info.alias)
    assert node.records()[info.alias].handle is None

    await _settle(wm)

    [relaunched] = node.daemon.containers.values()
    assert node.records()[info.alias].handle == DockerHandle(
        container_id=relaunched.id, container_name=relaunched.name
    )
    assert node.daemon.runs == 2


@pytest.mark.asyncio
async def test_a_stop_is_recorded_before_the_worker_stops(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(WorkerInitConfig())
    monkeypatch.setattr(node.store, "put", MagicMock(side_effect=ConnectionError))

    with pytest.raises(ConnectionError):
        await wm.stop_worker(info.alias)

    [container] = node.daemon.containers.values()
    assert container.status == "running"


@pytest.mark.asyncio
async def test_a_stopped_worker_keeps_its_record_and_a_destroyed_one_loses_it(
    node: _Node,
) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(_gpu_request(gpu_count=1))

    await wm.stop_worker(info.alias)
    record = node.records()[info.alias]
    assert (record.run_state, record.handle) == (RunState.STOPPED, None)
    assert node.rm.available_gpu_count() == 3

    await wm.destroy_worker(info.alias)
    assert node.records() == {}
    assert node.rm.available_gpu_count() == 4


@pytest.mark.asyncio
async def test_a_relaunch_records_no_container_it_has_replaced(node: _Node) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(WorkerInitConfig())
    [container] = node.daemon.containers.values()
    container.status = "exited"
    wm._registry.get_by_alias(info.alias).set_status(WorkerStatus.STOPPED)
    node.daemon.refuse_runs = True

    assert not await wm.start_worker(info.alias)

    assert node.daemon.containers == {}
    record = node.records()[info.alias]
    assert (record.run_state, record.handle) == (RunState.RUNNING, None)


@pytest.mark.asyncio
async def test_a_shutdown_the_store_refuses_still_stops_the_manager(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    node.write_config(_entry("w1"), _entry("w2"))
    wm = node.supervisor()
    await _run(wm)
    monkeypatch.setattr(node.store, "put", MagicMock(side_effect=ConnectionError))

    await wm.stop()

    assert not wm.is_started
    assert len(node.daemon.containers) == 2
    assert set(node.records()) == {"w1", "w2"}


# ---------------------------------------------------------------- removal ----


@pytest.mark.asyncio
@pytest.mark.parametrize("restarted", [False, True])
async def test_an_unconfirmed_removal_keeps_the_record_and_gpus_until_it_is_confirmed(
    node: _Node, restarted: bool
) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(_gpu_request(cuda_devices=[1]))
    if restarted:
        wm = await _restarted_removing(node, info.alias)
        node.daemon.unreachable = True
    else:
        node.daemon.refuse_removal = True
        assert not await wm.destroy_worker(info.alias)
        assert not wm._registry.exists_by_alias(info.alias)

    await _settle(wm)
    assert node.records()[info.alias].state is RecordState.REMOVING
    assert 1 not in node.rm._env.available_gpus

    node.daemon.refuse_removal = node.daemon.unreachable = False
    await _settle(wm)
    assert node.records() == {} and node.daemon.containers == {}
    assert 1 in node.rm._env.available_gpus


@pytest.mark.asyncio
async def test_a_destroy_during_a_launch_stays_a_removal(node: _Node) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(WorkerInitConfig(init_on_start=False))
    node.daemon.gate = threading.Event()
    starting = asyncio.ensure_future(wm.start_worker(info.alias))
    await asyncio.to_thread(node.daemon.launching.wait, 5)
    node.daemon.refuse_removal = True
    destroying = asyncio.ensure_future(wm.destroy_worker(info.alias))
    await asyncio.sleep(0.05)

    node.daemon.gate.set()
    assert await starting
    assert not await destroying

    assert node.records()[info.alias].state is RecordState.REMOVING
    node.daemon.refuse_removal = False
    await _settle(wm)
    assert node.records() == {} and node.daemon.containers == {}


@pytest.mark.asyncio
async def test_one_removal_attempt_per_worker_is_in_flight(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    node.write_config(_entry("w1"))
    await _run(node.supervisor())
    wm = await _restarted_removing(node, "w1")
    calls = 0
    gate = asyncio.Event()
    loop = asyncio.get_running_loop()

    def remove(handle: ProviderHandle) -> Removal:
        nonlocal calls
        calls += 1
        asyncio.run_coroutine_threadsafe(gate.wait(), loop).result(5)
        return Removal.REMOVED

    monkeypatch.setattr(wm._providers["docker"].factory, "remove", remove)
    wm._settle_records()
    await asyncio.sleep(0.05)
    wm._settle_records()
    gate.set()
    await asyncio.gather(*list(wm._tasks))

    assert calls == 1


# ------------------------------------------------- grace and the heartbeat ----


@pytest.mark.asyncio
async def test_a_survivor_that_never_registers_is_removed_once_the_grace_ends(
    node: _Node,
) -> None:
    node.write_config(_entry("silent"), _entry("back"))
    await _run(node.supervisor())
    second = node.supervisor()
    await _run(second)
    second.worker_registered(second._registry.get_by_alias("back"), None)

    node.clock[0] += WORKER_RECONNECT_GRACE_SEC - 1
    await _settle(second)
    assert len(node.daemon.containers) == 2

    node.clock[0] += 1
    await _settle(second)

    assert [c.name for c in node.daemon.containers.values()] == ["back"]
    assert set(node.records()) == {"back"}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["restart", "recreate"])
async def test_an_operator_action_ends_a_restored_workers_grace(
    node: _Node, action: str
) -> None:
    node.write_config(_entry("w1"))
    await _run(node.supervisor())
    second = node.supervisor()
    await _run(second)
    if action == "restart":
        await second.stop_worker("w1")
        assert await second.start_worker("w1")
    else:
        await second.destroy_worker("w1")
        await second.create_worker(
            WorkerInitConfig(worker_config={**_DEFAULTS, "worker_alias": "w1"})
        )
    [launched] = node.daemon.containers.values()

    node.expire_grace()
    await _settle(second)

    assert node.daemon.containers == {launched.id: launched}
    assert node.records()["w1"].handle is not None


@pytest.mark.asyncio
async def test_the_grace_leaves_a_worker_the_operator_is_launching(
    node: _Node,
) -> None:
    node.write_config(_entry("w1"))
    await _run(node.supervisor())
    second = node.supervisor()
    await _run(second)
    await second.stop_worker("w1")
    node.daemon.gate = threading.Event()
    starting = asyncio.ensure_future(second.start_worker("w1"))
    await asyncio.to_thread(node.daemon.launching.wait, 5)

    node.expire_grace()
    await _settle(second)
    node.daemon.gate.set()
    assert await starting

    [launched] = node.daemon.containers.values()
    assert node.records()["w1"].handle == DockerHandle(
        container_id=launched.id, container_name="w1"
    )


@pytest.mark.asyncio
async def test_interrupted_stops_and_removals_finish_without_grace(
    node: _Node,
) -> None:
    node.write_config(_entry("stopping"), _entry("removing"))
    await _run(node.supervisor())
    for alias, changes in (
        ("stopping", {"run_state": RunState.STOPPED}),
        ("removing", {"state": RecordState.REMOVING}),
    ):
        node.store.put(node.records()[alias].model_copy(update=changes))

    second = node.supervisor()
    await _run(second)
    assert not second._registry.exists_by_alias("removing")
    await _settle(second)

    assert node.daemon.containers == {}
    record = node.records()["stopping"]
    assert (record.run_state, record.handle) == (RunState.STOPPED, None)
    assert set(node.records()) == {"stopping"}


# --------------------------------------------- interrupted Docker creation ----


def _launching(node: _Node, alias: str, token: str) -> None:
    node.store.create(
        worker_record(
            alias,
            token=token,
            config={**_DEFAULTS, "worker_alias": alias, "container_name": alias},
            state=RecordState.PROVISIONING,
        )
    )


@pytest.mark.asyncio
async def test_an_interrupted_create_finds_its_container_by_name_and_token(
    node: _Node,
) -> None:
    _launching(node, "w1", "tok-1")
    container = node.daemon.add_foreign("w1", token="tok-1")

    await _run(node.supervisor())

    assert node.records()["w1"].handle == DockerHandle(
        container_id=container.id, container_name="w1"
    )
    assert node.daemon.runs == 0


@pytest.mark.asyncio
async def test_an_interrupted_create_leaves_a_container_another_token_runs(
    node: _Node,
) -> None:
    _launching(node, "w1", "tok-1")
    foreign = node.daemon.add_foreign("w1", token="tok-other")

    wm = node.supervisor()
    await _run(wm)
    assert not await wm.start_worker("w1")

    assert node.daemon.containers == {foreign.id: foreign}
    assert node.records()["w1"].handle is None
    assert node.daemon.runs == 0


@pytest.mark.asyncio
async def test_an_interrupted_create_that_made_nothing_is_launched(
    node: _Node,
) -> None:
    _launching(node, "w1", "tok-1")

    wm = node.supervisor()
    await _run(wm)

    assert node.daemon.runs == 1
    assert wm._registry.get_by_alias("w1").token == "tok-1"


# ---------------------------------------------------------------- VastAI ----


@pytest.mark.asyncio
async def test_a_vast_instance_is_recorded_before_it_is_queried_and_restored(
    node: _Node,
) -> None:
    wm = node.supervisor()
    await _run(wm)
    seen: list[ProviderHandle | None] = []
    node.vast.on_show_instance = lambda _: seen.append(
        next(iter(node.records().values())).handle
    )

    info = await wm.create_worker(WorkerInitConfig(provider="vastai"))

    assert seen == [VastHandle(instance_id=100, created_instance=True)]
    assert node.vast_keys == ["deployment-key"]
    second = node.supervisor()
    await _run(second)
    assert node.vast.created == [100]
    assert second._registry.get_by_alias(info.alias).holds_worker()


@pytest.mark.asyncio
async def test_an_interrupted_vast_create_is_never_rented_again(node: _Node) -> None:
    node.store.create(
        worker_record(
            "v1",
            provider="vastai",
            config={"worker_alias": "v1"},
            state=RecordState.PROVISIONING,
        )
    )
    await _run(node.supervisor())

    assert node.vast.created == []
    assert "v1" in node.records()


@pytest.mark.asyncio
async def test_a_cancelled_vast_start_keeps_its_launch_unresolved(
    node: _Node,
) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(
        WorkerInitConfig(provider="vastai", init_on_start=False)
    )
    node.vast.gate = threading.Event()
    starting = asyncio.ensure_future(wm.start_worker(info.alias))
    await asyncio.to_thread(node.vast.creating.wait, 5)

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    assert node.records()[info.alias].state is RecordState.PROVISIONING
    node.vast.gate.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "destroy_result, listing, gone",
    [(None, None, True), ("", [], True), ("", "", False), ("", [{"id": 100}], False)],
)
async def test_a_vast_removal_is_confirmed_only_by_the_api(
    node: _Node, destroy_result: Any, listing: Any, gone: bool
) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(WorkerInitConfig(provider="vastai"))
    restarted = await _restarted_removing(node, info.alias)
    node.vast.destroy_result, node.vast.listing = destroy_result, listing

    await _settle(restarted)

    assert (info.alias not in node.records()) is gone


@pytest.mark.asyncio
async def test_a_vast_worker_whose_instance_is_gone_stops(node: _Node) -> None:
    wm = node.supervisor()
    await _run(wm)
    info = await wm.create_worker(WorkerInitConfig(provider="vastai"))
    node.vast.live.clear()
    node.vast.destroy_result = ""

    assert await wm.stop_worker(info.alias)

    assert node.records()[info.alias].handle is None


def test_vast_workers_need_the_deployment_key(node: _Node) -> None:
    factory = vastai_adapter.VastAIWorkerFactory(_PRINCIPAL, None, lambda _: False)
    with pytest.raises(ValueError, match="VAST_API_KEY"):
        factory.create_worker(
            WorkerTokenType("tok"), vastai_adapter.VastAIWorkerConfig()
        )


# --------------------------------------------------- aliases and hardware ----


@pytest.mark.asyncio
async def test_a_generated_alias_skips_one_a_restored_worker_holds(
    node: _Node,
) -> None:
    first = node.supervisor()
    await _run(first)
    for provider in ("docker", "vastai"):
        info = await first.create_worker(WorkerInitConfig(provider=provider))
        await first.stop_worker(info.alias)

    second = node.supervisor()
    await _run(second)
    docker = await second.create_worker(WorkerInitConfig())
    vast = await second.create_worker(WorkerInitConfig(provider="vastai"))

    assert docker.alias == "flowmesh_server_worker_cpu_1"
    assert vast.alias == "flowmesh_vastai_worker_1"


@pytest.mark.asyncio
async def test_an_external_worker_cannot_take_a_recorded_alias(
    node: _Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.env.EXTERNAL_WORKER_TOKEN", "secret")
    node.write_config(_entry("w1"))
    await _run(node.supervisor())
    wm = await _restarted_removing(node, "w1")
    node.daemon.unreachable = True

    token = WorkerTokenType(mint_external_token("secret", "w1"))
    assert await wm.admit_worker(token) is None

    assert node.records()["w1"].state is RecordState.REMOVING


@pytest.mark.asyncio
async def test_a_docker_worker_takes_its_registration_hardware_only_unprobed(
    node: _Node,
) -> None:
    node.write_config(_entry("w1"))
    await _run(node.supervisor())
    second = node.supervisor()
    await _run(second)
    worker = second._registry.get_by_alias("w1")
    reported = WorkerHardware.model_validate({"cpu": {"logical_cores": 8}})

    second.worker_registered(worker, reported)
    assert worker.get_info().hardware == reported

    probed = WorkerHardware.model_validate({"cpu": {"logical_cores": 2}})
    worker._hardware = probed  # type: ignore[attr-defined]
    second.worker_registered(worker, reported)
    assert worker.get_info().hardware == probed


@pytest.mark.parametrize(
    "config, error",
    [
        ({"workers": [{"provider": "docker"}]}, "must set worker_config.worker_alias"),
        ({"workers": [_entry("a"), _entry("a")]}, "declared more than once"),
        (
            {"default_worker_config": {"worker_alias": "a"}, "workers": []},
            "cannot set worker_alias",
        ),
        (
            {
                "workers": [
                    {
                        "provider": "vastai",
                        "worker_config": {"worker_alias": "a", "label": "b"},
                    }
                ]
            },
            "label other than its alias",
        ),
    ],
)
def test_boot_entries_must_name_distinct_aliases(
    config: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        ServerWorkerConfig.model_validate(config)
