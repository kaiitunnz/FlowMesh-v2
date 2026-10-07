"""A provisioned worker's registration is recorded, and the registrations a previous
supervisor run held are unregistered until the root has applied the unregister."""

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis
import grpc
import pytest

from server.clients.redis import worker_key
from server.hooks import PrincipalContext
from server.registries.worker import WorkerRegistry as WorkerRecords
from server.supervisor.manager import WorkerManager
from server.supervisor.provisioning import WorkerProvisioningStore
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.task_listener import TaskListener
from tests.server.redis_helpers import fake_redis_client
from tests.server.supervisor_helpers import memory_store, worker_record
from tests.server.test_docker_removal_in_progress import _adapter
from tests.server.test_external_worker import _Aborted, _FakeTaskListener, _register
from tests.server.test_external_worker_reregistration import _RecordingRelay

_LOGGER = logging.getLogger("test.previous_registrations")


def _servicer() -> (
    tuple[
        SupervisorServicer, WorkerManager, _RecordingRelay, Any, WorkerProvisioningStore
    ]
):
    client = fake_redis_client(fakeredis.FakeServer())
    registry = WorkerRegistry()
    store = memory_store()
    manager = WorkerManager(
        MagicMock(spec=PrincipalContext), "unused", registry, _LOGGER, store
    )
    manager._is_started = True
    relay = _RecordingRelay()
    servicer = SupervisorServicer(
        registry,
        client.sync,
        WorkerRecords(client),
        "nde-1",
        "box",
        cast(TaskListener, _FakeTaskListener()),
        relay,
        manager,
        _LOGGER,
    )
    return servicer, manager, relay, client.sync, store


def _provisioned(manager: WorkerManager) -> Any:
    adapter = _adapter(MagicMock())
    record = worker_record(adapter.alias, token=adapter.token)
    manager._provisioned.create(record)
    manager._registry.add(adapter)
    return adapter


def _unregisters(relay: _RecordingRelay) -> list[str]:
    return [e["worker_id"] for e in relay.events if e["type"] == "UNREGISTER"]


@pytest.mark.asyncio
async def test_a_provisioned_worker_registers_once_its_id_is_recorded() -> None:
    servicer, manager, _, _, store = _servicer()
    adapter = _provisioned(manager)

    worker_id = await _register(servicer, adapter.token, adapter.alias)

    [record] = store.load()
    assert record.worker_id == worker_id
    assert servicer._registry.get_worker_id(adapter.token) == worker_id


@pytest.mark.asyncio
async def test_a_registration_whose_id_cannot_be_recorded_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servicer, manager, relay, redis, store = _servicer()
    adapter = _provisioned(manager)
    monkeypatch.setattr(store, "put", MagicMock(side_effect=ConnectionError))

    with pytest.raises(_Aborted) as aborted:
        await _register(servicer, adapter.token, adapter.alias)

    assert aborted.value.code is grpc.StatusCode.UNAVAILABLE
    assert servicer._registry.get_worker_id(adapter.token) is None
    assert len(_unregisters(relay)) == 1


def test_previous_registrations_are_unregistered_until_the_root_drops_them() -> None:
    servicer, _, relay, redis, _ = _servicer()
    for worker_id, node_alias in (("wkr-1", "box"), ("wkr-2", "other")):
        redis.hash_set(worker_key(worker_id), {"node_alias": node_alias})

    servicer.retire_previous_registrations(["wkr-1", "wkr-2", "wkr-3"])
    assert _unregisters(relay) == ["wkr-1", "wkr-2", "wkr-3"]

    servicer.reconcile_workers()
    assert _unregisters(relay)[3:] == ["wkr-1"]

    redis.delete(worker_key("wkr-1"))
    servicer.reconcile_workers()
    servicer.reconcile_workers()
    assert _unregisters(relay)[4:] == []
