"""Resident materialization owns the serve task under the resolved system principal.

A materialized resident replica's backing serve/`dev_model` task is submitted and
registered under the principal `FLOWMESH_API_KEY` resolves to — not a synthetic owner no
principal can authenticate as — so the resource registrars record its ownership and an
operator reads its logs through the normal owner-scoped path.
"""

import asyncio
import json
import logging
from typing import Any

from flowmesh_hook import ResourceKind
from lumid_hooks import PrincipalContext, ResourceRef

from server.config import ResidentCapacityConfig
from server.hooks import RESOURCE_REGISTRARS
from server.resident import ReplicaIncarnation, ServiceFamily
from server.services.resident_materializer import materialize_resident_replica

_LOGGER = logging.getLogger("test.resident_materializer")
_SYSTEM = PrincipalContext(
    principal_id="operator",
    org_id="acme",
    external_id="op",
    principal_type="admin",
    scopes=["*"],
)
_FAMILY = ServiceFamily(family="m", engine_batch_key="m", model_ref="m")
_REPLICA = ReplicaIncarnation(replica_id="rpl-1", family="m", incarnation=1)


class _Entry:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class _FakeRuntime:
    def __init__(self) -> None:
        self.register_call: tuple[str, str, str, str] | None = None

    async def register(
        self, owner_id: str, org_id: str, payload: str, format: str
    ) -> tuple[str, list[_Entry]]:
        self.register_call = (owner_id, org_id, payload, format)
        return "wfl-resident", [_Entry("tsk-resident")]


class _RecordingRegistrar:
    name = "recording"

    def __init__(self) -> None:
        self.registered: list[tuple[str, ResourceRef]] = []

    async def register(
        self, principal: PrincipalContext, resource: ResourceRef, logger: Any
    ) -> None:
        self.registered.append((principal.principal_id, resource))

    async def deregister(self, *args: Any, **kwargs: Any) -> None: ...

    async def reconcile(self, *args: Any, **kwargs: Any) -> None: ...


def _materialize(config: ResidentCapacityConfig, runtime: Any, *registrars: Any) -> str:
    RESOURCE_REGISTRARS.clear()
    RESOURCE_REGISTRARS.extend(registrars)
    try:
        return asyncio.run(
            materialize_resident_replica(
                runtime, _SYSTEM, config, _FAMILY, _REPLICA, _LOGGER
            )
        )
    finally:
        RESOURCE_REGISTRARS.clear()


def test_serve_task_is_owned_and_registered_under_the_system_principal() -> None:
    runtime = _FakeRuntime()
    registrar = _RecordingRegistrar()
    config = ResidentCapacityConfig(substrate="dev_model", access_mode="direct")

    task_id = _materialize(config, runtime, registrar)
    assert task_id == "tsk-resident"

    assert runtime.register_call is not None
    owner_id, org_id, payload, fmt = runtime.register_call
    assert (owner_id, org_id) == ("operator", "acme")
    assert (owner_id, org_id) != ("resident-capacity-control", "system")
    assert fmt == "native"
    assert json.loads(payload)["spec"]["taskType"] == "dev_model"

    owned = {(pid, res.kind, res.id) for pid, res in registrar.registered}
    assert ("operator", ResourceKind.WORKFLOW.value, "wfl-resident") in owned
    assert ("operator", ResourceKind.TASK.value, "tsk-resident") in owned
    task_meta = next(
        res.metadata
        for pid, res in registrar.registered
        if res.kind == ResourceKind.TASK.value
    )
    assert task_meta == {"workflow_id": "wfl-resident"}


def test_gpu_serve_substrate_requests_a_gpu_replica() -> None:
    runtime = _FakeRuntime()
    config = ResidentCapacityConfig(substrate="serve", serve_ttl_sec=600)

    _materialize(config, runtime, _RecordingRegistrar())

    assert runtime.register_call is not None
    spec = json.loads(runtime.register_call[2])["spec"]
    assert spec["taskType"] == "serve"
    assert spec["resources"]["hardware"]["gpu"]["count"] == 1
    assert spec["ttlSeconds"] == 600


def test_materialization_survives_without_registered_registrars() -> None:
    runtime = _FakeRuntime()
    config = ResidentCapacityConfig(substrate="dev_model")

    task_id = _materialize(config, runtime)
    assert task_id == "tsk-resident"
    assert runtime.register_call is not None
    assert runtime.register_call[:2] == ("operator", "acme")
