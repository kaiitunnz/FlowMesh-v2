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

import pytest
from fastapi import HTTPException
from flowmesh_hook import ResourceAction, ResourceKind
from lumid_hooks import PrincipalContext, ResourceRef

from server.auth import require_permission
from server.config import ResidentCapacityConfig
from server.hooks import PERMISSION_CHECKERS, RESOURCE_REGISTRARS
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
_FOREIGN = PrincipalContext(
    principal_id="tenant-x",
    org_id="tenant-x",
    external_id="tx",
    principal_type="user",
    scopes=["user"],
)
_FAMILY = ServiceFamily(family="m", engine_batch_key="m", model_ref="m")
_REPLICA = ReplicaIncarnation(replica_id="rpl-1", family="m", incarnation=1)


@pytest.fixture(autouse=True)
def _clear_hooks() -> Any:
    RESOURCE_REGISTRARS.clear()
    PERMISSION_CHECKERS.clear()
    yield
    RESOURCE_REGISTRARS.clear()
    PERMISSION_CHECKERS.clear()


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


class _OwnershipRegistrar:
    """Records each registered resource's owning principal, keyed by resource id."""

    name = "ownership"

    def __init__(self) -> None:
        self.owner_of: dict[str, str] = {}

    async def register(
        self, principal: PrincipalContext, resource: ResourceRef, logger: Any
    ) -> None:
        if resource.id is not None:
            self.owner_of[resource.id] = principal.principal_id

    async def deregister(
        self, principal: PrincipalContext, resource: ResourceRef, logger: Any
    ) -> None:
        if resource.id is not None:
            self.owner_of.pop(resource.id, None)

    async def reconcile(self, *args: Any, **kwargs: Any) -> None: ...


class _OwnerOnlyChecker:
    """Grants an owner (and admins) access to its resource; denies everyone else.

    A `RESULT` check on a task id resolves against the owning task's registration,
    mirroring how result/log ownership is inferred from the owning task.
    """

    name = "owner-only"

    def __init__(self, owner_of: dict[str, str]) -> None:
        self._owner_of = owner_of

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: Any,
    ) -> None:
        if principal.principal_type == "admin":
            return
        owner = self._owner_of.get(resource.id) if resource.id is not None else None
        if owner is None or owner != principal.principal_id:
            raise HTTPException(status_code=403, detail="not owner")

    async def accessible_ids(self, *args: Any, **kwargs: Any) -> None:
        return None


# A real-auth operator that owns by principal id (not by an admin scope bypass).
_OWNER = PrincipalContext(
    principal_id="operator",
    org_id="acme",
    external_id="op",
    principal_type="user",
    scopes=["user"],
)


def _materialize_owned(owner: PrincipalContext) -> tuple[str, _OwnershipRegistrar]:
    registrar = _OwnershipRegistrar()
    RESOURCE_REGISTRARS.append(registrar)
    PERMISSION_CHECKERS.append(_OwnerOnlyChecker(registrar.owner_of))
    config = ResidentCapacityConfig(substrate="dev_model")
    runtime: Any = _FakeRuntime()
    task_id = asyncio.run(
        materialize_resident_replica(runtime, owner, config, _FAMILY, _REPLICA, _LOGGER)
    )
    return task_id, registrar


def _read_logs(principal: PrincipalContext, task_id: str) -> None:
    asyncio.run(
        require_permission(
            principal, ResourceKind.RESULT, task_id, ResourceAction.READ, _LOGGER
        )
    )


def test_owner_principal_may_read_the_resident_task_logs() -> None:
    task_id, registrar = _materialize_owned(_OWNER)
    assert registrar.owner_of[task_id] == "operator"

    # The owning principal is granted by ownership match (not by an admin bypass),
    # and the resolved admin operator is granted too.
    _read_logs(_OWNER, task_id)
    _read_logs(_SYSTEM, task_id)


def test_foreign_tenant_is_denied_the_resident_task_logs() -> None:
    task_id, _ = _materialize_owned(_OWNER)

    with pytest.raises(HTTPException) as excinfo:
        _read_logs(_FOREIGN, task_id)
    assert excinfo.value.status_code == 403
