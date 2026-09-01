"""Operator read access to resident-capacity state.

The SYSTEM/ADMIN-gated resident router projects families, replica incarnations, and
credit-bearing claims into explicit response schemas that never leak an endpoint
api_key, degrades to empty results when resident capacity is disabled, and denies a
non-admin principal.
"""

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, status
from httpx import ASGITransport, AsyncClient
from lumid_hooks import PrincipalContext, ResourceRef

from server.auth.security import authenticate_connection
from server.hooks import PERMISSION_CHECKERS
from server.resident import (
    AdmissionController,
    ClaimCredit,
    ClaimState,
    LifecycleScaleManager,
    ReplicaEndpoint,
    ReplicaIncarnation,
    ReplicaState,
    ResidentCapacityControl,
    ResidentPolicyLimits,
    ResidentStores,
    ServiceClaim,
    ServiceFamily,
)
from server.routers.v1 import resident as resident_router

PREFIX = "/api/v1"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _seeded_stores() -> ResidentStores:
    stores = ResidentStores()
    stores.families.register(
        ServiceFamily(family="fam", engine_batch_key="fam", model_ref="m")
    )
    stores.directory.add(
        ReplicaIncarnation(
            replica_id="rpl-warm",
            family="fam",
            incarnation=1,
            state=ReplicaState.WARM,
            healthy=True,
            serve_task_id="tsk-serve-1",
            worker_id="wkr-1",
            lease_id="lse-1",
            endpoint=ReplicaEndpoint(
                base_url="http://10.0.0.5:8001/v1", model="m", api_key="SECRET-KEY"
            ),
        )
    )
    stores.directory.add(
        ReplicaIncarnation(
            replica_id="rpl-preempted",
            family="fam",
            incarnation=2,
            state=ReplicaState.PREEMPTED,
            healthy=False,
        )
    )
    stores.claims.add(
        ServiceClaim(
            claim_id="scl-1",
            invocation_id="inv-1",
            family="fam",
            admission_epoch=0,
            state=ClaimState.RESERVED,
            credit=ClaimCredit(slots=1),
            replica_id="rpl-warm",
            incarnation=1,
        )
    )
    return stores


def _control(stores: ResidentStores) -> ResidentCapacityControl:
    limits = ResidentPolicyLimits()
    return ResidentCapacityControl(
        stores=stores,
        admission=AdmissionController(stores),
        lifecycle=LifecycleScaleManager(stores, limits=limits, admission_slots=2),
        adapter=object(),  # type: ignore[arg-type]  # read accessors never touch it
        limits=limits,
        binding_resolver=lambda task_id: None,
        settle_cb=lambda *a, **k: True,
        endpoint_probe=lambda serve_task_id: None,
    )


def _app(control: ResidentCapacityControl | None) -> FastAPI:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.resident_observability")
    app.state.resident_control = control
    app.include_router(resident_router.router, prefix=PREFIX)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def restore_checkers() -> Iterator[None]:
    saved = list(PERMISSION_CHECKERS)
    try:
        yield
    finally:
        PERMISSION_CHECKERS[:] = saved


def test_accessors_project_authoritative_state() -> None:
    control = _control(_seeded_stores())

    families = control.list_service_families()
    assert [f.family for f in families] == ["fam"]

    replicas = {r.replica_id: r for r in control.list_replica_incarnations()}
    assert set(replicas) == {"rpl-warm", "rpl-preempted"}  # preempted stays visible

    claims, held = control.list_credit_bearing_claims()
    assert [c.claim_id for c in claims] == ["scl-1"]
    assert held == {"rpl-warm": 1}  # recomputed on read from the credit ledger


@pytest.mark.anyio
async def test_families_endpoint_lists_registered_family() -> None:
    async with _client(_app(_control(_seeded_stores()))) as client:
        resp = await client.get(f"{PREFIX}/resident/families")
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body[0]["family"] == "fam"
    assert body[0]["model_ref"] == "m"


@pytest.mark.anyio
async def test_replicas_endpoint_projects_host_port_without_secret() -> None:
    async with _client(_app(_control(_seeded_stores()))) as client:
        resp = await client.get(f"{PREFIX}/resident/replicas")
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    by_id = {r["replica_id"]: r for r in body}
    assert set(by_id) == {"rpl-warm", "rpl-preempted"}
    warm = by_id["rpl-warm"]
    assert warm["serve_task_id"] == "tsk-serve-1"
    assert warm["endpoint"] == {"host": "10.0.0.5", "port": 8001}
    assert by_id["rpl-preempted"]["endpoint"] is None
    assert "SECRET-KEY" not in resp.text
    assert "api_key" not in resp.text
    assert "base_url" not in resp.text


@pytest.mark.anyio
async def test_replicas_endpoint_filters_by_family() -> None:
    async with _client(_app(_control(_seeded_stores()))) as client:
        hit = await client.get(f"{PREFIX}/resident/replicas", params={"family": "fam"})
        miss = await client.get(
            f"{PREFIX}/resident/replicas", params={"family": "other"}
        )
    assert {r["replica_id"] for r in hit.json()} == {"rpl-warm", "rpl-preempted"}
    assert miss.json() == []


@pytest.mark.anyio
async def test_claims_endpoint_reports_held_credit() -> None:
    async with _client(_app(_control(_seeded_stores()))) as client:
        resp = await client.get(f"{PREFIX}/resident/claims")
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["claims"][0]["claim_id"] == "scl-1"
    assert body["held_credit"] == [{"replica_id": "rpl-warm", "held_slots": 1}]
    assert "api_key" not in resp.text


@pytest.mark.anyio
async def test_endpoints_degrade_to_empty_when_disabled() -> None:
    app = _app(None)
    async with _client(app) as client:
        families = await client.get(f"{PREFIX}/resident/families")
        replicas = await client.get(f"{PREFIX}/resident/replicas")
        claims = await client.get(f"{PREFIX}/resident/claims")
    assert families.status_code == status.HTTP_200_OK and families.json() == []
    assert replicas.status_code == status.HTTP_200_OK and replicas.json() == []
    assert claims.json() == {"claims": [], "held_credit": []}


@pytest.mark.anyio
async def test_non_admin_principal_is_denied(restore_checkers: None) -> None:
    class _DenyNonAdmin:
        name = "deny-non-admin"

        async def accessible_ids(self, *a: Any, **k: Any) -> frozenset[str] | None:
            return None

        async def require(
            self,
            principal: PrincipalContext,
            resource: ResourceRef,
            action: str,
            logger: logging.Logger,
        ) -> None:
            if principal.principal_type != "admin":
                raise HTTPException(status.HTTP_403_FORBIDDEN, "forbidden")

    PERMISSION_CHECKERS.append(_DenyNonAdmin())
    app = _app(_control(_seeded_stores()))
    app.dependency_overrides[authenticate_connection] = lambda: PrincipalContext(
        principal_id="p",
        org_id="o",
        external_id="e",
        principal_type="user",
        scopes=[],
    )
    async with _client(app) as client:
        for path in ("families", "replicas", "claims"):
            resp = await client.get(f"{PREFIX}/resident/{path}")
            assert resp.status_code == status.HTTP_403_FORBIDDEN
