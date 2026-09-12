"""Operator read access to sealed activation-private state.

The SYSTEM/ADMIN-gated inventory projects sealed generations across the live engines
with their holder evidence and the advisory state-control decision, reports nothing
while the policy surface is off, denies a non-admin principal, and carries no state
bytes. Reading decides only: no generation is materialized, copied, or deleted, and no
attachment or claim is minted.
"""

import logging
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
from fastapi import FastAPI, HTTPException, status
from httpx import ASGITransport, AsyncClient
from lumid_hooks import PrincipalContext, ResourceRef

from server.auth.security import authenticate_connection
from server.config import PolicySurfaceConfig
from server.hooks import PERMISSION_CHECKERS
from server.policy import (
    PolicySurface,
    SealedGenerationEvidence,
    StateControlVerb,
    build_policy_surface,
)
from server.routers.v1 import private_state as private_state_router
from server.services.private_state_inventory import sealed_state_inventory
from shared.private_state import BundleProfile, SealedComponent, StateComponentKind

PREFIX = "/api/v1"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _evidence(reference_id: str, **overrides: Any) -> SealedGenerationEvidence:
    base: dict[str, Any] = dict(
        reference_id=reference_id,
        instance_id="wfl-1",
        activation_id="act-1",
        owner_id="user-1",
        org_id="org-1",
        tenant="tenant-1",
        profile=BundleProfile.AGENT_HARNESS,
        generation=3,
        owner_worker_id="wkr-1",
        owner_incarnation=2,
        components=(
            SealedComponent(
                kind=StateComponentKind.HARNESS_HOME_FS,
                schema_version=1,
                content_digest="digest-home",
                size_bytes=10,
                entry_count=2,
            ),
        ),
        attached=False,
        resumable=False,
        exportable=False,
        sealed_at="2026-01-01T00:00:00Z",
    )
    return SealedGenerationEvidence(**{**base, **overrides})


def _runtime(*evidence: SealedGenerationEvidence) -> Any:
    runtime = mock.Mock()
    runtime.sealed_private_state.return_value = list(evidence)
    return runtime


def _surface(warm: int = 8) -> PolicySurface:
    surface = build_policy_surface(
        PolicySurfaceConfig(enabled=True, warm_generations=warm)
    )
    assert surface is not None
    return surface


def _app(runtime: Any, surface: PolicySurface | None) -> FastAPI:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.private_state_inventory")
    app.state.runtime = runtime
    app.state.policy_surface = surface
    app.include_router(private_state_router.router, prefix=PREFIX)
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


def test_inventory_reports_the_newest_seal_first() -> None:
    older = _evidence("aps-old", sealed_at="2026-01-01T00:00:00Z")
    newer = _evidence("aps-new", sealed_at="2026-02-01T00:00:00Z")

    entries = sealed_state_inventory(_runtime(older, newer), _surface())

    assert [entry.evidence.reference_id for entry in entries] == ["aps-new", "aps-old"]


def test_a_generation_a_holder_writes_gets_no_decision() -> None:
    entries = sealed_state_inventory(
        _runtime(_evidence("aps-live", attached=True)), _surface()
    )

    assert entries[0].decision is None


def test_a_cold_eviction_is_honored_and_a_resumable_one_is_not() -> None:
    cold = _evidence("aps-cold", sealed_at="2026-01-01T00:00:00Z")
    resumable = _evidence(
        "aps-resumable", sealed_at="2026-01-02T00:00:00Z", resumable=True
    )

    verbs = {
        entry.evidence.reference_id: entry.decision.verb if entry.decision else None
        for entry in sealed_state_inventory(_runtime(cold, resumable), _surface(warm=0))
    }

    assert verbs["aps-cold"] is StateControlVerb.EVICT
    assert verbs["aps-resumable"] is StateControlVerb.RETAIN


def test_the_inventory_reads_without_touching_the_ledger() -> None:
    runtime = _runtime(_evidence("aps-1"))

    sealed_state_inventory(runtime, _surface())

    assert runtime.mock_calls == [mock.call.sealed_private_state()]


@pytest.mark.anyio
async def test_endpoint_projects_evidence_and_decision() -> None:
    app = _app(_runtime(_evidence("aps-1")), _surface(warm=0))
    async with _client(app) as client:
        resp = await client.get(f"{PREFIX}/private-state/generations")

    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()[0]
    assert body["reference_id"] == "aps-1"
    assert body["generation"] == 3
    assert body["owner_worker_id"] == "wkr-1"
    assert body["owner_incarnation"] == 2
    assert body["exportable"] is False
    assert body["components"] == [
        {
            "kind": "harness_home_fs",
            "schema_version": 1,
            "content_digest": "digest-home",
            "size_bytes": 10,
            "entry_count": 2,
        }
    ]
    assert body["decision"] == StateControlVerb.EVICT.value


@pytest.mark.anyio
async def test_endpoint_filters_by_query() -> None:
    runtime = _runtime(_evidence("aps-1"), _evidence("aps-2", instance_id="wfl-2"))
    async with _client(_app(runtime, _surface())) as client:
        hit = await client.get(
            f"{PREFIX}/private-state/generations", params={"instance_id": "wfl-2"}
        )

    assert [item["reference_id"] for item in hit.json()] == ["aps-2"]


@pytest.mark.anyio
async def test_endpoint_reports_nothing_while_the_surface_is_off() -> None:
    runtime = _runtime(_evidence("aps-1"))
    async with _client(_app(runtime, None)) as client:
        resp = await client.get(f"{PREFIX}/private-state/generations")

    assert resp.status_code == status.HTTP_200_OK
    assert resp.json() == []
    runtime.sealed_private_state.assert_not_called()


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
    app = _app(_runtime(_evidence("aps-1")), _surface())
    app.dependency_overrides[authenticate_connection] = lambda: PrincipalContext(
        principal_id="p",
        org_id="o",
        external_id="e",
        principal_type="user",
        scopes=[],
    )
    async with _client(app) as client:
        resp = await client.get(f"{PREFIX}/private-state/generations")

    assert resp.status_code == status.HTTP_403_FORBIDDEN
