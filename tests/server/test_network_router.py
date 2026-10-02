"""Tests for the /api/v1/network echo route's command-error translation."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.app_state import get_logger, get_network_plane, get_node_registry
from server.auth.security import authenticate_connection, default_principal
from server.routers.v1 import network as network_router
from shared.schemas.command import CommandErrorCode, CommandResponse

PREFIX = "/api/v1"
_BODY = {"origin_node_id": "nod-1", "listener": {"replica_id": "r", "node_id": "nod-2"}}


def _make_app(response: CommandResponse) -> FastAPI:
    plane = MagicMock()
    plane.resolve = AsyncMock(return_value=(MagicMock(), MagicMock()))
    plane.connect_budget_sec = 1.0
    registry = MagicMock()
    registry.exec_node_cmd = AsyncMock(return_value=response)
    app = FastAPI()
    app.include_router(network_router.router, prefix=PREFIX)
    app.dependency_overrides[authenticate_connection] = default_principal
    app.dependency_overrides[get_network_plane] = lambda: plane
    app.dependency_overrides[get_node_registry] = lambda: registry
    app.dependency_overrides[get_logger] = lambda: logging.getLogger("test.network")
    return app


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        (CommandErrorCode.INTERNAL, 500),
        (CommandErrorCode.NOT_READY, 503),
        (CommandErrorCode.UNKNOWN_COMMAND, 501),
    ],
)
async def test_a_failed_route_plan_delivery_maps_its_error_code(
    error_code: CommandErrorCode, expected: int
) -> None:
    response = CommandResponse(
        command_id="c", success=False, message="boom", error_code=error_code
    )
    with patch.object(network_router, "require_permission", AsyncMock()):
        async with AsyncClient(
            transport=ASGITransport(app=_make_app(response)), base_url="http://t"
        ) as ac:
            resp = await ac.post(f"{PREFIX}/network/echo", json=_BODY)
    assert resp.status_code == expected
    assert resp.json()["detail"] == "boom"
