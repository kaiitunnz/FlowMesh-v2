"""Tests for the /api/v1/network echo route."""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.app_state import get_logger, get_network_plane, get_node_registry
from server.auth.security import authenticate_connection, default_principal
from server.network.state import ResolvedRoute, RouteCandidate, RouteHop, Transport
from server.routers.v1 import network as network_router
from shared.network.frame_stream import MAX_PROBE_BYTES
from shared.schemas.command import CommandErrorCode, CommandResponse

PREFIX = "/api/v1"
_BODY = {"origin_node_id": "nod-1", "listener": {"replica_id": "r", "node_id": "nod-2"}}


def _route(*transports: Transport) -> ResolvedRoute:
    return ResolvedRoute(
        origin_id="rog-1",
        target_node_id="nod-2",
        listener_generation=0,
        route_epoch=1,
        candidates=tuple(
            RouteCandidate(
                transport=transport,
                hops=(
                    ()
                    if transport is Transport.CONTROL_RELAY
                    else (RouteHop(transport=transport, endpoint="h:1"),)
                ),
            )
            for transport in transports
        ),
    )


def _make_app(
    response: CommandResponse,
    route: ResolvedRoute | None = None,
) -> tuple[FastAPI, MagicMock, MagicMock]:
    plane = MagicMock()
    plane.resolve = AsyncMock(
        return_value=(
            MagicMock(),
            route or _route(Transport.NODE_RELAY, Transport.CONTROL_RELAY),
        )
    )
    plane.connect_budget_sec = 1.0
    plane.reachability_states = MagicMock(return_value={})
    registry = MagicMock()
    registry.exec_node_cmd = AsyncMock(return_value=response)
    app = FastAPI()
    app.include_router(network_router.router, prefix=PREFIX)
    app.dependency_overrides[authenticate_connection] = default_principal
    app.dependency_overrides[get_network_plane] = lambda: plane
    app.dependency_overrides[get_node_registry] = lambda: registry
    app.dependency_overrides[get_logger] = lambda: logging.getLogger("test.network")
    return app, plane, registry


async def _echo(app: FastAPI, body: dict) -> Any:
    with patch.object(network_router, "require_permission", AsyncMock()):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as ac:
            return await ac.post(f"{PREFIX}/network/echo", json=body)


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
    app, _plane, _registry = _make_app(response)
    resp = await _echo(app, _BODY)
    assert resp.status_code == expected
    assert resp.json()["detail"] == "boom"


@pytest.mark.anyio
async def test_a_verified_probe_returns_the_request_payload() -> None:
    response = CommandResponse(
        command_id="c",
        success=True,
        data={
            "selected_transport": "node_relay",
            "observations": [{"transport": "node_relay", "outcome": "verified"}],
        },
    )
    app, plane, _registry = _make_app(response)
    resp = await _echo(app, _BODY | {"payload": "hello"})
    assert resp.status_code == 200
    assert resp.json()["selected_transport"] == "node_relay"
    assert resp.json()["echoed"] == "hello"
    plane.record_observations.assert_called_once()


@pytest.mark.anyio
async def test_an_oversized_payload_is_refused_before_a_route_resolves() -> None:
    app, plane, registry = _make_app(CommandResponse(command_id="c", success=True))
    resp = await _echo(app, _BODY | {"payload": "x" * (MAX_PROBE_BYTES + 1)})
    assert resp.status_code == 422
    plane.resolve.assert_not_awaited()
    registry.exec_node_cmd.assert_not_awaited()


@pytest.mark.anyio
async def test_a_route_with_nothing_to_dial_sends_no_node_command() -> None:
    app, plane, registry = _make_app(
        CommandResponse(command_id="c", success=True),
        route=_route(Transport.CONTROL_RELAY),
    )
    resp = await _echo(app, _BODY)
    assert resp.status_code == 200
    assert resp.json()["selected_transport"] is None
    assert resp.json()["candidates"] == ["control_relay"]
    registry.exec_node_cmd.assert_not_awaited()
    plane.record_observations.assert_not_called()
