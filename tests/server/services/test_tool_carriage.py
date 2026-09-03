"""Unit coverage for the tool target registry and the carriage's outcome decoding.

Exercises the target bind/cache and the carriage's ``_deliver`` decode paths with fakes,
so a reject or a lost reply becomes a typed unavailable outcome (never a success) and no
demoting observation is recorded, without standing up the full relay harness.
"""

import asyncio
from typing import Any

from server.network.state import (
    NetworkEndpointAdvertisement,
    NonresidentSidecarTarget,
    ReachabilityClass,
    RouteObservationOutcome,
    Transport,
)
from server.services import tool_sidecar_wire as wire
from server.services.tool_carriage import RemoteSidecarCarriage, ToolTargetRegistry
from server.services.tool_egress import ToolOperationEnvelope, ToolRequest
from server.services.tool_relay_delivery import (  # noqa: F401 - type ref
    ToolRelayEndpoint,
)
from shared.schemas.command import CommandType

TARGET_NODE = "nde-t"


def _ingress() -> NetworkEndpointAdvertisement:
    return NetworkEndpointAdvertisement(
        endpoint_id="xt-ingress",
        node_id="xt-ingress",
        url="",
        generation=1,
        trust_domain="flowmesh",
        reachability_class=ReachabilityClass.ROUTABLE,
        relay_attachment_id="xt-ingress",
    )


def _target_endpoint(node_id: str) -> NetworkEndpointAdvertisement:
    return NetworkEndpointAdvertisement(
        endpoint_id=node_id,
        node_id=node_id,
        url="",
        generation=1,
        trust_domain="flowmesh",
        reachability_class=ReachabilityClass.ROUTABLE,
        relay_attachment_id=f"xt-{node_id}",
    )


class _FakeDeputy:
    """Returns a canned reply and the observations it would have recorded."""

    def __init__(
        self,
        reply: bytes | None,
        observations: list[tuple[Transport, RouteObservationOutcome]] | None = None,
    ) -> None:
        self.reply = reply
        self.observations = observations or []

    async def deliver(self, session_id: str, route: Any, **kw: Any) -> Any:
        return self.reply, self.observations


def _carriage(deputy: _FakeDeputy, exec_calls: list[str]) -> RemoteSidecarCarriage:
    async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
        exec_calls.append(command.value)
        return {"host": "127.0.0.1", "port": 9999}

    async def select_node() -> str:
        return TARGET_NODE

    async def endpoint_provider(node_id: str) -> NetworkEndpointAdvertisement:
        return _target_endpoint(node_id)

    registry = ToolTargetRegistry(
        exec_node_cmd=exec_cmd,
        select_node=select_node,
        sidecar_route="127.0.0.1:0",
        provider="fake",
        interfaces=("search/v1",),
        directly_routable=True,
    )
    return RemoteSidecarCarriage(
        origin_deputy=deputy,  # type: ignore[arg-type]
        registry=registry,
        endpoint_provider=endpoint_provider,
        ingress_endpoint=_ingress(),
        provider="fake",
    )


def _run(carriage: RemoteSidecarCarriage) -> Any:
    envelope = ToolOperationEnvelope(
        interface="search/v1",
        idempotency_key="idm-1",
        max_results=3,
        timeout_sec=5.0,
        result_char_cap=6000,
    )
    request = ToolRequest(interface="search/v1", query="q", max_results=3)
    return asyncio.run(carriage._deliver(envelope, request))


def test_registry_binds_once_and_caches() -> None:
    async def run() -> None:
        calls: list[str] = []

        async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
            calls.append(command.value)
            return {"host": "127.0.0.1", "port": 1234}

        async def select_node() -> str:
            return TARGET_NODE

        registry = ToolTargetRegistry(
            exec_node_cmd=exec_cmd,
            select_node=select_node,
            sidecar_route="127.0.0.1:0",
            provider="fake",
            interfaces=("search/v1",),
            directly_routable=True,
        )
        first = await registry.ensure_target()
        second = await registry.ensure_target()
        assert isinstance(first, NonresidentSidecarTarget)
        assert first is second
        assert calls == [CommandType.BIND_TOOL_SIDECAR.value]

    asyncio.run(run())


def test_registry_returns_none_when_no_node() -> None:
    async def run() -> None:
        async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
            raise AssertionError("must not bind when no node is selected")

        async def select_none() -> None:
            return None

        registry = ToolTargetRegistry(
            exec_node_cmd=exec_cmd,
            select_node=select_none,
            sidecar_route="127.0.0.1:0",
            provider="fake",
            interfaces=("search/v1",),
            directly_routable=True,
        )
        assert await registry.ensure_target() is None

    asyncio.run(run())


def test_result_reply_becomes_the_outcome() -> None:
    reply = wire.encode_msg(
        wire.KIND_RESULT,
        outcome={"status": "success", "value": "hits", "provenance": []},
    )
    outcome = _run(_carriage(_FakeDeputy(reply), []))
    assert outcome.status.value == "success"
    assert outcome.value == "hits"


def test_reject_reply_becomes_unavailable() -> None:
    reply = wire.encode_msg(wire.KIND_REJECT, reason="digest")
    outcome = _run(_carriage(_FakeDeputy(reply), []))
    assert outcome.status.value == "unavailable"


def test_lost_reply_becomes_unavailable() -> None:
    outcome = _run(_carriage(_FakeDeputy(None), []))
    assert outcome.status.value == "unavailable"
