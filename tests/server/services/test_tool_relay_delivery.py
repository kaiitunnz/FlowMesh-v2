"""Round-trip the claim-free tool control_relay over the xt:* reverse-rendezvous.

Wires two ``ToolRelayEndpoint``s and their attachments to the shared root bridge over a
``FakeBinaryRedis`` and a loopback ``ExternalToolSidecarServer``, and drives one bounded
operation from the in-server origin to the worker sidecar and back — proving the
delivery uses the tool namespace, egresses at the target, and returns the typed outcome.
"""

import asyncio
import time

from server.network.rendezvous import RootCursorStore, RootRendezvousBridge
from server.network.reverse_relay import (
    TOOL_RELAY_KEYSPACE,
    RelaySessionStore,
    RelayStreamStore,
)
from server.network.state import RouteCandidate, RouteHop, Transport
from server.services import tool_sidecar_wire as wire
from server.services.external_tool_sidecar import (
    ExternalToolSidecarListener,
    ExternalToolSidecarServer,
)
from server.services.search_providers import SearchResult
from server.services.tool_egress import (
    ExternalToolSidecar,
    RemoteToolOperationEnvelope,
    ToolRequest,
    tool_request_digest,
)
from server.services.tool_relay_delivery import ToolRelayEndpoint
from server.supervisor.services.reverse_relay_attachment import ReverseRelayAttachment
from tests.server.network._relay_fakes import FakeBinaryRedis

TARGET_ID = "stg-1"
TARGET_GEN = 3
PROVIDER = "fake"


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(
        self, query: str, *, max_results: int, timeout_sec: float
    ) -> list[SearchResult]:
        self.calls.append(query)
        return [SearchResult(title=f"r:{query}", url="http://x", snippet="s")]


def _envelope(query: str, max_results: int = 3) -> RemoteToolOperationEnvelope:
    return RemoteToolOperationEnvelope(
        interface="search/v1",
        provider=PROVIDER,
        idempotency_key="idm-abc",
        request_digest=tool_request_digest("search/v1", query, max_results),
        target_id=TARGET_ID,
        target_generation=TARGET_GEN,
        deadline_epoch=time.time() + 30,
        max_results=max_results,
        timeout_sec=5.0,
        result_char_cap=6000,
    )


def _operation_payload(query: str) -> bytes:
    env = _envelope(query)
    req = ToolRequest(interface="search/v1", query=query, max_results=3)
    return wire.encode_msg(
        wire.KIND_OPERATION,
        envelope=env.model_dump(mode="json"),
        request=req.model_dump(mode="json"),
    )


def _route(sidecar_route: str) -> RouteCandidate:
    return RouteCandidate(
        transport=Transport.CONTROL_RELAY,
        hops=(
            RouteHop(transport=Transport.CONTROL_RELAY, endpoint="", node_id="nde-o"),
            RouteHop(
                transport=Transport.CONTROL_RELAY,
                endpoint=sidecar_route,
                node_id="nde-t",
            ),
        ),
    )


class _Harness:
    def __init__(self, provider: _FakeProvider) -> None:
        self.redis = FakeBinaryRedis()
        self.sidecar = ExternalToolSidecarListener(
            ExternalToolSidecarServer(
                sidecar=ExternalToolSidecar(provider),
                target_id=TARGET_ID,
                target_generation=TARGET_GEN,
                provider=PROVIDER,
                interfaces=frozenset({"search/v1"}),
            ),
            route="127.0.0.1:0",
        )
        self.origin = ToolRelayEndpoint(self.redis, "nde-o", recv_budget_sec=5.0)
        self.target = ToolRelayEndpoint(self.redis, "nde-t", recv_budget_sec=5.0)
        self.bridge = RootRendezvousBridge(
            RelayStreamStore(self.redis, TOOL_RELAY_KEYSPACE),
            RelaySessionStore(self.redis, TOOL_RELAY_KEYSPACE),
            RootCursorStore(self.redis, TOOL_RELAY_KEYSPACE),
        )
        self.origin_attach = ReverseRelayAttachment(
            self.redis, "nde-o", self.origin, owner="o", keyspace=TOOL_RELAY_KEYSPACE
        )
        self.target_attach = ReverseRelayAttachment(
            self.redis, "nde-t", self.target, owner="t", keyspace=TOOL_RELAY_KEYSPACE
        )

    async def start(self) -> str:
        host, port = await self.sidecar.start()
        return f"{host}:{port}"

    async def pump_step(self) -> None:
        await self.bridge.pump_node("nde-o")
        await self.target_attach.pump_once()
        await self.bridge.pump_node("nde-t")
        await self.origin_attach.pump_once()
        await asyncio.sleep(0.002)


async def _drive(harness: _Harness, done: asyncio.Event) -> None:
    while not done.is_set():
        await harness.pump_step()


def test_operation_round_trips_over_the_tool_relay() -> None:
    async def run() -> None:
        provider = _FakeProvider()
        harness = _Harness(provider)
        sidecar_route = await harness.start()
        done = asyncio.Event()

        async def exchange() -> bytes | None:
            reply = await harness.origin.deliver(
                "xtr-1",
                invocation_id="inv-1",
                idm="idm-abc",
                origin_node="nde-o",
                target_node="nde-t",
                sidecar_route=sidecar_route,
                operation_payload=_operation_payload("what is flowmesh"),
            )
            done.set()
            return reply

        driver = asyncio.ensure_future(_drive(harness, done))
        try:
            reply = await asyncio.wait_for(exchange(), timeout=10.0)
        finally:
            done.set()
            await driver
        assert reply is not None
        msg = wire.decode_msg(reply)
        assert msg["kind"] == wire.KIND_RESULT
        assert msg["outcome"]["status"] == "success"
        assert "what is flowmesh" in msg["outcome"]["value"]
        # The provider egress happened at the target sidecar, not the origin.
        assert provider.calls == ["what is flowmesh"]
        await harness.sidecar.stop()

    asyncio.run(run())
