"""End-to-end remote external-tool carriage behind the ExecutionTransport seam.

Runs the carriage's async network I/O on a background event loop and drives its
synchronous ``__call__`` from the test thread — the same sync→async bridge the broker's
thread pool uses — over both a forward dial and a forced reverse-rendezvous, and proves
a keyed provider's credential never leaves the sidecar's local environment.
"""

import asyncio
import threading
from concurrent.futures import Future
from typing import Any

from server.network.rendezvous import RootCursorStore, RootRendezvousBridge
from server.network.reverse_relay import (
    TOOL_RELAY_KEYSPACE,
    RelaySessionStore,
    RelayStreamStore,
)
from server.network.state import NetworkEndpointAdvertisement, ReachabilityClass
from server.supervisor.services.reverse_relay_attachment import ReverseRelayAttachment
from server.tools.external_tool_sidecar import (
    ExternalToolSidecarListener,
    ExternalToolSidecarServer,
)
from server.tools.tool_carriage import (
    RemoteSidecarCarriage,
    ToolEgressOriginDeputy,
    ToolTargetRegistry,
)
from server.tools.tool_egress import (
    ExternalToolSidecar,
    ToolOperationEnvelope,
    ToolRequest,
)
from server.tools.tool_relay_delivery import ToolRelayEndpoint
from shared.schemas.command import CommandType
from shared.tools.providers import SearchResult
from tests.server.network._relay_fakes import FakeBinaryRedis

INGRESS = "xt-ingress"
TARGET_NODE = "nde-t"
TARGET_WORKER = "wrk-1"
TARGET_INCARNATION = 7
QUERY = "what is flowmesh"
SECRET = "sk-super-secret-key"


class _RecordingProvider:
    def __init__(self, api_key: str | None = None) -> None:
        self.calls: list[str] = []
        self.api_key = api_key

    def search(
        self, query: str, *, max_results: int, timeout_sec: float
    ) -> list[SearchResult]:
        self.calls.append(query)
        return [SearchResult(title=f"r:{query}", url="http://x", snippet="s")]


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._ready.set)
        self.loop.run_forever()

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(2.0)

    def submit(self, coro: Any) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(2.0)


class _Harness:
    def __init__(self, *, directly_routable: bool, api_key: str | None = None) -> None:
        self.redis = FakeBinaryRedis()
        self.provider = _RecordingProvider(api_key=api_key)
        self.exec_payloads: list[dict[str, Any]] = []
        self._sidecar: ExternalToolSidecarListener | None = None
        self._directly_routable = directly_routable
        self.loopt = _LoopThread()

        self.ingress_endpoint = ToolRelayEndpoint(self.redis, INGRESS)
        self.target_endpoint = ToolRelayEndpoint(self.redis, TARGET_NODE)
        self.bridge = RootRendezvousBridge(
            RelayStreamStore(self.redis, TOOL_RELAY_KEYSPACE),
            RelaySessionStore(self.redis, TOOL_RELAY_KEYSPACE),
            RootCursorStore(self.redis, TOOL_RELAY_KEYSPACE),
        )
        self.ingress_attach = ReverseRelayAttachment(
            self.redis,
            INGRESS,
            self.ingress_endpoint,
            owner="i",
            keyspace=TOOL_RELAY_KEYSPACE,
        )
        self.target_attach = ReverseRelayAttachment(
            self.redis,
            TARGET_NODE,
            self.target_endpoint,
            owner="t",
            keyspace=TOOL_RELAY_KEYSPACE,
        )
        registry = ToolTargetRegistry(
            exec_node_cmd=self._exec_node_cmd,
            resolve_target=self._resolve_target,
            sidecar_route="127.0.0.1:0",
            provider="fake",
            interfaces=("search/v1",),
            directly_routable=directly_routable,
        )
        self.registry = registry
        self.carriage = RemoteSidecarCarriage(
            origin_deputy=ToolEgressOriginDeputy(
                relay_endpoint=self.ingress_endpoint, connect_budget_sec=3.0
            ),
            registry=registry,
            endpoint_provider=self._target_endpoint_ad,
            ingress_endpoint=NetworkEndpointAdvertisement(
                endpoint_id=INGRESS,
                node_id=INGRESS,
                url="",
                generation=1,
                trust_domain="flowmesh",
                reachability_class=ReachabilityClass.ROUTABLE,
                relay_attachment_id=f"xt-{INGRESS}",
            ),
            provider="fake",
            deadline_sec=30.0,
            route_ttl_sec=30.0,
            connect_budget_sec=3.0,
        )
        self._pump: Future[Any] | None = None

    async def _exec_node_cmd(
        self, node_id: str, command: CommandType, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.exec_payloads.append({"command": command.value, **payload})
        if command is CommandType.BIND_TOOL_SIDECAR:
            server = ExternalToolSidecarServer(
                sidecar=ExternalToolSidecar(self.provider),
                target_id=str(payload["target_id"]),
                target_generation=int(payload["target_generation"]),
                provider="fake",
                interfaces=frozenset({"search/v1"}),
            )
            self._sidecar = ExternalToolSidecarListener(
                server, route=str(payload["route"])
            )
            host, port = await self._sidecar.start()
            self._route = f"{host}:{port}"
            return {"host": host, "port": port}
        return {}

    async def _resolve_target(self, task_id: str) -> tuple[str, str, int]:
        return TARGET_NODE, TARGET_WORKER, TARGET_INCARNATION

    async def _target_endpoint_ad(self, node_id: str) -> NetworkEndpointAdvertisement:
        # No inbound URL, so node_relay is unavailable; control_relay stays feasible via
        # the attachment id. A directly routable target also offers worker_direct.
        return NetworkEndpointAdvertisement(
            endpoint_id=node_id,
            node_id=node_id,
            url="",
            generation=1,
            trust_domain="flowmesh",
            reachability_class=ReachabilityClass.ROUTABLE,
            relay_attachment_id=f"xt-{node_id}",
        )

    async def _pump_forever(self) -> None:
        while True:
            for nid in (INGRESS, TARGET_NODE):
                await self.bridge.pump_node(nid)
            await self.ingress_attach.pump_once()
            await self.target_attach.pump_once()
            await asyncio.sleep(0.003)

    def start(self) -> None:
        self.loopt.start()
        self.carriage.bind_loop(self.loopt.loop)
        if not self._directly_routable:
            # One driver only: pump the bridge and both attachments in a single loop,
            # never alongside the attachments' own run loops, so they do not contend.
            self._pump = self.loopt.submit(self._pump_forever())

    def run_search(self, query: str, max_results: int = 3) -> Any:
        envelope = ToolOperationEnvelope(
            interface="search/v1",
            idempotency_key="idm-1",
            max_results=max_results,
            timeout_sec=5.0,
            result_char_cap=6000,
            task_id="tsk-1",
        )
        request = ToolRequest(
            interface="search/v1", query=query, max_results=max_results
        )
        # The broker calls the carriage synchronously from a pool thread; do the same.
        result: dict[str, Any] = {}

        def call() -> None:
            result["outcome"] = self.carriage(envelope, request)

        t = threading.Thread(target=call)
        t.start()
        t.join(15.0)
        return result["outcome"]

    def stop(self) -> None:
        if self._pump is not None:
            self.loopt.loop.call_soon_threadsafe(self._pump.cancel)
        if self._sidecar is not None:
            self.loopt.submit(self._sidecar.stop()).result(5.0)
        self.loopt.stop()


def test_forward_dial_egresses_at_the_remote_sidecar() -> None:
    harness = _Harness(directly_routable=True)
    harness.start()
    try:
        outcome = harness.run_search(QUERY)
    finally:
        harness.stop()
    assert outcome.status.value == "success"
    assert QUERY in outcome.value
    assert harness.provider.calls == [QUERY]


def test_reverse_rendezvous_fallback_egresses_at_the_remote_sidecar() -> None:
    harness = _Harness(directly_routable=False)
    harness.start()
    try:
        outcome = harness.run_search(QUERY)
    finally:
        harness.stop()
    assert outcome.status.value == "success"
    assert QUERY in outcome.value
    assert harness.provider.calls == [QUERY]


def _assert_credential_absent(harness: _Harness, outcome: Any) -> None:
    # The secret lives only in the sidecar's local provider; it never enters a node
    # command payload, the returned outcome, or any relay stream frame.
    blob = repr(harness.exec_payloads) + outcome.model_dump_json()
    for stream in harness.redis._streams.values():  # type: ignore[attr-defined]
        for _id, fields in stream:
            blob += repr(fields)
    assert SECRET not in blob
    assert harness.provider.api_key == SECRET


def test_keyed_credential_never_enters_the_fabric() -> None:
    harness = _Harness(directly_routable=True, api_key=SECRET)
    harness.start()
    try:
        outcome = harness.run_search(QUERY)
    finally:
        harness.stop()
    assert outcome.status.value == "success"
    _assert_credential_absent(harness, outcome)


def test_keyed_credential_absent_from_reverse_rendezvous_frames() -> None:
    # Force the reverse-rendezvous path so the operation frame travels the redis relay
    # streams, then prove the secret is absent from those captured on-wire frames.
    harness = _Harness(directly_routable=False, api_key=SECRET)
    harness.start()
    try:
        outcome = harness.run_search(QUERY)
    finally:
        harness.stop()
    assert outcome.status.value == "success"
    _assert_credential_absent(harness, outcome)
