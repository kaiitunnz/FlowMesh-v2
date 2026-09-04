"""Unit coverage for the tool target registry and the carriage's outcome decoding.

Exercises the target bind/cache and the carriage's ``_deliver`` decode paths with fakes,
so a reject or a lost reply becomes a typed unavailable outcome (never a success) and no
demoting observation is recorded, without standing up the full relay harness.
"""

import asyncio
import threading
from typing import Any

from server.config import WebSearchConfig
from server.network import wire as netwire
from server.network.reachability import NetworkReachabilityView, is_demoting
from server.network.state import (
    NetworkEndpointAdvertisement,
    NonresidentSidecarTarget,
    PolicyClass,
    ReachabilityClass,
    ReachabilityState,
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    RouteObservation,
    RouteObservationOutcome,
    Transport,
)
from server.orchestration.tool_dispatch import ToolOutcome
from server.tools import tool_sidecar_wire as wire
from server.tools.fabric_tool_broker import FabricToolBroker
from server.tools.tool_carriage import (
    RemoteSidecarCarriage,
    ToolEgressOriginDeputy,
    ToolTargetRegistry,
)
from server.tools.tool_egress import (
    AmbiguousDelivery,
    ToolOperationEnvelope,
    ToolRequest,
)
from server.tools.tool_relay_delivery import ToolRelayEndpoint
from shared.schemas.command import CommandType
from tests.server.network._relay_fakes import FakeBinaryRedis

TARGET_NODE = "nde-t"
TARGET_WORKER = "wrk-1"
TARGET_INCARNATION = 7


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
    """Returns a canned reply, the observations it would have recorded, and egress."""

    def __init__(
        self,
        reply: bytes | None,
        observations: list[tuple[Transport, RouteObservationOutcome]] | None = None,
        egressed: bool = False,
    ) -> None:
        self.reply = reply
        self.observations = observations or []
        self.egressed = egressed
        self.cancelled: list[str] = []

    async def deliver(self, session_id: str, route: Any, **kw: Any) -> Any:
        return self.reply, self.observations, self.egressed

    async def cancel(self, session_id: str) -> None:
        self.cancelled.append(session_id)


def _carriage(deputy: _FakeDeputy, exec_calls: list[str]) -> RemoteSidecarCarriage:
    async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
        exec_calls.append(command.value)
        return {"host": "127.0.0.1", "port": 9999}

    async def resolve_target(task_id: str) -> tuple[str, str, int]:
        return TARGET_NODE, TARGET_WORKER, TARGET_INCARNATION

    async def endpoint_provider(node_id: str) -> NetworkEndpointAdvertisement:
        return _target_endpoint(node_id)

    registry = ToolTargetRegistry(
        exec_node_cmd=exec_cmd,
        resolve_target=resolve_target,
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


def _fast_carriage(deputy: Any) -> RemoteSidecarCarriage:
    async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
        return {"host": "127.0.0.1", "port": 9999}

    async def resolve_target(task_id: str) -> tuple[str, str, int]:
        return TARGET_NODE, TARGET_WORKER, TARGET_INCARNATION

    async def endpoint_provider(node_id: str) -> NetworkEndpointAdvertisement:
        return _target_endpoint(node_id)

    registry = ToolTargetRegistry(
        exec_node_cmd=exec_cmd,
        resolve_target=resolve_target,
        sidecar_route="127.0.0.1:0",
        provider="fake",
        interfaces=("search/v1",),
        directly_routable=True,
    )
    return RemoteSidecarCarriage(
        origin_deputy=deputy,
        registry=registry,
        endpoint_provider=endpoint_provider,
        ingress_endpoint=_ingress(),
        provider="fake",
        connect_budget_sec=0.05,
        outer_margin_sec=0.1,
    )


def _run(carriage: RemoteSidecarCarriage) -> Any:
    envelope = ToolOperationEnvelope(
        interface="search/v1",
        idempotency_key="idm-1",
        max_results=3,
        timeout_sec=5.0,
        result_char_cap=6000,
        task_id="tsk-1",
    )
    request = ToolRequest(interface="search/v1", query="q", max_results=3)
    return asyncio.run(carriage._deliver(envelope, request, "xtr-test"))


def test_registry_binds_once_and_caches() -> None:
    async def run() -> None:
        calls: list[str] = []

        async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
            calls.append(command.value)
            return {"host": "127.0.0.1", "port": 1234}

        async def resolve_target(task_id: str) -> tuple[str, str, int]:
            return TARGET_NODE, TARGET_WORKER, TARGET_INCARNATION

        registry = ToolTargetRegistry(
            exec_node_cmd=exec_cmd,
            resolve_target=resolve_target,
            sidecar_route="127.0.0.1:0",
            provider="fake",
            interfaces=("search/v1",),
            directly_routable=True,
        )
        first = await registry.ensure_target("tsk-1")
        second = await registry.ensure_target("tsk-1")
        assert isinstance(first, NonresidentSidecarTarget)
        assert first is second
        assert calls == [CommandType.BIND_TOOL_SIDECAR.value]

    asyncio.run(run())


def test_registry_returns_none_when_no_node() -> None:
    async def run() -> None:
        async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
            raise AssertionError("must not bind when no node is selected")

        async def resolve_none(task_id: str) -> None:
            return None

        registry = ToolTargetRegistry(
            exec_node_cmd=exec_cmd,
            resolve_target=resolve_none,
            sidecar_route="127.0.0.1:0",
            provider="fake",
            interfaces=("search/v1",),
            directly_routable=True,
        )
        assert await registry.ensure_target("tsk-1") is None

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


class _HangDeputy:
    """A deputy whose delivery never returns, forcing the carriage's outer bound."""

    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def deliver(self, session_id: str, route: Any, **kw: Any) -> Any:
        await asyncio.Event().wait()

    async def cancel(self, session_id: str) -> None:
        self.cancelled.append(session_id)


def test_outer_timeout_holds_ambiguous_and_aborts() -> None:
    # A delivery that outruns the outer bound is a nonterminal ambiguous loss, never a
    # terminal outcome, and the carriage fires a best-effort cancel to reap the session.
    deputy = _HangDeputy()
    carriage = _fast_carriage(deputy)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    carriage.bind_loop(loop)
    envelope = ToolOperationEnvelope(
        interface="search/v1",
        idempotency_key="idm-1",
        max_results=3,
        timeout_sec=0.05,
        result_char_cap=6000,
        task_id="tsk-1",
    )
    request = ToolRequest(interface="search/v1", query="q", max_results=3)
    try:
        result = carriage(envelope, request)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2.0)
    assert isinstance(result, AmbiguousDelivery)
    assert len(deputy.cancelled) == 1


def test_lost_reply_after_egress_is_ambiguous() -> None:
    # The operation reached a wire but no reply came back: a nonterminal ambiguous
    # delivery the control path holds pending, never a terminal outcome.
    result = _run(_carriage(_FakeDeputy(None, egressed=True), []))
    assert isinstance(result, AmbiguousDelivery)


def test_lost_reply_without_egress_is_terminal_unavailable() -> None:
    # The operation never reached a wire (a pre-delivery failure): safe to terminalize.
    outcome = _run(_carriage(_FakeDeputy(None, egressed=False), []))
    assert isinstance(outcome, ToolOutcome)
    assert outcome.status.value == "unavailable"


def test_target_cache_is_bounded_and_evicts_lru() -> None:
    async def run() -> None:
        async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
            return {"host": "127.0.0.1", "port": 1}

        async def resolve_target(task_id: str) -> tuple[str, str, int]:
            # A distinct worker per task, so each ensure_target binds a new entry.
            return TARGET_NODE, f"wkr-{task_id}", TARGET_INCARNATION

        registry = ToolTargetRegistry(
            exec_node_cmd=exec_cmd,
            resolve_target=resolve_target,
            sidecar_route="127.0.0.1:0",
            provider="fake",
            interfaces=("search/v1",),
            directly_routable=True,
            max_cached_targets=2,
        )
        await registry.ensure_target("a")
        await registry.ensure_target("b")
        # Re-touch "a" so "b" becomes the least-recently-used entry.
        await registry.ensure_target("a")
        await registry.ensure_target("c")  # over the cap: evicts the LRU ("wkr-b")
        assert set(registry._targets) == {"wkr-a", "wkr-c"}

    asyncio.run(run())


def test_unresolvable_target_is_ambiguous_not_terminal() -> None:
    # The episode's assigned worker is not resolvable (gone or mid-reassignment): a
    # transient condition held pending for re-drive, never a spurious terminal outcome.
    async def resolve_none(task_id: str) -> None:
        return None

    async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
        raise AssertionError("must not bind when the target is unresolvable")

    async def endpoint_provider(node_id: str) -> NetworkEndpointAdvertisement:
        return _target_endpoint(node_id)

    registry = ToolTargetRegistry(
        exec_node_cmd=exec_cmd,
        resolve_target=resolve_none,
        sidecar_route="127.0.0.1:0",
        provider="fake",
        interfaces=("search/v1",),
        directly_routable=True,
    )
    carriage = RemoteSidecarCarriage(
        origin_deputy=_FakeDeputy(None),  # type: ignore[arg-type]
        registry=registry,
        endpoint_provider=endpoint_provider,
        ingress_endpoint=_ingress(),
        provider="fake",
    )
    assert isinstance(_run(carriage), AmbiguousDelivery)


def test_registry_invalidate_rebinds_and_unbinds_the_stale_sidecar() -> None:
    async def run() -> None:
        cmds: list[tuple[str, str]] = []

        async def exec_cmd(node_id: str, command: CommandType, payload: dict) -> dict:
            cmds.append((command.value, str(payload["target_id"])))
            return {"host": "127.0.0.1", "port": 1}

        async def resolve_target(task_id: str) -> tuple[str, str, int]:
            return TARGET_NODE, TARGET_WORKER, TARGET_INCARNATION

        registry = ToolTargetRegistry(
            exec_node_cmd=exec_cmd,
            resolve_target=resolve_target,
            sidecar_route="127.0.0.1:0",
            provider="fake",
            interfaces=("search/v1",),
            directly_routable=True,
        )
        first = await registry.ensure_target("tsk-1")
        assert first is not None
        await registry.invalidate(first)
        second = await registry.ensure_target("tsk-1")
        assert second is not None
        assert first.target_generation == second.target_generation == TARGET_INCARNATION
        assert first.target_id == second.target_id == TARGET_WORKER
        # Bind, then unbind the stale target on invalidate, then rebind a fresh one.
        assert cmds == [
            (CommandType.BIND_TOOL_SIDECAR.value, first.target_id),
            (CommandType.UNBIND_TOOL_SIDECAR.value, first.target_id),
            (CommandType.BIND_TOOL_SIDECAR.value, second.target_id),
        ]

    asyncio.run(run())


def test_broker_build_defers_a_missing_provider_credential() -> None:
    # A keyed provider with no key must not fail broker build: the provider is built
    # lazily, so a deployment that egresses only off-server never builds it server-side.
    config = WebSearchConfig(provider="serper", api_key=None)
    broker = FabricToolBroker.build(config, lambda _t, _c, _v: None)
    broker.shutdown()


def test_forward_dial_reply_read_uses_the_operation_deadline() -> None:
    async def run() -> None:
        # A loopback sidecar that answers only after a delay far over the connect
        # budget: the reply read must use the operation deadline, not that budget.
        async def handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await netwire.read_frame(reader)
            await asyncio.sleep(0.3)
            await netwire.write_frame(writer, b"late-reply")
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        deputy = ToolEgressOriginDeputy(
            relay_endpoint=ToolRelayEndpoint(FakeBinaryRedis(), "nde-o"),
            connect_budget_sec=0.1,
        )
        route = ResolvedRoute(
            origin_id="rog-1",
            target_node_id=TARGET_NODE,
            listener_generation=1,
            route_epoch=1,
            candidates=(
                RouteCandidate(
                    transport=Transport.WORKER_DIRECT,
                    hops=(
                        RouteHop(
                            transport=Transport.WORKER_DIRECT,
                            endpoint=f"{host}:{port}",
                            node_id=TARGET_NODE,
                        ),
                    ),
                ),
            ),
        )
        try:
            reply, obs, egressed = await deputy.deliver(
                "xtr-1",
                route,
                invocation_id="i",
                idm="m",
                operation_payload=b"op",
                read_deadline_sec=2.0,
            )
        finally:
            server.close()
            await server.wait_closed()
        assert reply == b"late-reply"
        assert obs == [(Transport.WORKER_DIRECT, RouteObservationOutcome.VERIFIED)]
        assert egressed is False

    asyncio.run(run())


def test_forward_dial_read_timeout_is_ambiguous_and_non_demoting() -> None:
    async def run() -> None:
        # The connect succeeds and the op is written, but the sidecar never replies
        # within the read deadline: an ambiguous post-send loss, classified as a
        # non-demoting application error so provider silence never demotes the path.
        async def handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            # Read the op, then hold the connection open without replying, so the client
            # read fires a real timeout rather than seeing an early close.
            await netwire.read_frame(reader)
            await asyncio.sleep(1.0)

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        deputy = ToolEgressOriginDeputy(
            relay_endpoint=ToolRelayEndpoint(FakeBinaryRedis(), "nde-o"),
            connect_budget_sec=1.0,
        )
        route = ResolvedRoute(
            origin_id="rog-1",
            target_node_id=TARGET_NODE,
            listener_generation=1,
            route_epoch=1,
            candidates=(
                RouteCandidate(
                    transport=Transport.WORKER_DIRECT,
                    hops=(
                        RouteHop(
                            transport=Transport.WORKER_DIRECT,
                            endpoint=f"{host}:{port}",
                            node_id=TARGET_NODE,
                        ),
                    ),
                ),
            ),
        )
        try:
            reply, obs, egressed = await deputy.deliver(
                "xtr-1",
                route,
                invocation_id="i",
                idm="m",
                operation_payload=b"op",
                read_deadline_sec=0.2,
            )
        finally:
            server.close()
        assert reply is None
        assert egressed is True
        assert obs == [
            (Transport.WORKER_DIRECT, RouteObservationOutcome.APPLICATION_ERROR)
        ]
        # The classified outcome does not demote reachability: feeding it to a view
        # keeps the path usable, so a slow or silent provider never marks it down.
        assert is_demoting(RouteObservationOutcome.APPLICATION_ERROR) is False
        view = NetworkReachabilityView()
        now = 100.0
        view.observe(
            RouteObservation(
                origin_id="rog-1",
                policy_class=PolicyClass.DEFAULT,
                target_node_id=TARGET_NODE,
                incarnation=1,
                listener_generation=1,
                transport=Transport.WORKER_DIRECT,
                outcome=RouteObservationOutcome.APPLICATION_ERROR,
            ),
            now=now,
        )
        assert (
            view.state_for(
                "rog-1",
                PolicyClass.DEFAULT,
                TARGET_NODE,
                1,
                1,
                Transport.WORKER_DIRECT,
                now=now,
            )
            is not ReachabilityState.DEMOTED
        )

    asyncio.run(run())
