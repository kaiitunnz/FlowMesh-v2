"""Bounded relay session plus the deputy driving the transport ladder end to end."""

import asyncio
import socket

from server.network.deputy import run_echo
from server.network.listeners import NetworkControlRelay, NetworkPlaneListeners
from server.network.relay import RelaySession
from server.network.state import (
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    RouteObservationOutcome,
    Transport,
)
from server.network.wire import APP_ERROR_SENTINEL


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _candidate(transport: Transport, *endpoints: str) -> RouteCandidate:
    hops = tuple(
        RouteHop(transport=transport, endpoint=endpoint) for endpoint in endpoints
    )
    return RouteCandidate(transport=transport, hops=hops)


def _route(*candidates: RouteCandidate) -> ResolvedRoute:
    return ResolvedRoute(
        origin_id="rog-1",
        target_node_id="nde-1",
        listener_generation=0,
        route_epoch=1,
        candidates=candidates,
    )


def test_relay_session_round_trips_and_bounds_buffer() -> None:
    async def run() -> bytes:
        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            while data := await reader.read(1024):
                writer.write(data)
                await writer.drain()
            writer.close()

        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        caller_reader, caller_writer = await asyncio.open_connection("127.0.0.1", port)
        target_reader, target_writer = await asyncio.open_connection("127.0.0.1", port)
        session = RelaySession("rly-test", buffer_bytes=32768)
        task = asyncio.create_task(
            session.relay(caller_reader, caller_writer, target_reader, target_writer)
        )
        caller_writer.write(b"relay-me")
        await caller_writer.drain()
        got = await asyncio.wait_for(caller_reader.readexactly(8), 2.0)
        session.cancel()
        await asyncio.gather(task, return_exceptions=True)
        server.close()
        await server.wait_closed()
        return got

    assert asyncio.run(run()) == b"relay-me"


class _Fixture:
    def __init__(self) -> None:
        self.sidecar_port = _free_port()
        self.relay_port = _free_port()
        self.control_port = _free_port()
        self.listeners = NetworkPlaneListeners(
            sidecar_url=f"127.0.0.1:{self.sidecar_port}",
            endpoint_url=f"127.0.0.1:{self.relay_port}",
            buffer_bytes=65536,
        )
        self.control = NetworkControlRelay(
            control_relay_url=f"127.0.0.1:{self.control_port}", buffer_bytes=65536
        )

    async def __aenter__(self) -> "_Fixture":
        await self.listeners.start()
        await self.control.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.listeners.stop()
        await self.control.stop()


def test_worker_direct_round_trip() -> None:
    async def run():
        async with _Fixture() as fx:
            route = _route(
                _candidate(Transport.WORKER_DIRECT, f"127.0.0.1:{fx.sidecar_port}")
            )
            return await run_echo(route, b"direct", connect_budget_sec=3.0)

    outcome = asyncio.run(run())
    assert outcome.selected_transport is Transport.WORKER_DIRECT
    assert outcome.echoed == b"direct"


def test_node_relay_round_trip_through_session() -> None:
    async def run():
        async with _Fixture() as fx:
            route = _route(
                _candidate(Transport.NODE_RELAY, f"127.0.0.1:{fx.relay_port}")
            )
            return await run_echo(route, b"relayed", connect_budget_sec=3.0)

    outcome = asyncio.run(run())
    assert outcome.selected_transport is Transport.NODE_RELAY
    assert outcome.echoed == b"relayed"


def test_control_relay_round_trip() -> None:
    async def run():
        async with _Fixture() as fx:
            route = _route(
                _candidate(
                    Transport.CONTROL_RELAY,
                    f"127.0.0.1:{fx.control_port}",
                    f"127.0.0.1:{fx.sidecar_port}",
                )
            )
            return await run_echo(route, b"controlled", connect_budget_sec=3.0)

    outcome = asyncio.run(run())
    assert outcome.selected_transport is Transport.CONTROL_RELAY
    assert outcome.echoed == b"controlled"


def test_connect_failure_falls_to_next_candidate() -> None:
    async def run():
        async with _Fixture() as fx:
            dead = _free_port()
            route = _route(
                _candidate(Transport.WORKER_DIRECT, f"127.0.0.1:{dead}"),
                _candidate(Transport.NODE_RELAY, f"127.0.0.1:{fx.relay_port}"),
            )
            return await run_echo(route, b"failover", connect_budget_sec=3.0)

    outcome = asyncio.run(run())
    assert outcome.selected_transport is Transport.NODE_RELAY
    assert outcome.observations[0] == (
        Transport.WORKER_DIRECT,
        RouteObservationOutcome.CONNECT_FAILURE,
    )


def test_application_error_is_not_a_path_failure() -> None:
    async def run():
        async with _Fixture() as fx:
            route = _route(
                _candidate(Transport.WORKER_DIRECT, f"127.0.0.1:{fx.sidecar_port}")
            )
            return await run_echo(route, APP_ERROR_SENTINEL, connect_budget_sec=3.0)

    outcome = asyncio.run(run())
    assert outcome.selected_transport is None
    assert outcome.observations == [
        (Transport.WORKER_DIRECT, RouteObservationOutcome.APPLICATION_ERROR)
    ]
