"""The deputy probes a route over the listeners and wire production transports use."""

import asyncio
import socket
from typing import Any, cast

import pytest

from server.network.deputy import run_echo
from server.network.state import (
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    RouteObservationOutcome,
    Transport,
)
from server.supervisor.services.peer_listener import NodePeerListener
from shared.network.frame_stream import (
    MAX_PROBE_BYTES,
    FrameStreamError,
    ProbeFrame,
    read_relay_frame,
    read_stream_frame,
    write_probe,
    write_relay_frame,
)
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from tests.support.certs import new_ca
from worker.resident.peer_listener import ResidentPeerListener


class _Bridge:
    """Records what reaches the node's uplink."""

    def __init__(self) -> None:
        self.frames: list[RelayFrame] = []
        self.bound: list[str] = []

    def bind_peer(self, session_id: str, sink: Any) -> None:
        self.bound.append(session_id)

    def release_peer(self, session_id: str) -> None:
        pass

    async def on_frame(self, frame: RelayFrame) -> None:
        self.frames.append(frame)


def _mtls(ca: Any, identity: str, *sans: str) -> MutualTlsMaterial:
    issued = ca.issue(identity, *sans)
    return MutualTlsMaterial.from_b64(
        ca_b64=ca.ca_b64, cert_b64=issued.cert_b64, key_b64=issued.key_b64
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _route(*candidates: tuple[Transport, str]) -> ResolvedRoute:
    return ResolvedRoute(
        origin_id="rog-1",
        target_node_id="nde-1",
        listener_generation=0,
        route_epoch=1,
        candidates=tuple(
            RouteCandidate(
                transport=transport,
                hops=(RouteHop(transport=transport, endpoint=endpoint),),
            )
            for transport, endpoint in candidates
        ),
    )


async def _node_listener(
    material: MutualTlsMaterial | None, bridge: _Bridge
) -> NodePeerListener:
    listener = NodePeerListener(
        endpoint=f"127.0.0.1:{_free_port()}",
        material=material,
        bridge=cast(Any, bridge),
    )
    await listener.start()
    return listener


@pytest.mark.parametrize("mtls", [True, False])
def test_node_relay_is_verified_by_the_node_peer_listener(mtls: bool) -> None:
    ca = new_ca()
    bridge = _Bridge()

    async def run() -> Any:
        listener = await _node_listener(
            _mtls(ca, "node-target", "127.0.0.1") if mtls else None, bridge
        )
        try:
            context = (
                client_context(_mtls(ca, "node-origin", "127.0.0.1")) if mtls else None
            )
            return await run_echo(
                _route((Transport.NODE_RELAY, f"127.0.0.1:{listener.port}")),
                b"ping",
                connect_budget_sec=2.0,
                ssl_context=context,
            )
        finally:
            await listener.stop()

    outcome = asyncio.run(run())

    assert outcome.selected_transport is Transport.NODE_RELAY
    assert outcome.echoed == b"ping"
    assert outcome.observations == [
        (Transport.NODE_RELAY, RouteObservationOutcome.VERIFIED)
    ]
    assert bridge.frames == [] and bridge.bound == []


def test_worker_direct_is_verified_by_the_worker_peer_listener() -> None:
    ca = new_ca()
    delivered: list[RelayFrame] = []

    async def deliver(frame: RelayFrame, sink: Any) -> None:
        delivered.append(frame)

    async def run() -> Any:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        listener = ResidentPeerListener(
            sock=sock,
            material=_mtls(ca, "wkr-target", "127.0.0.1"),
            deliver=cast(Any, deliver),
        )
        await listener.start()
        try:
            return await run_echo(
                _route((Transport.WORKER_DIRECT, f"127.0.0.1:{listener.port}")),
                b"direct",
                connect_budget_sec=2.0,
                ssl_context=client_context(_mtls(ca, "node-origin", "127.0.0.1")),
            )
        finally:
            await listener.stop()

    outcome = asyncio.run(run())

    assert outcome.selected_transport is Transport.WORKER_DIRECT
    assert outcome.echoed == b"direct"
    assert delivered == []


def test_a_dead_direct_route_falls_over_to_node_relay() -> None:
    async def run() -> Any:
        listener = await _node_listener(None, _Bridge())
        try:
            return await run_echo(
                _route(
                    (Transport.WORKER_DIRECT, f"127.0.0.1:{_free_port()}"),
                    (Transport.NODE_RELAY, f"127.0.0.1:{listener.port}"),
                ),
                b"failover",
                connect_budget_sec=2.0,
            )
        finally:
            await listener.stop()

    outcome = asyncio.run(run())

    assert outcome.selected_transport is Transport.NODE_RELAY
    assert outcome.observations[0] == (
        Transport.WORKER_DIRECT,
        RouteObservationOutcome.CONNECT_FAILURE,
    )


def test_a_target_certificate_not_covering_the_dialed_host_is_a_tls_failure() -> None:
    ca = new_ca()

    async def run() -> Any:
        listener = await _node_listener(_mtls(ca, "node-elsewhere"), _Bridge())
        try:
            return await run_echo(
                _route((Transport.NODE_RELAY, f"127.0.0.1:{listener.port}")),
                b"ping",
                connect_budget_sec=2.0,
                ssl_context=client_context(_mtls(ca, "node-origin", "127.0.0.1")),
            )
        finally:
            await listener.stop()

    outcome = asyncio.run(run())

    assert outcome.observations == [
        (Transport.NODE_RELAY, RouteObservationOutcome.TLS_FAILURE)
    ]


def test_a_listener_refusing_the_deputy_identity_is_not_a_path_failure() -> None:
    ca, other = new_ca(), new_ca("another-ca")

    async def run() -> Any:
        listener = await _node_listener(
            _mtls(ca, "node-target", "127.0.0.1"), _Bridge()
        )
        try:
            # The deputy trusts the target's CA but presents an identity it never
            # issued, so the target refuses it after the dialer's side completed.
            origin = _mtls(other, "node-origin", "127.0.0.1")
            context = client_context(
                MutualTlsMaterial(
                    ca_pem=_mtls(ca, "x").ca_pem,
                    cert_pem=origin.cert_pem,
                    key_pem=origin.key_pem,
                )
            )
            return await run_echo(
                _route((Transport.NODE_RELAY, f"127.0.0.1:{listener.port}")),
                b"ping",
                connect_budget_sec=2.0,
                ssl_context=context,
            )
        finally:
            await listener.stop()

    outcome = asyncio.run(run())

    assert outcome.observations == [
        (Transport.NODE_RELAY, RouteObservationOutcome.APPLICATION_ERROR)
    ]


async def _serve_once(handler: Any) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def test_a_listener_that_predates_probes_is_not_a_path_failure() -> None:
    async def run() -> Any:
        async def old_listener(reader, writer) -> None:
            # A listener that reads only relay frames closes on a header it cannot
            # decode.
            with pytest.raises(FrameStreamError):
                await read_relay_frame(reader)
            writer.close()

        server, port = await _serve_once(old_listener)
        async with server:
            return await run_echo(
                _route((Transport.NODE_RELAY, f"127.0.0.1:{port}")),
                b"ping",
                connect_budget_sec=2.0,
            )

    outcome = asyncio.run(run())

    assert outcome.selected_transport is None
    assert outcome.observations == [
        (Transport.NODE_RELAY, RouteObservationOutcome.APPLICATION_ERROR)
    ]


def test_an_answer_carrying_other_bytes_is_a_path_failure() -> None:
    async def run() -> Any:
        async def corrupting(reader, writer) -> None:
            probe = await read_stream_frame(reader)
            assert isinstance(probe, ProbeFrame)
            await write_probe(writer, probe.payload[::-1])
            writer.close()

        server, port = await _serve_once(corrupting)
        async with server:
            return await run_echo(
                _route((Transport.NODE_RELAY, f"127.0.0.1:{port}")),
                b"ping",
                connect_budget_sec=2.0,
            )

    outcome = asyncio.run(run())

    assert outcome.observations == [
        (Transport.NODE_RELAY, RouteObservationOutcome.ROUTE_FAILURE)
    ]


def test_an_unanswered_probe_is_a_timeout() -> None:
    async def run() -> Any:
        async def silent(reader, writer) -> None:
            await reader.read()
            writer.close()

        server, port = await _serve_once(silent)
        async with server:
            return await run_echo(
                _route((Transport.NODE_RELAY, f"127.0.0.1:{port}")),
                b"ping",
                connect_budget_sec=0.3,
            )

    outcome = asyncio.run(run())

    assert outcome.observations == [
        (Transport.NODE_RELAY, RouteObservationOutcome.TIMEOUT)
    ]


def _data(seq: int) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=seq,
        payload=b"body",
    )


def test_the_listener_answers_one_probe_and_closes() -> None:
    bridge = _Bridge()

    async def run() -> tuple[Any, bytes]:
        listener = await _node_listener(None, bridge)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", listener.port)
            await write_probe(writer, b"one")
            await write_probe(writer, b"two")
            await write_relay_frame(writer, _data(1))
            answer = await read_stream_frame(reader)
            rest = await asyncio.wait_for(reader.read(), timeout=2.0)
            writer.close()
            return answer, rest
        finally:
            await listener.stop()

    answer, rest = asyncio.run(run())

    assert answer == ProbeFrame(b"one")
    assert rest == b""
    assert bridge.frames == []


def test_a_probe_after_a_session_frame_closes_the_connection() -> None:
    bridge = _Bridge()

    async def run() -> bytes:
        listener = await _node_listener(None, bridge)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", listener.port)
            await write_relay_frame(writer, _data(1))
            await write_probe(writer, b"late")
            rest = await asyncio.wait_for(reader.read(), timeout=2.0)
            writer.close()
            return rest
        finally:
            await listener.stop()

    assert asyncio.run(run()) == b""
    assert [frame.seq for frame in bridge.frames] == [1]


def test_an_oversized_probe_is_refused() -> None:
    async def run() -> bytes:
        listener = await _node_listener(None, _Bridge())
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", listener.port)
            meta = b'{"kind":"probe"}'
            body = b"x" * (MAX_PROBE_BYTES + 1)
            writer.write(
                len(meta).to_bytes(4, "big")
                + meta
                + len(body).to_bytes(4, "big")
                + body
            )
            await writer.drain()
            rest = await asyncio.wait_for(reader.read(), timeout=2.0)
            writer.close()
            return rest
        finally:
            await listener.stop()

    assert asyncio.run(run()) == b""
    with pytest.raises(FrameStreamError):
        asyncio.run(write_probe(cast(Any, None), b"x" * (MAX_PROBE_BYTES + 1)))


def test_a_probe_is_refused_by_every_relay_codec() -> None:
    wire = _data(1).to_wire() | {"kind": "probe"}
    with pytest.raises(ValueError):
        RelayFrame.from_wire(wire)
    fields = _data(1).to_fields() | {b"k": b"probe"}
    with pytest.raises(ValueError):
        RelayFrame.from_fields(fields)

    async def read() -> None:
        reader = asyncio.StreamReader()
        meta = b'{"kind":"probe"}'
        reader.feed_data(len(meta).to_bytes(4, "big") + meta + (2).to_bytes(4, "big"))
        reader.feed_data(b"hi")
        await read_relay_frame(reader)

    with pytest.raises(FrameStreamError):
        asyncio.run(read())
