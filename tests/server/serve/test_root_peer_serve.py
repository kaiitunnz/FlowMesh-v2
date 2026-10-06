"""The root's gated serve origin dials a trusted target itself.

The root drives the serve origin over a socket it opens to the replica worker's
claim-gated listener or to its node's peer listener, with its own node identity, so the
request and response never enter the rendezvous. A dial that fails before any frame is
delivered completes on the relay under the same session; a loss after delivery is
uncertain and never replays on the relay. Each attempt's socket and reader end with the
attempt.
"""

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

import fakeredis
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from server.network.rendezvous import RootCursorStore, RootRendezvousBridge
from server.network.reverse_relay import (
    RESIDENT_RELAY_KEYSPACE,
    BinaryRedis,
    RelaySessionStore,
    RelayStreamStore,
)
from server.network.worker_bridge import RelayWorkerBridge
from server.serve.relay import SERVE_EDGE_STREAM_ID, ServeRelayExecutor
from server.supervisor.services.peer_listener import NodePeerListener
from server.supervisor.services.reverse_relay_attachment import (
    ReverseRelayAttachment,
)
from server.telemetry.tracing import format_traceparent
from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.relay_frame import RelayFrame
from shared.resident.carriage import ResidentCarriagePlan
from shared.resident.contracts import (
    AdmissionHandoff,
    ReplicaEndpoint,
    RouteAuthorization,
)
from shared.resident.envelope import ServeRequestEnvelope, freeze_request_envelope
from shared.resident.peer_carriage import PeerCarriage, PeerDialer
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentRouteObservation,
    ResidentStreamChunk,
    ResidentStreamHead,
    ResidentStreamStatus,
)
from shared.schemas.network import RouteObservationOutcome, Transport
from shared.telemetry.semconv import PHYSICAL_TRANSPORT
from tests.support.certs import TestCa as _TestCa
from tests.support.certs import new_ca
from worker.resident.engine import EngineResponse, RawEngineResponse
from worker.resident.peer_listener import ResidentPeerListener
from worker.resident.replica_sidecar import ResidentReplicaSidecar

_SERVE_TASK = "tsk-serve"
_BODY = b'{"messages":[{"role":"user","content":"hi"}],"stream":true}'
_PARTS = [b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"]
_DEADLINE = 10.0
# The origin notices a silent loss at its stream deadline, well inside the wait bound.
_STREAM_DEADLINE = 2.0


def _envelope() -> ServeRequestEnvelope:
    return freeze_request_envelope(
        method="POST",
        upstream_path="v1/chat/completions",
        query="",
        headers=[("content-type", "application/json")],
        body=_BODY,
    )


def _handoff(envelope: ServeRequestEnvelope) -> AdmissionHandoff:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family=f"serve/{_SERVE_TASK}",
        tenant="t1",
        origin_id="rog-root",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
        serve_task_id=_SERVE_TASK,
        binding_generation=0,
        descriptor_digest=envelope.digest(),
    )


def _auth() -> RouteAuthorization:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        tenant="t1",
        origin_id="rog-root",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
        serve_task_id=_SERVE_TASK,
        binding_generation=0,
    )


async def _unused_engine(
    endpoint: ReplicaEndpoint, request: str | None, *args: Any
) -> EngineResponse:
    raise AssertionError("a serve request never reaches the parsed engine path")


class _RawEngine:
    """The replica's engine: streams fixed parts, optionally pausing between them."""

    def __init__(self, pause: asyncio.Event | None = None) -> None:
        self.calls = 0
        self._pause = pause

    async def __call__(
        self, endpoint: ReplicaEndpoint, envelope: ServeRequestEnvelope
    ) -> RawEngineResponse:
        self.calls += 1
        pause = self._pause

        async def chunks() -> AsyncIterator[bytes]:
            for index, part in enumerate(_PARTS):
                if index == 1 and pause is not None:
                    await pause.wait()
                yield part

        async def aclose() -> None:
            return None

        return RawEngineResponse(
            status=200,
            headers=(("content-type", "text/event-stream"),),
            chunks=chunks(),
            aclose=aclose,
        )


class _Control:
    """Stands in for resident control: authorizes on the ack, records the rest."""

    def __init__(self) -> None:
        self.executor: ServeRelayExecutor | None = None
        self.acks: list[ResidentBootstrapAck] = []
        self.heads: list[ResidentStreamHead] = []
        self.chunks: list[bytes] = []
        self.outcomes: list[ResidentOpOutcome] = []
        self.observations: list[ResidentRouteObservation] = []
        self.done = asyncio.Event()
        self.streaming = asyncio.Event()

    def on_bootstrap_ack(self, ack: ResidentBootstrapAck) -> None:
        self.acks.append(ack)
        if ack.outcome is ResidentBootstrapOutcome.ACKED:
            assert self.executor is not None
            self.executor.authorize(ack.session_id, _auth())
        else:
            self.done.set()

    def on_stream_head(self, head: ResidentStreamHead) -> None:
        self.heads.append(head)

    def on_stream_chunk(self, chunk: ResidentStreamChunk) -> None:
        self.chunks.append(chunk.payload)
        self.streaming.set()

    def on_outcome(self, outcome: ResidentOpOutcome) -> None:
        self.outcomes.append(outcome)
        self.done.set()

    def on_route_observation(self, observation: ResidentRouteObservation) -> None:
        self.observations.append(observation)


async def _until(predicate: Callable[[], bool]) -> None:
    """Wait for a condition the attempt reaches asynchronously, within the deadline."""
    async with asyncio.timeout(_DEADLINE):
        while not predicate():
            await asyncio.sleep(0.02)


def _bound_socket() -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock, sock.getsockname()[1]


def _closed_port() -> int:
    sock, port = _bound_socket()
    sock.close()
    return port


class _Harness:
    """The root executor over a fake rendezvous, and a replica worker's listener."""

    def __init__(
        self,
        *,
        root: MutualTlsMaterial | None,
        target: MutualTlsMaterial | None,
        engine: _RawEngine,
        tracer_provider: TracerProvider | None = None,
        max_connections: int = 64,
    ) -> None:
        self.redis = fakeredis.aioredis.FakeRedis()
        self._relay = _RelayLeg(cast(BinaryRedis, self.redis))
        self.control = _Control()
        self.engine = engine
        self.executor = ServeRelayExecutor(
            relay_redis=cast(BinaryRedis, self.redis),
            edge_id=SERVE_EDGE_STREAM_ID,
            control=self.control,
            peer=PeerDialer(
                ssl_context=client_context(root) if root is not None else None,
                connect_budget_sec=2.0,
            ),
            stream_deadline_sec=_STREAM_DEADLINE,
            tracer=(
                tracer_provider.get_tracer("test")
                if tracer_provider is not None
                else None
            ),
        )
        self.control.executor = self.executor
        self.attachment = _Attachment(self._relay.bridge)
        self.sidecar = ResidentReplicaSidecar(
            sink=self.attachment,
            engine_open=_unused_engine,
            engine_open_raw=engine,
        )
        self.sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
            serve_task_id=_SERVE_TASK,
            binding_generation=0,
        )
        sock, self.port = _bound_socket()
        self.dialed: list[RelayFrame] = []
        self.listener = ResidentPeerListener(
            sock=sock,
            material=target,
            deliver=self._dialed_frame,
            max_connections=max_connections,
        )
        self.node_listener = NodePeerListener(
            endpoint="127.0.0.1:0",
            material=target,
            bridge=cast(Any, _DialedUplink(self._relay.bridge, self.dialed)),
        )

    async def _dialed_frame(self, frame: RelayFrame, sink: Any) -> None:
        self.dialed.append(frame)
        await self.sidecar.on_frame(frame, sink)

    @property
    def open_connections(self) -> int:
        return self.listener._listener._open + self.node_listener._listener._open

    async def start(self) -> None:
        await self.listener.start()
        await self.node_listener.start()
        self.executor.start(asyncio.get_running_loop())
        await self._relay.start(self.sidecar)

    async def aclose(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.listener.stop(), timeout=5.0)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.node_listener.stop(), timeout=5.0)
        await self._relay.stop()
        await self.sidecar.aclose()
        await self.redis.aclose()

    @property
    def relayed_to_replica(self) -> list[RelayFrame]:
        """The frames the rendezvous carried to the replica's node."""
        return self._relay.delivered

    def open(
        self,
        transport: Transport,
        endpoint: str | None = None,
        traceparent: str | None = None,
    ) -> None:
        envelope = _envelope()
        if endpoint is None:
            port = (
                self.node_listener.port
                if transport is Transport.NODE_RELAY
                else self.port
            )
            endpoint = f"127.0.0.1:{port}"
        self.executor.open(
            session_id="rly-1",
            invocation_id="inv-1",
            idm="idm-1",
            task_id="inv-1",
            call_correlation="serve:inv-1",
            handoff=_handoff(envelope),
            envelope=envelope,
            plan=ResidentCarriagePlan(
                session_id="rly-1",
                selected_transport=transport.value,
                selected_endpoint=endpoint,
            ),
            traceparent=traceparent,
        )

    async def relayed_frames(self) -> int:
        """How many frames reached the rendezvous, on any stream."""
        total = 0
        for key in await self.redis.keys("*"):
            if await self.redis.type(key) == b"stream":
                total += await self.redis.xlen(key)
        return total

    async def connections_drain(self) -> int:
        with contextlib.suppress(TimeoutError):
            await _until(lambda: self.open_connections == 0)
        return self.open_connections


class _Attachment:
    """The replica worker's attachment: its frames reach the node's bridge, which
    answers a dialed session over the origin's connection and publishes any other up
    the node's stream."""

    def __init__(self, bridge: RelayWorkerBridge) -> None:
        self.frames: list[Any] = []
        self._bridge = bridge

    async def send(self, frame: Any) -> None:
        self.frames.append(frame)
        await self._bridge.publish_up(frame)


class _DialedUplink:
    """The node's bridge as its peer listener sees it, recording what a dialer sent."""

    def __init__(self, bridge: RelayWorkerBridge, dialed: list[RelayFrame]) -> None:
        self._bridge = bridge
        self._dialed = dialed

    def bind_peer(self, session_id: str, sink: Any) -> None:
        self._bridge.bind_peer(session_id, sink)

    def release_peer(self, session_id: str) -> None:
        self._bridge.release_peer(session_id)

    async def on_frame(self, frame: RelayFrame) -> None:
        self._dialed.append(frame)
        await self._bridge.on_frame(frame)


class _RelayLeg:
    """The rendezvous between the root's serve edge and the replica's node.

    The root bridge forwards each direction by the session record; the node's attachment
    hands what reaches it to the node's worker bridge, which enqueues it to the replica
    worker, as a supervisor does.
    """

    _NODE = "nde-replica"
    _WORKER = "wkr-replica"

    def __init__(self, redis: BinaryRedis) -> None:
        self._sessions = RelaySessionStore(redis, RESIDENT_RELAY_KEYSPACE)
        streams = RelayStreamStore(redis, RESIDENT_RELAY_KEYSPACE)
        self._root = RootRendezvousBridge(
            streams, self._sessions, RootCursorStore(redis, RESIDENT_RELAY_KEYSPACE)
        )
        self.bridge = RelayWorkerBridge(
            redis, self._NODE, self._enqueue, keyspace=RESIDENT_RELAY_KEYSPACE
        )
        self.delivered: list[RelayFrame] = []
        self._sidecar: ResidentReplicaSidecar | None = None
        self._node = ReverseRelayAttachment(
            redis, self._NODE, self, owner="node", keyspace=RESIDENT_RELAY_KEYSPACE
        )
        self._pump: asyncio.Task[None] | None = None

    async def on_frame(self, frame: RelayFrame) -> None:
        self.delivered.append(frame)
        await self.bridge.on_frame(frame)

    async def _enqueue(self, worker_id: str, payload: dict[str, Any]) -> bool:
        assert self._sidecar is not None and worker_id == self._WORKER
        await self._sidecar.on_frame(RelayFrame.from_wire(payload["payload"]))
        return True

    async def start(self, sidecar: ResidentReplicaSidecar) -> None:
        self._sidecar = sidecar
        await self._sessions.update(
            "rly-1",
            origin_node=SERVE_EDGE_STREAM_ID,
            target_node=self._NODE,
            target_worker=self._WORKER,
        )
        self._node.start(asyncio.get_running_loop())

        async def pump() -> None:
            while True:
                await self._root.pump_ready([SERVE_EDGE_STREAM_ID, self._NODE], 50)

        self._pump = asyncio.ensure_future(pump())

    async def stop(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump
        await self._node.stop()


def _run(
    body: Callable[[_Harness], Awaitable[None]],
    *,
    root: MutualTlsMaterial | None,
    target: MutualTlsMaterial | None,
    engine: _RawEngine | None = None,
    tracer_provider: TracerProvider | None = None,
    max_connections: int = 64,
) -> None:
    async def run() -> None:
        h = _Harness(
            root=root,
            target=target,
            engine=engine or _RawEngine(),
            tracer_provider=tracer_provider,
            max_connections=max_connections,
        )
        await h.start()
        try:
            await body(h)
        finally:
            await h.executor.stop()
            await h.aclose()

    asyncio.run(run())


@pytest.fixture(scope="module")
def ca() -> _TestCa:
    return new_ca()


@pytest.mark.parametrize("transport", [Transport.WORKER_DIRECT, Transport.NODE_RELAY])
def test_the_root_carries_a_serve_call_over_the_socket_it_dials(
    ca: _TestCa, transport: Transport
) -> None:
    async def body(h: _Harness) -> None:
        h.open(transport)
        await asyncio.wait_for(h.control.done.wait(), _DEADLINE)

        assert [o.status for o in h.control.outcomes] == [ResidentStreamStatus.SUCCESS]
        assert h.control.heads[0].status == 200
        assert b"".join(h.control.chunks) == b"".join(_PARTS)
        assert h.engine.calls == 1
        # Neither the request nor the response entered the rendezvous.
        assert await h.relayed_frames() == 0
        assert h.relayed_to_replica == [] and h.dialed
        # The replica answers a node-relayed session through its node's bridge, which
        # returns it over the connection the root dialed, and a direct one over that
        # connection itself.
        assert bool(h.attachment.frames) is (transport is Transport.NODE_RELAY)
        assert [(o.transport, o.outcome) for o in h.control.observations] == [
            (transport.value, RouteObservationOutcome.VERIFIED.value)
        ]
        assert await h.connections_drain() == 0

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
    )


def _foreign_leaf(ca: _TestCa) -> MutualTlsMaterial:
    """The deployment's CA bundle beside a leaf another CA issued."""
    leaf = new_ca("another-ca").material("root-node")
    return MutualTlsMaterial(
        ca_pem=ca.material("x").ca_pem, cert_pem=leaf.cert_pem, key_pem=leaf.key_pem
    )


@pytest.mark.parametrize(
    ("case", "observed"),
    [
        ("refused port", [RouteObservationOutcome.CONNECT_FAILURE]),
        ("target certificate for another host", [RouteObservationOutcome.TLS_FAILURE]),
        ("root leaf another CA issued", [RouteObservationOutcome.TLS_FAILURE]),
        ("root naming no identity", [RouteObservationOutcome.TLS_FAILURE]),
        # Load on a working path is not path evidence.
        ("listener at its cap", []),
    ],
)
def test_a_dial_that_fails_before_delivery_completes_on_the_relay(
    ca: _TestCa, case: str, observed: list[RouteObservationOutcome]
) -> None:
    root = ca.material("root-node")
    target = ca.material("worker-node", "127.0.0.1")
    if case == "target certificate for another host":
        target = ca.material("worker-node", "10.9.9.9")
    elif case == "root leaf another CA issued":
        root = _foreign_leaf(ca)
    elif case == "root naming no identity":
        root = ca.anonymous_material()

    async def body(h: _Harness) -> None:
        endpoint = f"127.0.0.1:{_closed_port()}" if case == "refused port" else None
        h.open(Transport.WORKER_DIRECT, endpoint=endpoint)
        await asyncio.wait_for(h.control.done.wait(), _DEADLINE)

        assert [o.status for o in h.control.outcomes] == [ResidentStreamStatus.SUCCESS]
        assert b"".join(h.control.chunks) == b"".join(_PARTS)
        assert [a.outcome for a in h.control.acks] == [ResidentBootstrapOutcome.ACKED]
        assert h.engine.calls == 1
        # Every frame of the attempt reached the replica over the rendezvous, under
        # the session, invocation and idempotency key control admitted.
        assert h.dialed == []
        assert {
            (f.session_id, f.correlation_id, f.operation_id)
            for f in h.relayed_to_replica
        } == {("rly-1", "inv-1", "idm-1")}
        assert [o.outcome for o in h.control.observations] == [
            o.value for o in observed
        ]
        assert await h.connections_drain() == 0

    _run(
        body,
        root=root,
        target=target,
        max_connections=0 if case == "listener at its cap" else 64,
    )


def test_a_loss_after_delivery_is_uncertain_and_never_replays_on_the_relay(
    ca: _TestCa,
) -> None:
    pause = asyncio.Event()

    async def body(h: _Harness) -> None:
        h.open(Transport.WORKER_DIRECT)
        await asyncio.wait_for(h.control.streaming.wait(), _DEADLINE)
        # The target's side of the connection drops mid-stream.
        await h.listener.stop()
        await asyncio.wait_for(h.control.done.wait(), _DEADLINE)

        assert [o.status for o in h.control.outcomes] == [
            ResidentStreamStatus.UNCERTAIN
        ]
        assert await h.relayed_frames() == 0
        assert h.control.observations[-1].outcome != (
            RouteObservationOutcome.VERIFIED.value
        )

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
        engine=_RawEngine(pause),
    )


@pytest.mark.parametrize("release", ["reap", "shutdown"])
def test_a_released_attempt_closes_its_socket(ca: _TestCa, release: str) -> None:
    pause = asyncio.Event()

    async def body(h: _Harness) -> None:
        h.open(Transport.WORKER_DIRECT)
        await asyncio.wait_for(h.control.streaming.wait(), _DEADLINE)
        assert h.open_connections == 1
        if release == "reap":
            h.executor.close("rly-1")
        else:
            await h.executor.stop()

        assert await h.connections_drain() == 0
        assert cast(PeerCarriage, h.executor._carriage)._sinks == {}
        # Releasing a healthy attempt is not path evidence.
        assert [o.outcome for o in h.control.observations] == [
            RouteObservationOutcome.VERIFIED.value
        ]
        # A release settles nothing: only the fenced terminal does.
        assert h.control.outcomes == []
        # The replica, as when the root dies mid-stream, finds its connection gone on
        # its next write: it drops the session and its engine request, and reports
        # nothing over its attachment.
        pause.set()
        await _until(lambda: not h.sidecar._sessions)
        assert h.attachment.frames == []

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
        engine=_RawEngine(pause),
    )


@pytest.mark.parametrize(
    ("reachable", "realized"),
    [(True, "worker_direct"), (False, "control_relay")],
)
def test_the_transport_span_records_the_transport_actually_used(
    ca: _TestCa, reachable: bool, realized: str
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    async def body(h: _Harness) -> None:
        endpoint = None if reachable else f"127.0.0.1:{_closed_port()}"
        h.open(
            Transport.WORKER_DIRECT,
            endpoint=endpoint,
            traceparent=format_traceparent(0xABC, 0xDEF),
        )
        await asyncio.wait_for(h.control.done.wait(), _DEADLINE)
        await _until(lambda: bool(exporter.get_finished_spans()))

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
        tracer_provider=provider,
    )
    (span,) = exporter.get_finished_spans()
    assert span.name == "flowmesh.transport.worker_direct"
    assert span.attributes is not None
    assert span.attributes[PHYSICAL_TRANSPORT] == realized
    assert span.context is not None and span.context.trace_id == 0xABC


def test_an_attempt_reaped_while_dialing_records_the_transport_it_was_dialing(
    ca: _TestCa,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    async def body(h: _Harness) -> None:
        reached = asyncio.Event()

        async def silent(reader, writer) -> None:
            # Takes the connection and never accepts it, holding the dial open.
            reached.set()
            with contextlib.suppress(OSError, asyncio.IncompleteReadError):
                await reader.read()
            writer.close()

        server = await asyncio.start_server(silent, "127.0.0.1", 0)
        async with server:
            h.open(
                Transport.WORKER_DIRECT,
                endpoint=f"127.0.0.1:{server.sockets[0].getsockname()[1]}",
                traceparent=format_traceparent(0xABC, 0xDEF),
            )
            await asyncio.wait_for(reached.wait(), _DEADLINE)
            h.executor.close("rly-1")
            await _until(lambda: bool(exporter.get_finished_spans()))

    _run(
        body,
        root=None,
        target=None,
        tracer_provider=provider,
    )
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    # Nothing rode the relay: the attempt ended while its dial was still open.
    assert span.attributes[PHYSICAL_TRANSPORT] == "worker_direct"


def test_a_stopped_root_opens_no_attempt(ca: _TestCa) -> None:
    async def body(h: _Harness) -> None:
        await h.executor.stop()
        # A re-drive that was waiting when the root shut down arrives afterwards.
        h.open(Transport.WORKER_DIRECT)

        # The open started no drive, so nothing can dial, relay or report later.
        assert h.executor._drive._by_session == {}
        assert h.dialed == [] and h.relayed_to_replica == []
        assert cast(PeerCarriage, h.executor._carriage)._sinks == {}
        assert h.control.outcomes == [] and h.control.observations == []

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
    )
