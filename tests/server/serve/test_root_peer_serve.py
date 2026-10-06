"""The root's gated serve origin dials a trusted target itself.

The root drives the serve origin over a socket it opens to the replica worker's
claim-gated listener, with its own node identity, so the request and response never
enter the rendezvous. A dial that fails before any frame is delivered rides the relay
base under the same session; a loss after delivery is uncertain and never replays on the
relay. Each attempt's socket and reader end with the attempt.
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

from server.network.peer_tls import PeerDialer
from server.network.reverse_relay import BinaryRedis
from server.serve.relay import ServeRelayExecutor
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
    ) -> None:
        self.redis = fakeredis.aioredis.FakeRedis()
        self.control = _Control()
        self.engine = engine
        self.executor = ServeRelayExecutor(
            relay_redis=cast(BinaryRedis, self.redis),
            edge_id="serve-edge",
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
        self.attachment = _Attachment()
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
        self.listener = ResidentPeerListener(
            sock=sock, material=target, deliver=self.sidecar.on_frame
        )

    @property
    def open_connections(self) -> int:
        return self.listener._listener._open

    async def start(self) -> None:
        await self.listener.start()

    async def aclose(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.listener.stop(), timeout=5.0)
        await self.sidecar.aclose()
        await self.redis.aclose()

    def open(
        self,
        transport: Transport,
        endpoint: str | None = None,
        traceparent: str | None = None,
    ) -> None:
        envelope = _envelope()
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
                selected_endpoint=(
                    endpoint if endpoint is not None else f"127.0.0.1:{self.port}"
                ),
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
        for _ in range(100):
            if self.open_connections == 0:
                break
            await asyncio.sleep(0.02)
        return self.open_connections


class _Attachment:
    """The replica worker's attachment: a dialed session never answers over it."""

    def __init__(self) -> None:
        self.frames: list[Any] = []

    async def send(self, frame: Any) -> None:
        self.frames.append(frame)


def _run(
    body: Callable[[_Harness], Awaitable[None]],
    *,
    root: MutualTlsMaterial | None,
    target: MutualTlsMaterial | None,
    engine: _RawEngine | None = None,
    tracer_provider: TracerProvider | None = None,
) -> None:
    async def run() -> None:
        h = _Harness(
            root=root,
            target=target,
            engine=engine or _RawEngine(),
            tracer_provider=tracer_provider,
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
        assert [(o.transport, o.outcome) for o in h.control.observations] == [
            (transport.value, RouteObservationOutcome.VERIFIED.value)
        ]
        assert await h.connections_drain() == 0

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
    )


def test_a_refused_dial_rides_the_relay_base_under_the_same_session(
    ca: _TestCa,
) -> None:
    async def body(h: _Harness) -> None:
        h.open(Transport.WORKER_DIRECT, endpoint=f"127.0.0.1:{_closed_port()}")
        for _ in range(100):
            if await h.relayed_frames():
                break
            await asyncio.sleep(0.02)

        entries = []
        for key in await h.redis.keys("*"):
            if await h.redis.type(key) == b"stream":
                entries.extend(await h.redis.xrange(key))
        frames = [RelayFrame.from_fields(entry[1]) for entry in entries]
        assert frames, "the bootstrap rides the relay base"
        assert {(f.session_id, f.correlation_id, f.operation_id) for f in frames} == {
            ("rly-1", "inv-1", "idm-1")
        }
        assert [o.outcome for o in h.control.observations] == [
            RouteObservationOutcome.CONNECT_FAILURE.value
        ]
        # The attempt is still live on the relay base: nothing is uncertain.
        assert h.control.outcomes == []
        assert h.engine.calls == 0

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
    )


def test_a_target_certificate_for_another_host_reaches_no_engine(ca: _TestCa) -> None:
    async def body(h: _Harness) -> None:
        h.open(Transport.WORKER_DIRECT)
        for _ in range(100):
            if h.control.observations:
                break
            await asyncio.sleep(0.02)

        assert [o.outcome for o in h.control.observations] == [
            RouteObservationOutcome.TLS_FAILURE.value
        ]
        assert await h.relayed_frames() > 0
        assert h.engine.calls == 0

    _run(
        body,
        root=ca.material("root-node"),
        target=ca.material("worker-node", "10.9.9.9"),
    )


def test_a_root_the_deployment_ca_did_not_issue_reaches_no_engine(ca: _TestCa) -> None:
    async def body(h: _Harness) -> None:
        h.open(Transport.WORKER_DIRECT)
        await asyncio.wait_for(h.control.done.wait(), _DEADLINE)

        assert all(
            o.status is not ResidentStreamStatus.SUCCESS for o in h.control.outcomes
        )
        assert h.control.acks == [] or all(
            a.outcome is not ResidentBootstrapOutcome.ACKED for a in h.control.acks
        )
        assert h.engine.calls == 0

    _run(
        body,
        root=new_ca().material("root-node"),
        target=ca.material("worker-node", "127.0.0.1"),
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
        assert h.executor._carriage.transport_of("rly-1") == "control_relay"
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
        for _ in range(100):
            if not h.sidecar._sessions:
                break
            await asyncio.sleep(0.02)
        assert h.sidecar._sessions == {}
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
        if reachable:
            await asyncio.wait_for(h.control.done.wait(), _DEADLINE)
        else:
            for _ in range(100):
                if await h.relayed_frames():
                    break
                await asyncio.sleep(0.02)
            h.executor.close("rly-1")
        for _ in range(100):
            if exporter.get_finished_spans():
                break
            await asyncio.sleep(0.02)

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
