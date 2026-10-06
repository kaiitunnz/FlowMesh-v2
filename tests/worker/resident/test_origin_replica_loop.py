"""The origin driver and replica sidecar complete a resident invocation together.

The two worker lanes are wired sink-to-sink, with the test standing in for control: it
mints the handoff, issues a route authorization when the origin reports the engine ack,
and reads the fenced manifest the origin materializes. A post-manifest re-drive
re-reports the recorded reference without re-running the engine.
"""

import asyncio
import contextlib
import contextvars
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import Any

import pytest

from shared.network.mtls import MutualTlsMaterial, client_context
from shared.network.mtls_listener import MAX_CONNECTIONS
from shared.network.relay_frame import RelayFrame
from shared.network.session import FramedRelaySession  # noqa: F401 - re-export check
from shared.resident.carriage import ControlRelayCarriage, ResidentCarriagePlan
from shared.resident.contracts import (
    AdmissionHandoff,
    ReplicaEndpoint,
    RouteAuthorization,
)
from shared.resident.peer_carriage import PeerCarriage
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamStatus,
)
from shared.schemas.network import RouteObservationOutcome
from tests.shared.outcome_helpers import InMemoryContentStore
from tests.support.certs import new_ca
from worker.resident.engine import EngineResponse
from worker.resident.origin_driver import ResidentOriginDriver, ResidentOriginRequest
from worker.resident.peer_listener import ResidentPeerListener
from worker.resident.replica_sidecar import ResidentReplicaSidecar

_COMPLETION = "the resident model reply, streamed in pieces"


class _ToPeer:
    """One direction's frame sink, delivering straight into the peer's handler.

    Delivery runs in a fresh context: the peer is another process, so it inherits
    nothing ambient from the sender and reads only what the frame carries.
    """

    def __init__(self, observe: Callable[[RelayFrame], None] | None = None) -> None:
        self.on_peer: Callable[[RelayFrame], Coroutine[Any, Any, None]] | None = None
        self._observe = observe

    async def send(self, frame: RelayFrame) -> None:
        assert self.on_peer is not None
        if self._observe is not None:
            self._observe(frame)
        await asyncio.get_running_loop().create_task(
            self.on_peer(frame), context=contextvars.Context()
        )


def _engine(calls: list[int]) -> Callable[..., Awaitable[EngineResponse]]:
    async def engine(
        endpoint: ReplicaEndpoint,
        request: str | None,
        adapter_name: str | None = None,
        adapter_source: str | None = None,
    ) -> EngineResponse:
        calls.append(1)
        size = 8

        async def chunks() -> AsyncIterator[str]:
            for start in range(0, len(_COMPLETION), size):
                yield _COMPLETION[start : start + size]

        async def aclose() -> None:
            return None

        return EngineResponse(chunks=chunks(), aclose=aclose)

    return engine


def _handoff(session_no: int = 1) -> AdmissionHandoff:
    return AdmissionHandoff(
        token=f"hnd-{session_no}",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    )


def _auth() -> RouteAuthorization:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    )


class _Harness:
    def __init__(self, observe: Callable[[RelayFrame], None] | None = None) -> None:
        self.store = InMemoryContentStore()
        self.engine_calls: list[int] = []
        self.acks: list[ResidentBootstrapAck] = []
        self.outcomes: list[ResidentOpOutcome] = []
        self.done = asyncio.Event()

        origin_sink, replica_sink = _ToPeer(observe), _ToPeer(observe)
        self.sidecar = ResidentReplicaSidecar(
            sink=replica_sink, engine_open=_engine(self.engine_calls)
        )
        self.sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        )
        self.origin = ResidentOriginDriver(
            carriage=ControlRelayCarriage(origin_sink),
            content_store_for=lambda task_id: self.store,
            report_ack=self._on_ack,
            report_outcome=self._on_outcome,
        )
        origin_sink.on_peer = self.sidecar.on_frame
        replica_sink.on_peer = self.origin.on_frame

    def _on_ack(self, ack: ResidentBootstrapAck) -> None:
        self.acks.append(ack)
        if ack.outcome is ResidentBootstrapOutcome.ACKED:
            # Control accepts the claim and issues the post-acceptance authorization.
            self.origin.authorize(_auth())

    def _on_outcome(self, outcome: ResidentOpOutcome) -> None:
        self.outcomes.append(outcome)
        self.done.set()

    def begin(
        self,
        session_no: int = 1,
        request_payload: str | None = '{"prompt": "hi"}',
        traceparent: str | None = None,
    ) -> None:
        self.origin.begin(
            ResidentOriginRequest(
                task_id="tsk-1",
                call_correlation="call-1",
                session_id=f"rly-{session_no}",
                handoff=_handoff(session_no),
                request_payload=request_payload,
                carriage_plan=ResidentCarriagePlan(session_id=f"rly-{session_no}"),
                traceparent=traceparent,
            )
        )


def test_origin_and_replica_complete_by_reference() -> None:
    async def run() -> None:
        h = _Harness()
        h.begin()
        await asyncio.wait_for(h.done.wait(), timeout=10.0)
        assert len(h.outcomes) == 1
        outcome = h.outcomes[0]
        assert outcome.status is ResidentStreamStatus.SUCCESS
        assert outcome.manifest is not None
        # The completion materialized by reference; it never crossed as an inline value.
        assert h.store.hydrate(outcome.manifest.content).decode() == _COMPLETION
        assert h.engine_calls == [1]
        await h.sidecar.aclose()

    asyncio.run(run())


def test_uncaptured_request_holds_uncertain_without_running_the_engine() -> None:
    async def run() -> None:
        h = _Harness()
        # A re-drive landed on a worker that never captured the request (peek miss):
        # the driver must hold the credit UNCERTAIN, never bootstrap an empty-prompt
        # request that could settle a bogus success.
        h.begin(request_payload=None)
        await asyncio.wait_for(h.done.wait(), timeout=10.0)
        assert len(h.outcomes) == 1
        assert h.outcomes[0].status is ResidentStreamStatus.UNCERTAIN
        assert h.engine_calls == []  # the engine never ran
        assert h.acks == []  # no bootstrap was even attempted
        await h.sidecar.aclose()

    asyncio.run(run())


def test_post_manifest_redrive_reuses_the_reference() -> None:
    async def run() -> None:
        h = _Harness()
        h.begin(session_no=1)
        await asyncio.wait_for(h.done.wait(), timeout=10.0)
        first = h.outcomes[0]
        assert first.status is ResidentStreamStatus.SUCCESS
        assert first.manifest is not None

        # A re-drive after the manifest committed: control re-relays a fresh handoff,
        # but the origin finds the recorded outcome and re-reports it without a second
        # engine run.
        h.done.clear()
        h.begin(session_no=2)
        await asyncio.wait_for(h.done.wait(), timeout=10.0)
        second = h.outcomes[-1]
        assert second.status is ResidentStreamStatus.SUCCESS
        assert second.manifest is not None
        assert second.manifest.content == first.manifest.content
        assert h.engine_calls == [1]  # the engine ran once, not twice
        await h.sidecar.aclose()

    asyncio.run(run())


class _DialedHarness:
    """The two lanes over a real dialed socket, as a trusted peer carries them, with
    the relay between them as the base a dial that fails before delivery falls to."""

    def __init__(
        self,
        sock,
        port: int,
        *,
        target: MutualTlsMaterial | None = None,
        origin: MutualTlsMaterial | None = None,
        max_connections: int = MAX_CONNECTIONS,
    ) -> None:
        self.store = InMemoryContentStore()
        self.engine_calls: list[int] = []
        self.outcomes: list[ResidentOpOutcome] = []
        self.observed: list[RouteObservationOutcome] = []
        self.dialed: list[RelayFrame] = []
        self.relayed: list[RelayFrame] = []
        self.done = asyncio.Event()

        base, replica_attachment = _ToPeer(self.relayed.append), _ToPeer()
        self.sidecar = ResidentReplicaSidecar(
            sink=replica_attachment, engine_open=_engine(self.engine_calls)
        )
        self.sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        )
        self.listener = ResidentPeerListener(
            sock=sock,
            material=target,
            deliver=self._dialed_frame,
            max_connections=max_connections,
        )
        self.carriage = PeerCarriage(
            base=base,
            deliver=lambda frame: self.origin.on_frame(frame),
            observe=lambda session, transport, outcome: self.observed.append(outcome),
            ssl_context=client_context(origin) if origin is not None else None,
            connect_budget_sec=2.0,
        )
        self.origin = ResidentOriginDriver(
            carriage=self.carriage,
            content_store_for=lambda task_id: self.store,
            report_ack=self._on_ack,
            report_outcome=self._on_outcome,
        )
        base.on_peer = self.sidecar.on_frame
        replica_attachment.on_peer = self.origin.on_frame
        self._endpoint = f"127.0.0.1:{port}"

    async def _dialed_frame(self, frame: RelayFrame, sink: Any) -> None:
        self.dialed.append(frame)
        await self.sidecar.on_frame(frame, sink)

    def _on_ack(self, ack: ResidentBootstrapAck) -> None:
        if ack.outcome is ResidentBootstrapOutcome.ACKED:
            self.origin.authorize(_auth())

    def _on_outcome(self, outcome: ResidentOpOutcome) -> None:
        self.outcomes.append(outcome)
        self.done.set()

    async def invoke(self, session_no: int, traceparent: str | None = None) -> None:
        self.done.clear()
        self.origin.begin(
            ResidentOriginRequest(
                task_id="tsk-1",
                call_correlation=f"call-{session_no}",
                session_id=f"rly-{session_no}",
                handoff=_handoff(session_no),
                request_payload='{"prompt": "hi"}',
                carriage_plan=ResidentCarriagePlan(
                    session_id=f"rly-{session_no}",
                    selected_transport="worker_direct",
                    selected_endpoint=self._endpoint,
                ),
                traceparent=traceparent,
            )
        )
        await asyncio.wait_for(self.done.wait(), timeout=10.0)


def test_repeated_peers_release_both_ends_of_the_dialed_socket() -> None:
    # The target's connection cannot end until the origin closes, so an attempt that
    # leaks its sink also pins a connection against the listener's cap: after enough
    # invocations the listener refuses every further peer session and the feature
    # silently falls back to the relay.
    async def run() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        h = _DialedHarness(sock, port)
        await h.listener.start()
        try:
            for session_no in range(1, 4):
                await h.invoke(session_no)
                assert h.outcomes[-1].status is ResidentStreamStatus.SUCCESS
                assert h.carriage._sinks == {}
                for _ in range(50):
                    if h.listener._listener._open == 0:
                        break
                    await asyncio.sleep(0.02)
                assert h.listener._listener._open == 0
        finally:
            # A connection the origin never released would keep the listener's close
            # waiting, so bound it rather than hanging the failure.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(h.listener.stop(), timeout=5.0)
            await h.sidecar.aclose()

    asyncio.run(run())


def _foreign_leaf(ca) -> MutualTlsMaterial:
    """The deployment's CA bundle beside a leaf another CA issued."""
    leaf = new_ca("another-ca").material("worker-origin")
    return MutualTlsMaterial(
        ca_pem=ca.material("x").ca_pem, cert_pem=leaf.cert_pem, key_pem=leaf.key_pem
    )


@pytest.mark.parametrize(
    ("case", "observed"),
    [
        ("origin leaf another CA issued", [RouteObservationOutcome.TLS_FAILURE]),
        ("origin naming no identity", [RouteObservationOutcome.TLS_FAILURE]),
        # Load on a working path is not path evidence.
        ("listener at its cap", []),
    ],
)
def test_a_target_refusing_the_dial_completes_the_invocation_on_the_relay(
    case: str, observed: list[RouteObservationOutcome]
) -> None:
    ca = new_ca()
    origin = ca.material("worker-origin")
    if case == "origin leaf another CA issued":
        origin = _foreign_leaf(ca)
    elif case == "origin naming no identity":
        origin = ca.anonymous_material()

    async def run() -> _DialedHarness:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        h = _DialedHarness(
            sock,
            sock.getsockname()[1],
            target=ca.material("worker-target", "127.0.0.1"),
            origin=origin,
            max_connections=0 if case == "listener at its cap" else MAX_CONNECTIONS,
        )
        await h.listener.start()
        try:
            await h.invoke(1)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(h.listener.stop(), timeout=5.0)
            await h.sidecar.aclose()
        return h

    h = asyncio.run(run())

    assert [o.status for o in h.outcomes] == [ResidentStreamStatus.SUCCESS]
    assert h.engine_calls == [1]
    assert h.dialed == []
    assert {(f.session_id, f.correlation_id, f.operation_id) for f in h.relayed} == {
        ("rly-1", "inv-1", "idm-1")
    }
    assert h.observed == observed
