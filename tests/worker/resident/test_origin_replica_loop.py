"""The origin driver and replica sidecar complete a resident invocation together.

The two worker lanes are wired sink-to-sink, with the test standing in for control: it
mints the handoff, issues a route authorization when the origin reports the engine ack,
and reads the fenced manifest the origin materializes. A post-manifest re-drive
re-reports the recorded reference without re-running the engine.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from shared.network.relay_frame import RelayFrame
from shared.outcome import FabricContentStore, OutcomeManifest
from shared.outcome.manifest import content_digest
from shared.resident.carriage import ControlRelayCarriage, ResidentCarriagePlan
from shared.resident.contracts import (
    AdmissionHandoff,
    ReplicaEndpoint,
    RouteAuthorization,
)
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamStatus,
)
from shared.resident.session import ResidentRelaySession  # noqa: F401 - re-export check
from worker.resident.engine import EngineResponse
from worker.resident.origin_driver import ResidentOriginDriver, ResidentOriginRequest
from worker.resident.replica_sidecar import ResidentReplicaSidecar

_COMPLETION = "the resident model reply, streamed in pieces"


class _MemStore(FabricContentStore):
    """An in-memory content-addressed store: find-or-commit by idempotency key."""

    def __init__(self) -> None:
        self._by_idem: dict[str, OutcomeManifest] = {}
        self._by_digest: dict[str, bytes] = {}

    def find(self, idempotency_key: str) -> OutcomeManifest | None:
        return self._by_idem.get(idempotency_key)

    def materialize(
        self, idempotency_key: str, data: bytes, *, media_type: str
    ) -> OutcomeManifest:
        if idempotency_key in self._by_idem:
            return self._by_idem[idempotency_key]
        digest = content_digest(data)
        self._by_digest[digest] = data
        manifest = OutcomeManifest(
            content_digest=digest,
            size_bytes=len(data),
            media_type=media_type,
            idempotency_key=idempotency_key,
        )
        self._by_idem[idempotency_key] = manifest
        return manifest

    def read(self, digest: str) -> bytes:
        return self._by_digest[digest]


class _ToPeer:
    def __init__(self) -> None:
        self.on_peer: Callable[[RelayFrame], Awaitable[None]] | None = None

    async def send(self, frame: RelayFrame) -> None:
        assert self.on_peer is not None
        await self.on_peer(frame)


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
    def __init__(self) -> None:
        self.store = _MemStore()
        self.engine_calls: list[int] = []
        self.acks: list[ResidentBootstrapAck] = []
        self.outcomes: list[ResidentOpOutcome] = []
        self.done = asyncio.Event()

        origin_sink, replica_sink = _ToPeer(), _ToPeer()
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
            content_store=self.store,
            report_ack=self._on_ack,
            report_outcome=self._on_outcome,
        )
        origin_sink.on_peer = self.sidecar.on_frame
        replica_sink.on_peer = self.origin.on_frame

    def _on_ack(self, ack: ResidentBootstrapAck) -> None:
        self.acks.append(ack)
        if ack.outcome is ResidentBootstrapOutcome.ACKED:
            # Control accepts the claim and issues the post-acceptance authorization.
            self.origin.authorize(ack.call_correlation, _auth())

    def _on_outcome(self, outcome: ResidentOpOutcome) -> None:
        self.outcomes.append(outcome)
        self.done.set()

    def begin(
        self, session_no: int = 1, request_payload: str | None = '{"prompt": "hi"}'
    ) -> None:
        self.origin.begin(
            ResidentOriginRequest(
                task_id="tsk-1",
                call_correlation="call-1",
                session_id=f"rly-{session_no}",
                handoff=_handoff(session_no),
                request_payload=request_payload,
                carriage_plan=ResidentCarriagePlan(session_id=f"rly-{session_no}"),
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
        assert h.store.hydrate(outcome.manifest).decode() == _COMPLETION
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
        assert second.manifest.content_digest == first.manifest.content_digest
        assert h.engine_calls == [1]  # the engine ran once, not twice
        await h.sidecar.aclose()

    asyncio.run(run())
