"""The resident lane host runs both lanes on its loop and bridges the sync thread.

Two hosts, each on its own asyncio loop thread, are wired frame-sink to frame-sink to
stand in for the reverse-relay the supervisors bridge. With the test standing in for
control, a resident invocation completes across the two hosts and materializes by
reference — exercising the loop marshal, the frame wire round-trip, and the
direction-based routing to the origin vs replica lane.
"""

import threading
from collections.abc import AsyncIterator
from typing import Any

from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.outcome import FabricContentStore, OutcomeManifest
from shared.outcome.manifest import content_digest
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
from worker.resident.engine import EngineResponse
from worker.resident.lane_host import ResidentLaneHost

_COMPLETION = "a resident completion streamed across two lane hosts in pieces"


class _MemStore(FabricContentStore):
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


async def _fake_engine(
    endpoint: ReplicaEndpoint, request: str | None
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        for start in range(0, len(_COMPLETION), 7):
            yield _COMPLETION[start : start + 7]

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


def _handoff() -> dict[str, Any]:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    ).model_dump(mode="json")


def _auth() -> dict[str, Any]:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    ).model_dump(mode="json")


def test_wire_round_trip() -> None:
    frame = RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="s1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=3,
        ack=0,
        payload=b'{"kind": "chunk", "data": "\xe2\x9c\x93"}',
    )
    restored = RelayFrame.from_wire(frame.to_wire())
    assert restored == frame


def test_two_hosts_complete_a_resident_invocation() -> None:
    store = _MemStore()
    outcomes: list[ResidentOpOutcome] = []
    done = threading.Event()

    hosts: dict[str, ResidentLaneHost] = {}

    def origin_pushes(frame: dict[str, Any]) -> None:
        hosts["replica"].route("resident_frame", frame)

    def replica_pushes(frame: dict[str, Any]) -> None:
        hosts["origin"].route("resident_frame", frame)

    def on_ack(ack: ResidentBootstrapAck) -> None:
        if ack.outcome is ResidentBootstrapOutcome.ACKED:
            hosts["origin"].route(
                "resident_authorization",
                {"call_correlation": ack.call_correlation, "auth": _auth()},
            )

    def on_outcome(outcome: ResidentOpOutcome) -> None:
        outcomes.append(outcome)
        done.set()

    def noop_ack(_ack: ResidentBootstrapAck) -> None:
        return None

    def noop_outcome(_outcome: ResidentOpOutcome) -> None:
        return None

    def peek(_task: str, _call: str) -> str | None:
        return '{"prompt": "hi"}'

    deleted: list[tuple[str, str]] = []

    origin = ResidentLaneHost(
        push_frame=origin_pushes,
        report_ack=on_ack,
        report_outcome=on_outcome,
        content_store=store,
        peek_request=peek,
        delete_request=lambda t, c: deleted.append((t, c)),
    )
    replica = ResidentLaneHost(
        push_frame=replica_pushes,
        report_ack=noop_ack,
        report_outcome=noop_outcome,
        content_store=None,
        peek_request=lambda _t, _c: None,
        delete_request=lambda _t, _c: None,
        engine_open=_fake_engine,
    )
    hosts["origin"], hosts["replica"] = origin, replica
    origin.start()
    replica.start()
    try:
        replica.route(
            "resident_sidecar_bind",
            {
                "replica_id": "rpl-1",
                "incarnation": 1,
                "listener_generation": 1,
                "engine": {
                    "base_url": "http://engine/v1",
                    "model": "m",
                    "api_key": None,
                },
            },
        )
        origin.route(
            "resident_handoff",
            {
                "task_id": "tsk-1",
                "call_correlation": "call-1",
                "session_id": "rly-1",
                "handoff": _handoff(),
            },
        )
        assert done.wait(timeout=10.0)
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.status is ResidentStreamStatus.SUCCESS
        assert outcome.manifest is not None
        assert store.hydrate(outcome.manifest).decode() == _COMPLETION

        # The fenced-terminal reap drops the worker-private raw request so it does not
        # outlive the invocation (the leak the delete hook closes).
        assert deleted == []
        origin.route(
            "resident_reap", {"task_id": "tsk-1", "call_correlation": "call-1"}
        )
        for _ in range(100):
            if deleted:
                break
            threading.Event().wait(0.02)
        assert deleted == [("tsk-1", "call-1")]
    finally:
        origin.stop()
        replica.stop()
