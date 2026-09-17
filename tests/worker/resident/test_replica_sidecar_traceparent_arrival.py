"""The bootstrap ``RelayFrame``'s ``tp`` field reaches the engine-request span parent.

A replica sidecar binding serves many invocations from many workflows, so
``flowmesh.engine_request`` cannot take an ambient parent -- it opens from whatever
``tp`` the bootstrap frame that started the session carried. This drives a real
``RelayFrame`` through ``ResidentReplicaSidecar.on_frame`` and asserts the value lands
on the session's recorded traceparent, which is what the span helper reads.
"""

import asyncio

from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.resident.contracts import ReplicaEndpoint
from worker.resident.engine import EngineResponse
from worker.resident.replica_sidecar import ResidentReplicaSidecar

_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


class _NullSink:
    async def send(self, frame: RelayFrame) -> None:
        return None


def _bootstrap_frame(tp: str | None) -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="ssn-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        tp=tp,
    )


async def _engine_open_unused(
    endpoint: ReplicaEndpoint,
    request: str | None,
    adapter_name: str | None,
    adapter_source: str | None,
) -> EngineResponse:
    raise AssertionError("engine_open must not be reached by this test")


def test_bootstrap_frame_tp_reaches_the_recorded_session_traceparent() -> None:
    sidecar = ResidentReplicaSidecar(sink=_NullSink(), engine_open=_engine_open_unused)

    async def scenario() -> None:
        await sidecar.on_frame(_bootstrap_frame(_TRACEPARENT))
        assert sidecar._traceparents["ssn-1"] == _TRACEPARENT
        await sidecar.aclose()

    asyncio.run(scenario())


def test_bootstrap_frame_with_no_tp_records_none() -> None:
    """Telemetry off upstream carries no ``tp`` at all (zero bytes on the wire)."""
    sidecar = ResidentReplicaSidecar(sink=_NullSink(), engine_open=_engine_open_unused)

    async def scenario() -> None:
        await sidecar.on_frame(_bootstrap_frame(None))
        assert sidecar._traceparents["ssn-1"] is None
        await sidecar.aclose()

    asyncio.run(scenario())
