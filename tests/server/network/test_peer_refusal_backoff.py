"""An origin a target keeps refusing demotes the path with a climbing backoff.

The dialer's side of a TLS 1.3 handshake completes before the target has checked its
certificate, so a path counts as verified only once the target accepts the connection.
"""

import asyncio
import socket

from server.network.reachability import NetworkReachabilityView
from server.network.state import PolicyClass, RouteObservation
from shared.network.mtls import client_context
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind
from shared.resident.carriage import ResidentCarriagePlan
from shared.resident.peer_carriage import PeerCarriage
from shared.schemas.network import RouteObservationOutcome, Transport
from tests.support.certs import new_ca
from worker.resident.peer_listener import ResidentPeerListener


class _Base:
    async def send(self, frame: RelayFrame) -> None:
        return None


async def _unused_delivery(frame: RelayFrame, sink: object) -> None:
    raise AssertionError("a refused dialer reaches no sidecar")


def test_a_persistently_refused_origin_backs_off_further_each_attempt() -> None:
    ca = new_ca()
    observed: list[RouteObservationOutcome] = []

    async def run() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        listener = ResidentPeerListener(
            sock=sock,
            material=ca.material("worker-target", "127.0.0.1"),
            deliver=_unused_delivery,
        )
        await listener.start()
        carriage = PeerCarriage(
            base=_Base(),
            deliver=lambda frame: asyncio.sleep(0),
            observe=lambda session, transport, outcome: observed.append(outcome),
            ssl_context=client_context(ca.anonymous_material()),
            connect_budget_sec=2.0,
        )
        try:
            for attempt in range(1, 4):
                session_id = f"rly-{attempt}"
                sink = carriage.select(
                    ResidentCarriagePlan(
                        session_id=session_id,
                        selected_transport=Transport.WORKER_DIRECT.value,
                        selected_endpoint=f"127.0.0.1:{listener.port}",
                    )
                )
                await sink.send(
                    RelayFrame(
                        kind=RelayFrameKind.DATA,
                        session_id=session_id,
                        correlation_id="inv-1",
                        operation_id="idm-1",
                        direction=RelayDirection.ORIGIN_TO_TARGET,
                        seq=1,
                    )
                )
                carriage.close(session_id)
        finally:
            await listener.stop()

    asyncio.run(run())

    view = NetworkReachabilityView()
    backoffs = []
    for at, outcome in enumerate(observed):
        now = 1000.0 * (at + 1)
        view.mark_optimistic(
            "rog-1",
            PolicyClass.DEFAULT,
            "nde-1",
            1,
            1,
            Transport.WORKER_DIRECT,
            now=now,
        )
        view.observe(
            RouteObservation(
                origin_id="rog-1",
                policy_class=PolicyClass.DEFAULT,
                target_node_id="nde-1",
                incarnation=1,
                listener_generation=1,
                transport=Transport.WORKER_DIRECT,
                outcome=outcome,
            ),
            now=now,
        )
        (entry,) = view.entries()
        assert entry.backoff_until is not None
        backoffs.append(entry.backoff_until - now)

    assert observed == [RouteObservationOutcome.TLS_FAILURE] * 3
    assert backoffs == [1.0, 2.0, 4.0]
