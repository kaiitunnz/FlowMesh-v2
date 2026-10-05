"""A workflow boundary rides a peer transport only when its origin worker can dial.

The node advertises the peer protocol whenever its peer plane is on, but dialing is the
worker's: a worker configured with the peer plane off (an external worker on a
peer-enabled node) holds only the relay carriage, so a peer plan it cannot carry would
hold the credit uncertain and re-drive into the same route.
"""

import asyncio

import pytest

from server.network.reachability import NetworkReachabilityView
from server.network.resolver import resolve_route
from server.network.service import TrustedPeerPolicy
from server.network.state import (
    NetworkEndpointAdvertisement,
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteOrigin,
)
from shared.schemas.network import PEER_PROTOCOL

from .test_service import _build, _env

_TRUSTED = TrustedPeerPolicy(
    enabled=True,
    trust_domain="td",
    classes=frozenset(ReachabilityClass),
    protocol=PEER_PROTOCOL,
)


class _TrustedPairNetwork:
    """A deployment trusting the origin-target pair, graded by the real resolver."""

    def __init__(self) -> None:
        self.observations: list = []

    async def resolve(
        self,
        origin_node_id: str,
        listener: ReplicaListenerAdvertisement,
        *,
        trust: TrustedPeerPolicy | None = None,
    ) -> tuple[RouteOrigin, ResolvedRoute]:
        origin = RouteOrigin(
            origin_id="rog-1",
            endpoint_id="ep-origin",
            node_id=origin_node_id,
            reachability_class=ReachabilityClass.ROUTABLE,
            trust_domain="td",
            protocols=(PEER_PROTOCOL,),
            relay_attachment_id="att-origin",
        )
        target = listener.model_copy(
            update={
                "routes": ("10.0.0.2:9500",),
                "protocols": (PEER_PROTOCOL,),
                "directly_routable": True,
            }
        )
        endpoint = NetworkEndpointAdvertisement(
            endpoint_id="ep-target",
            node_id=listener.node_id,
            url="10.0.0.2:9101",
            generation=1,
            trust_domain="td",
            reachability_class=ReachabilityClass.ROUTABLE,
            protocols=(PEER_PROTOCOL,),
            relay_attachment_id="att-target",
        )
        route = resolve_route(
            origin,
            target,
            endpoint,
            NetworkReachabilityView(),
            trust=_TRUSTED if trust is None else trust,
            now=0.0,
            route_epoch=1,
        )
        return origin, route

    def record_observations(self, origin, listener, observations) -> None:
        self.observations.extend(observations)

    async def endpoint_for(self, node_id: str):
        return None


@pytest.mark.parametrize(
    ("origin_listener_port", "transport"),
    [(0, "control_relay"), (41000, "worker_direct")],
)
def test_the_origin_worker_s_own_dial_capability_selects_the_transport(
    origin_listener_port: int, transport: str
) -> None:
    svc, _stores, _settled, delivery = _build()
    deps = svc._delivery
    assert deps is not None
    deps.network = _TrustedPairNetwork()
    deps.resident_listener_port_of = lambda worker_id: (
        origin_listener_port if worker_id == "wkr-origin" else 0
    )

    asyncio.run(svc._originate(_env()))

    plan = delivery.frame("resident_handoff")["carriage_plan"]
    assert plan["selected_transport"] == transport
