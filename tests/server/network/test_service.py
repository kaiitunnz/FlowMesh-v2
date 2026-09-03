"""The NetworkPlane service: resolve, observation folding, and rotation fencing."""

import asyncio
import logging

from server.config import NetworkPlaneConfig
from server.network.service import NetworkPlane
from server.network.state import (
    NetworkEndpointAdvertisement,
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    RouteObservationOutcome,
    Transport,
)
from server.registries.node import Node


class _FakeNodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def set(self, node: Node) -> None:
        self._nodes[node.id] = node

    async def get_node_async(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    async def list_nodes_async(self) -> list[Node]:
        return list(self._nodes.values())


def _node(node_id: str, *, generation: int, cls=ReachabilityClass.ROUTABLE) -> Node:
    return Node(
        id=node_id,
        namespace="ns",
        cluster="cl",
        alias=node_id,
        network_endpoint=NetworkEndpointAdvertisement(
            endpoint_id=f"ep-{node_id}",
            url=f"127.0.0.1:900{node_id[-1]}",
            generation=generation,
            trust_domain="fm",
            reachability_class=cls,
            relay_attachment_id=f"att-{node_id}",
        ),
    )


def _outbound_only_node(node_id: str, *, generation: int) -> Node:
    # A network-plane node with no inbound URL: empty endpoint_id/url and no attachment
    # id, as the provider mints it. The server derives the identity from the node.
    return Node(
        id=node_id,
        namespace="ns",
        cluster="cl",
        alias=node_id,
        network_endpoint=NetworkEndpointAdvertisement(
            endpoint_id="",
            url="",
            generation=generation,
            trust_domain="fm",
            reachability_class=ReachabilityClass.ROUTABLE,
        ),
    )


def _listener(node_id="nde-2", generation=0) -> ReplicaListenerAdvertisement:
    return ReplicaListenerAdvertisement(
        replica_id="rpl-1",
        family="echo",
        incarnation=1,
        listener_generation=generation,
        node_id=node_id,
        routes=("127.0.0.1:9500",),
        directly_routable=True,
    )


def _plane(registry: _FakeNodeRegistry) -> NetworkPlane:
    return NetworkPlane(
        NetworkPlaneConfig(enabled=True),
        registry,  # type: ignore[arg-type]
        logging.getLogger("test-network"),
    )


def test_resolve_returns_ladder() -> None:
    registry = _FakeNodeRegistry()
    registry.set(_node("nde-1", generation=1))
    registry.set(_node("nde-2", generation=1))
    plane = _plane(registry)
    result = asyncio.run(plane.resolve("nde-1", _listener()))
    assert result is not None
    _origin, route = result
    transports = [c.transport.value for c in route.candidates]
    assert transports[0] == "worker_direct"
    assert "node_relay" in transports and "control_relay" in transports


def test_resolve_none_without_origin_endpoint() -> None:
    registry = _FakeNodeRegistry()
    registry.set(
        Node(id="nde-1", namespace="ns", cluster="cl", alias="nde-1")
    )  # no advertisement
    registry.set(_node("nde-2", generation=1))
    plane = _plane(registry)
    assert asyncio.run(plane.resolve("nde-1", _listener())) is None


def test_outbound_only_nodes_resolve_only_the_control_relay() -> None:
    # Neither the origin nor the target has an inbound URL, so the forward-dial
    # transports drop out, but the reverse-relay base still resolves: the attachment
    # identity comes from the node, not the (absent) inbound endpoint.
    registry = _FakeNodeRegistry()
    registry.set(_outbound_only_node("nde-1", generation=1))
    registry.set(_outbound_only_node("nde-2", generation=1))
    plane = _plane(registry)
    listener = ReplicaListenerAdvertisement(
        replica_id="rpl-1",
        family="echo",
        incarnation=1,
        listener_generation=0,
        node_id="nde-2",
        routes=("127.0.0.1:9500",),
        directly_routable=False,
    )
    result = asyncio.run(plane.resolve("nde-1", listener))
    assert result is not None
    _origin, route = result
    transports = [c.transport.value for c in route.candidates]
    assert transports == ["control_relay"]


def test_observation_demotes_and_next_resolve_drops_direct() -> None:
    registry = _FakeNodeRegistry()
    registry.set(_node("nde-1", generation=1))
    registry.set(_node("nde-2", generation=1))
    plane = _plane(registry)

    async def scenario() -> list[str]:
        first = await plane.resolve("nde-1", _listener())
        assert first is not None
        origin, _route = first
        plane.record_observations(
            origin,
            _listener(),
            [(Transport.WORKER_DIRECT, RouteObservationOutcome.CONNECT_FAILURE)],
        )
        second = await plane.resolve("nde-1", _listener())
        assert second is not None
        return [c.transport.value for c in second[1].candidates]

    transports = asyncio.run(scenario())
    assert "worker_direct" not in transports


def test_rotation_invalidates_reachability() -> None:
    registry = _FakeNodeRegistry()
    registry.set(_node("nde-1", generation=1))
    registry.set(_node("nde-2", generation=1))
    plane = _plane(registry)

    async def scenario() -> dict[str, str]:
        first = await plane.resolve("nde-1", _listener())
        assert first is not None
        origin, _route = first
        plane.record_observations(
            origin,
            _listener(),
            [(Transport.WORKER_DIRECT, RouteObservationOutcome.VERIFIED)],
        )
        # Target re-registers with a higher endpoint generation.
        registry.set(_node("nde-2", generation=2))
        second = await plane.resolve("nde-1", _listener())
        assert second is not None
        return plane.reachability_states(second[0], _listener())

    states = asyncio.run(scenario())
    # The prior VERIFIED entry was invalidated; the fresh attempt is only optimistic.
    assert states["worker_direct"] != "verified"


def test_endpoints_are_stamped_with_node_id() -> None:
    registry = _FakeNodeRegistry()
    registry.set(_node("nde-1", generation=1))
    plane = _plane(registry)
    endpoints = asyncio.run(plane.endpoints())
    assert endpoints[0].node_id == "nde-1"
