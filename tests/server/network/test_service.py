"""The NetworkPlane service: resolve, observation folding, and rotation fencing."""

import asyncio
import logging

import pytest

from server.config import NetworkPlaneConfig, TrustedPeerConfig
from server.network.service import NetworkPlane
from server.network.state import (
    NetworkEndpointAdvertisement,
    PolicyClass,
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    RouteObservationOutcome,
    Transport,
)
from server.registries.node import Node
from server.supervisor.supervisor import _endpoint_advertisement_provider
from shared.schemas.network import PEER_PROTOCOL


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
            peer_url=f"127.0.0.1:910{node_id[-1]}",
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


def _plane(registry: _FakeNodeRegistry, *, peer: bool = False) -> NetworkPlane:
    return NetworkPlane(
        NetworkPlaneConfig(enabled=True, peer=TrustedPeerConfig(enabled=peer)),
        registry,  # type: ignore[arg-type]
        logging.getLogger("test-network"),
    )


def test_resolve_offers_only_the_relay_without_a_trusted_peer_posture() -> None:
    # The peer posture is off by default, so a deployment that declared no trusted
    # class carries resident traffic over the relay even though the target advertises a
    # dialable address.
    registry = _FakeNodeRegistry()
    registry.set(_node("nde-1", generation=1))
    registry.set(_node("nde-2", generation=1))
    plane = _plane(registry)
    result = asyncio.run(plane.resolve("nde-1", _listener()))
    assert result is not None
    _origin, route = result
    assert [c.transport.value for c in route.candidates] == ["control_relay"]


def test_a_probe_resolves_the_full_ladder() -> None:
    # Where peer listeners are served, a probe is offered every reachable peer path,
    # trusted for resident traffic or not.
    registry = _FakeNodeRegistry()
    registry.set(_node("nde-1", generation=1))
    registry.set(_node("nde-2", generation=1))
    plane = _plane(registry, peer=True)
    assert plane.probe_trust.probe
    result = asyncio.run(plane.resolve("nde-1", _listener(), trust=plane.probe_trust))
    assert result is not None
    _origin, route = result
    transports = [c.transport.value for c in route.candidates]
    assert transports[0] == "worker_direct"
    assert "node_relay" in transports and "control_relay" in transports


def test_a_probe_has_no_forward_dial_candidate_without_peer_listeners() -> None:
    # Without the peer plane no listener answers a forward dial, so a probe offers only
    # the relay base, which it does not dial.
    registry = _FakeNodeRegistry()
    registry.set(_node("nde-1", generation=1))
    registry.set(_node("nde-2", generation=1))
    plane = _plane(registry)
    result = asyncio.run(plane.resolve("nde-1", _listener(), trust=plane.probe_trust))
    assert result is not None
    _origin, route = result
    assert [c.transport.value for c in route.candidates] == ["control_relay"]


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


def test_a_peer_node_without_a_listener_dials_a_trusted_worker_directly() -> None:
    peer = TrustedPeerConfig(enabled=True, trust_domain="fm", classes=("routable",))
    origin_ad = _endpoint_advertisement_provider(
        NetworkPlaneConfig(enabled=True, trust_domain="fm", peer=peer)
    )()
    assert origin_ad is not None
    registry = _FakeNodeRegistry()
    registry.set(
        Node(
            id="nde-1",
            namespace="ns",
            cluster="cl",
            alias="nde-1",
            network_endpoint=origin_ad,
        )
    )
    registry.set(_node("nde-2", generation=1))
    plane = NetworkPlane(
        NetworkPlaneConfig(enabled=True, trust_domain="fm", peer=peer),
        registry,  # type: ignore[arg-type]
        logging.getLogger("test-network"),
    )
    listener = _listener().model_copy(update={"protocols": (PEER_PROTOCOL,)})

    result = asyncio.run(plane.resolve("nde-1", listener))

    assert result is not None
    _origin, route = result
    assert [c.transport.value for c in route.candidates] == [
        "worker_direct",
        "node_relay",
        "control_relay",
    ]


def _trusted_plane(registry: _FakeNodeRegistry) -> NetworkPlane:
    return NetworkPlane(
        NetworkPlaneConfig(
            enabled=True,
            peer=TrustedPeerConfig(
                enabled=True, trust_domain="fm", classes=("routable",)
            ),
        ),
        registry,  # type: ignore[arg-type]
        logging.getLogger("test-network"),
    )


def _peer_node(node_id: str) -> Node:
    node = _node(node_id, generation=1)
    assert node.network_endpoint is not None
    return node.model_copy(
        update={
            "network_endpoint": node.network_endpoint.model_copy(
                update={"protocols": (PEER_PROTOCOL,)}
            )
        }
    )


@pytest.mark.parametrize(
    ("demoted", "untouched"),
    [
        (PolicyClass.SERVE_INGRESS, PolicyClass.DEFAULT),
        (PolicyClass.DEFAULT, PolicyClass.SERVE_INGRESS),
    ],
)
def test_the_root_serve_ingress_and_the_nodes_workers_keep_separate_evidence(
    demoted: PolicyClass, untouched: PolicyClass
) -> None:
    # The root serve ingress and a worker on the root node dial from one node endpoint,
    # but a failure one of them observes never steers the other's route.
    registry = _FakeNodeRegistry()
    registry.set(_peer_node("nde-1"))
    registry.set(_peer_node("nde-2"))
    plane = _trusted_plane(registry)
    listener = _listener().model_copy(update={"protocols": (PEER_PROTOCOL,)})

    def head(policy_class: PolicyClass) -> tuple[str, str]:
        result = asyncio.run(
            plane.resolve("nde-1", listener, policy_class=policy_class)
        )
        assert result is not None
        origin, route = result
        return origin.origin_id, route.candidates[0].transport.value

    demoted_id, demoted_head = head(demoted)
    untouched_id, untouched_head = head(untouched)
    assert demoted_id != untouched_id
    assert demoted_head == untouched_head == "worker_direct"

    origin, _route = asyncio.run(
        plane.resolve("nde-1", listener, policy_class=demoted)
    ) or (None, None)
    assert origin is not None and origin.policy_class is demoted
    plane.record_observations(
        origin,
        listener,
        [(Transport.WORKER_DIRECT, RouteObservationOutcome.CONNECT_FAILURE)],
    )

    assert head(demoted)[1] != "worker_direct"
    assert head(untouched) == (untouched_id, "worker_direct")
