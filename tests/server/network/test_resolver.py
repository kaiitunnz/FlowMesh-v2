"""The pure route resolver: ladder legality, ordering, fencing, and purity."""

from server.network import NetworkReachabilityView, resolve_route
from server.network.state import (
    NetworkEndpointAdvertisement,
    PolicyClass,
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    RouteObservation,
    RouteObservationOutcome,
    RouteOrigin,
    Transport,
)


def _origin(
    reachability_class=ReachabilityClass.SAME_NODE, *, attached: bool = True
) -> RouteOrigin:
    return RouteOrigin(
        origin_id="rog-1",
        endpoint_id="e-origin",
        node_id="nde-origin",
        reachability_class=reachability_class,
        policy_class=PolicyClass.DEFAULT,
        trust_domain="fm",
        relay_attachment_id="att-origin" if attached else None,
    )


def _listener(
    *, directly_routable: bool, node_id="nde-1"
) -> ReplicaListenerAdvertisement:
    return ReplicaListenerAdvertisement(
        replica_id="rpl-1",
        family="echo",
        incarnation=1,
        listener_generation=0,
        node_id=node_id,
        routes=("127.0.0.1:9001",),
        directly_routable=directly_routable,
    )


def _endpoint(
    reachability_class=ReachabilityClass.ROUTABLE, *, attached: bool = True
) -> NetworkEndpointAdvertisement:
    return NetworkEndpointAdvertisement(
        endpoint_id="e-target",
        node_id="nde-1",
        url="127.0.0.1:9101",
        generation=1,
        trust_domain="fm",
        reachability_class=reachability_class,
        relay_attachment_id="att-target" if attached else None,
    )


def _transports(route) -> list[str]:
    return [candidate.transport.value for candidate in route.candidates]


def test_colocated_not_directly_routable_uses_node_relay() -> None:
    view = NetworkReachabilityView()
    route = resolve_route(
        _origin(),
        _listener(directly_routable=False),
        _endpoint(ReachabilityClass.SAME_NODE),
        view,
        now=0.0,
        route_epoch=1,
    )
    # Shared-node placement alone does not add worker_direct.
    assert _transports(route) == ["node_relay", "control_relay"] or _transports(
        route
    ) == ["node_relay"]
    assert "worker_direct" not in _transports(route)


def test_directly_routable_and_usable_class_adds_worker_direct() -> None:
    view = NetworkReachabilityView()
    route = resolve_route(
        _origin(),
        _listener(directly_routable=True),
        _endpoint(ReachabilityClass.ROUTABLE),
        view,
        now=0.0,
        route_epoch=1,
    )
    assert _transports(route)[0] == "worker_direct"
    assert set(_transports(route)) >= {"worker_direct", "node_relay"}


def test_routable_origin_cannot_reach_same_node_endpoint() -> None:
    view = NetworkReachabilityView()
    route = resolve_route(
        _origin(ReachabilityClass.ROUTABLE),
        _listener(directly_routable=True),
        _endpoint(ReachabilityClass.SAME_NODE),
        view,
        now=0.0,
        route_epoch=1,
    )
    assert "worker_direct" not in _transports(route)


def test_no_control_relay_without_both_attachments() -> None:
    view = NetworkReachabilityView()
    # No target endpoint means no target attachment: the reverse relay is infeasible.
    no_target = resolve_route(
        _origin(),
        _listener(directly_routable=True),
        None,
        view,
        now=0.0,
        route_epoch=1,
    )
    assert "control_relay" not in _transports(no_target)
    # An unattached origin is equally infeasible even with a fully attached target.
    no_origin = resolve_route(
        _origin(attached=False),
        _listener(directly_routable=False),
        _endpoint(ReachabilityClass.SAME_NODE),
        view,
        now=0.0,
        route_epoch=1,
    )
    assert "control_relay" not in _transports(no_origin)


def test_demoted_direct_falls_out_of_ladder() -> None:
    view = NetworkReachabilityView()
    view.observe(
        RouteObservation(
            origin_id="rog-1",
            policy_class=PolicyClass.DEFAULT,
            target_node_id="nde-1",
            incarnation=1,
            listener_generation=0,
            transport=Transport.WORKER_DIRECT,
            outcome=RouteObservationOutcome.CONNECT_FAILURE,
        ),
        now=0.0,
    )
    route = resolve_route(
        _origin(),
        _listener(directly_routable=True),
        _endpoint(ReachabilityClass.ROUTABLE),
        view,
        now=0.1,
        route_epoch=1,
    )
    assert "worker_direct" not in _transports(route)
    assert "node_relay" in _transports(route)


def test_verified_candidate_is_preferred() -> None:
    view = NetworkReachabilityView()
    view.observe(
        RouteObservation(
            origin_id="rog-1",
            policy_class=PolicyClass.DEFAULT,
            target_node_id="nde-1",
            incarnation=1,
            listener_generation=0,
            transport=Transport.NODE_RELAY,
            outcome=RouteObservationOutcome.VERIFIED,
        ),
        now=0.0,
    )
    route = resolve_route(
        _origin(),
        _listener(directly_routable=True),
        _endpoint(ReachabilityClass.ROUTABLE),
        view,
        now=0.0,
        route_epoch=1,
    )
    # A verified node_relay outranks an untried worker_direct.
    assert _transports(route)[0] == "node_relay"


def test_control_relay_names_origin_and_target_attachments() -> None:
    view = NetworkReachabilityView()
    route = resolve_route(
        _origin(),
        _listener(directly_routable=False),
        _endpoint(ReachabilityClass.SAME_NODE),
        view,
        now=0.0,
        route_epoch=1,
    )
    control = [c for c in route.candidates if c.transport is Transport.CONTROL_RELAY][0]
    origin_hop, target_hop = control.hops
    # The descriptor names the origin and target ends by node (the delivery routes by
    # node id) and the target's node-local sidecar delivery, not dialable TCP hops.
    assert origin_hop.node_id == "nde-origin"
    assert origin_hop.endpoint == ""  # the origin end names no dialable address
    assert target_hop.node_id == "nde-1"
    assert target_hop.endpoint == "127.0.0.1:9001"  # local sidecar delivery route


def test_resolver_is_pure() -> None:
    view = NetworkReachabilityView()
    origin = _origin()
    listener = _listener(directly_routable=True)
    endpoint = _endpoint()
    resolve_route(origin, listener, endpoint, view, now=0.0, route_epoch=1)
    # Reading the view during resolution allocates no reachability entry.
    assert view.entries() == []
    # A resolve emits only candidates for the given pair, never a peer scan.
    route = resolve_route(origin, listener, endpoint, view, now=0.0, route_epoch=2)
    assert all(c.hops for c in route.candidates)
    assert route.route_epoch == 2
