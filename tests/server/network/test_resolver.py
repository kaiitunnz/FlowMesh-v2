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
    TrustedPeerPolicy,
)
from shared.schemas.network import PEER_PROTOCOL

# The ladder tests below exercise transport legality and ordering, so they resolve under
# a policy that admits the pair; the trust gate itself is covered separately.
TRUSTED = TrustedPeerPolicy(
    enabled=True,
    trust_domain="fm",
    classes=frozenset(ReachabilityClass),
    protocol=PEER_PROTOCOL,
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
        protocols=(PEER_PROTOCOL,),
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
        protocols=(PEER_PROTOCOL,),
        directly_routable=directly_routable,
    )


def _endpoint(
    reachability_class=ReachabilityClass.ROUTABLE, *, attached: bool = True
) -> NetworkEndpointAdvertisement:
    return NetworkEndpointAdvertisement(
        endpoint_id="e-target",
        node_id="nde-1",
        url="127.0.0.1:9101",
        peer_url="127.0.0.1:9102",
        generation=1,
        trust_domain="fm",
        reachability_class=reachability_class,
        protocols=(PEER_PROTOCOL,),
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
        trust=TRUSTED,
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
        trust=TRUSTED,
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
        trust=TRUSTED,
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
        trust=TRUSTED,
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
        trust=TRUSTED,
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
        trust=TRUSTED,
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
        trust=TRUSTED,
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
        trust=TRUSTED,
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
    resolve_route(
        origin, listener, endpoint, view, trust=TRUSTED, now=0.0, route_epoch=1
    )
    # Reading the view during resolution allocates no reachability entry.
    assert view.entries() == []
    # A resolve emits only candidates for the given pair, never a peer scan.
    route = resolve_route(
        origin, listener, endpoint, view, trust=TRUSTED, now=0.0, route_epoch=2
    )
    assert all(c.hops for c in route.candidates)
    assert route.route_epoch == 2


def _resolve(trust: TrustedPeerPolicy, **kwargs):
    return resolve_route(
        kwargs.pop("origin", None) or _origin(),
        kwargs.pop("listener", None) or _listener(directly_routable=True),
        kwargs.pop("endpoint", None) or _endpoint(ReachabilityClass.ROUTABLE),
        NetworkReachabilityView(),
        trust=trust,
        now=0.0,
        route_epoch=1,
    )


def test_an_undeclared_deployment_is_offered_only_the_relay() -> None:
    # The default posture: the target advertises a dialable address and is directly
    # routable, and it still gets the relay alone.
    route = _resolve(TrustedPeerPolicy())
    assert _transports(route) == ["control_relay"]


def test_a_target_outside_the_trust_domain_is_offered_only_the_relay() -> None:
    route = _resolve(
        TRUSTED.model_copy(update={"trust_domain": "other"}),
    )
    assert _transports(route) == ["control_relay"]


def test_an_origin_outside_the_trust_domain_is_offered_only_the_relay() -> None:
    origin = _origin().model_copy(update={"trust_domain": "other"})
    route = _resolve(TRUSTED, origin=origin)
    assert _transports(route) == ["control_relay"]


def test_a_class_the_policy_does_not_admit_is_offered_only_the_relay() -> None:
    trust = TRUSTED.model_copy(
        update={"classes": frozenset({ReachabilityClass.SAME_NODE})}
    )
    route = _resolve(trust, endpoint=_endpoint(ReachabilityClass.ROUTABLE))
    assert _transports(route) == ["control_relay"]


def test_an_origin_without_the_transport_capability_is_offered_only_the_relay() -> None:
    origin = _origin().model_copy(update={"protocols": ()})
    route = _resolve(TRUSTED, origin=origin)
    assert _transports(route) == ["control_relay"]


def test_a_listener_without_the_transport_capability_gets_no_worker_direct() -> None:
    listener = _listener(directly_routable=True).model_copy(update={"protocols": ()})
    route = _resolve(TRUSTED, listener=listener)
    assert "worker_direct" not in _transports(route)


def test_a_node_without_the_transport_capability_gets_no_node_relay() -> None:
    endpoint = _endpoint(ReachabilityClass.ROUTABLE).model_copy(
        update={"protocols": ()}
    )
    route = _resolve(TRUSTED, endpoint=endpoint)
    assert "node_relay" not in _transports(route)


def test_a_trusted_pair_is_offered_both_peers_ahead_of_the_relay() -> None:
    route = _resolve(TRUSTED)
    assert _transports(route) == ["worker_direct", "node_relay", "control_relay"]


def test_the_relay_survives_when_both_peers_are_inadmissible() -> None:
    # Dropping inadmissible peer transports must never leave a registered pair with
    # nothing to carry the attempt: the through-root fallback is the terminal candidate.
    route = _resolve(TrustedPeerPolicy())
    assert route.candidates
    assert route.candidates[-1].transport is Transport.CONTROL_RELAY


def test_a_diagnostic_route_is_marked_as_a_probe() -> None:
    # The mark is what keeps a probe's candidates — graded without the deployment's
    # trust policy — from being treated as admitted by a caller carrying an invocation.
    assert _resolve(TrustedPeerPolicy(enabled=True, probe=True)).probe
    assert not _resolve(TRUSTED).probe
