"""A content transfer resolves to a holder's content listener under its own evidence."""

from server.network import NetworkReachabilityView, resolve_route
from server.network.state import (
    ContentListenerTarget,
    NetworkEndpointAdvertisement,
    PolicyClass,
    ReachabilityClass,
    RouteObservation,
    RouteObservationOutcome,
    RouteOrigin,
    Transport,
    TrustedPeerPolicy,
)
from shared.schemas.network import CONTENT_PEER_PROTOCOL, PEER_PROTOCOL

CONTENT = TrustedPeerPolicy(
    enabled=True,
    trust_domain="fm",
    classes=frozenset(ReachabilityClass),
    protocol=CONTENT_PEER_PROTOCOL,
)
BOTH = (PEER_PROTOCOL, CONTENT_PEER_PROTOCOL)


def _origin(
    policy_class: PolicyClass = PolicyClass.CONTENT,
    protocols: tuple[str, ...] = BOTH,
) -> RouteOrigin:
    return RouteOrigin(
        origin_id="rog-1",
        endpoint_id="e-origin",
        node_id="nde-origin",
        reachability_class=ReachabilityClass.ROUTABLE,
        policy_class=policy_class,
        trust_domain="fm",
        protocols=protocols,
        relay_attachment_id="att-origin",
    )


def _holder(*, directly_routable: bool) -> ContentListenerTarget:
    return ContentListenerTarget(
        worker_id="wkr-holder",
        node_id="nde-1",
        incarnation=3,
        listener_generation=3,
        routes=("10.0.0.2:41000",),
        protocols=(CONTENT_PEER_PROTOCOL,),
        directly_routable=directly_routable,
    )


def _endpoint(protocols: tuple[str, ...] = BOTH) -> NetworkEndpointAdvertisement:
    return NetworkEndpointAdvertisement(
        endpoint_id="e-target",
        node_id="nde-1",
        url="10.0.0.2:9101",
        peer_url="10.0.0.2:9102",
        generation=1,
        trust_domain="fm",
        reachability_class=ReachabilityClass.ROUTABLE,
        protocols=protocols,
        relay_attachment_id="att-target",
    )


def _ladder(origin, target, endpoint, view=None, trust=CONTENT) -> list[Transport]:
    route = resolve_route(
        origin,
        target,
        endpoint,
        view or NetworkReachabilityView(),
        trust=trust,
        now=0.0,
        route_epoch=1,
    )
    return [candidate.transport for candidate in route.candidates]


def test_a_routable_holder_is_dialed_directly_then_through_its_node() -> None:
    assert _ladder(_origin(), _holder(directly_routable=True), _endpoint()) == [
        Transport.WORKER_DIRECT,
        Transport.NODE_RELAY,
        Transport.CONTROL_RELAY,
    ]


def test_a_holder_not_declared_routable_is_reached_through_its_node() -> None:
    assert _ladder(_origin(), _holder(directly_routable=False), _endpoint()) == [
        Transport.NODE_RELAY,
        Transport.CONTROL_RELAY,
    ]


def test_a_node_that_cannot_carry_content_is_never_a_content_hop() -> None:
    assert _ladder(
        _origin(), _holder(directly_routable=False), _endpoint((PEER_PROTOCOL,))
    ) == [Transport.CONTROL_RELAY]


def test_an_origin_that_cannot_dial_content_keeps_the_relay() -> None:
    assert _ladder(
        _origin(protocols=(PEER_PROTOCOL,)),
        _holder(directly_routable=True),
        _endpoint(),
    ) == [Transport.CONTROL_RELAY]


def test_an_untrusted_pair_keeps_the_relay() -> None:
    assert _ladder(
        _origin(),
        _holder(directly_routable=True),
        _endpoint(),
        trust=TrustedPeerPolicy(),
    ) == [Transport.CONTROL_RELAY]


def test_content_evidence_never_grades_another_policy_class() -> None:
    view = NetworkReachabilityView()
    holder = _holder(directly_routable=True)
    view.observe(
        RouteObservation(
            origin_id="rog-1",
            policy_class=PolicyClass.CONTENT,
            target_node_id=holder.node_id,
            incarnation=holder.incarnation,
            listener_generation=holder.listener_generation,
            transport=Transport.WORKER_DIRECT,
            outcome=RouteObservationOutcome.CONNECT_FAILURE,
        ),
        now=0.0,
    )
    assert Transport.WORKER_DIRECT not in _ladder(_origin(), holder, _endpoint(), view)
    assert (
        _ladder(_origin(PolicyClass.DEFAULT), holder, _endpoint(), view)[0]
        is Transport.WORKER_DIRECT
    )
