"""Trusted target-leg eligibility and the two origins the resolver grades against."""

from server.network import NetworkReachabilityView, resolve_route
from server.network.state import (
    NetworkEndpointAdvertisement,
    PolicyClass,
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    RouteObservation,
    RouteObservationOutcome,
    RouteOrigin,
    TargetLegTrustPolicy,
    Transport,
)
from shared.schemas.network import TARGET_LEG_PROTOCOL

_TRUST = TargetLegTrustPolicy(
    enabled=True,
    trust_domain="fm",
    classes=frozenset({ReachabilityClass.SAME_CLUSTER}),
    protocol=TARGET_LEG_PROTOCOL,
)


def _origin(
    origin_id: str,
    reachability_class: ReachabilityClass = ReachabilityClass.SAME_CLUSTER,
) -> RouteOrigin:
    return RouteOrigin(
        origin_id=origin_id,
        endpoint_id=f"e-{origin_id}",
        node_id=f"nde-{origin_id}",
        reachability_class=reachability_class,
        policy_class=PolicyClass.DEFAULT,
        trust_domain="fm",
        relay_attachment_id=f"att-{origin_id}",
    )


def _listener(
    *, protocols: tuple[str, ...] = ("resident", TARGET_LEG_PROTOCOL)
) -> ReplicaListenerAdvertisement:
    return ReplicaListenerAdvertisement(
        replica_id="rpl-1",
        family="fam",
        incarnation=1,
        listener_generation=3,
        node_id="nde-target",
        routes=("10.0.0.4:41000",),
        protocols=protocols,
        directly_routable=True,
    )


def _endpoint(
    *,
    trust_domain: str = "fm",
    reachability_class: ReachabilityClass = ReachabilityClass.SAME_CLUSTER,
    protocols: tuple[str, ...] = (TARGET_LEG_PROTOCOL,),
    target_leg_url: str = "10.0.0.4:41100",
) -> NetworkEndpointAdvertisement:
    return NetworkEndpointAdvertisement(
        endpoint_id="e-target",
        node_id="nde-target",
        url="10.0.0.4:9101",
        target_leg_url=target_leg_url,
        generation=1,
        trust_domain=trust_domain,
        reachability_class=reachability_class,
        protocols=protocols,
        relay_attachment_id="att-target",
    )


def _resolve(
    view: NetworkReachabilityView,
    *,
    origin: RouteOrigin | None = None,
    root: RouteOrigin | None = None,
    listener: ReplicaListenerAdvertisement | None = None,
    endpoint: NetworkEndpointAdvertisement | None = None,
    trust: TargetLegTrustPolicy = _TRUST,
):
    return resolve_route(
        origin or _origin("worker"),
        listener or _listener(),
        endpoint if endpoint is not None else _endpoint(),
        view,
        target_leg_origin=root or _origin("root"),
        trust=trust,
        now=0.0,
        route_epoch=1,
    )


def _trusted(route) -> list[str]:
    return [c.transport.value for c in route.candidates if c.trusted]


def test_a_configured_trusted_pair_admits_both_forward_dial_offloads() -> None:
    route = _resolve(NetworkReachabilityView())
    assert _trusted(route) == ["worker_direct", "node_relay"]


def test_an_unconfigured_deployment_admits_no_offload() -> None:
    route = _resolve(NetworkReachabilityView(), trust=TargetLegTrustPolicy())
    assert _trusted(route) == []
    assert "control_relay" in [c.transport.value for c in route.candidates]


def test_another_trust_domain_admits_no_offload() -> None:
    route = _resolve(
        NetworkReachabilityView(), endpoint=_endpoint(trust_domain="other")
    )
    assert _trusted(route) == []


def test_an_untrusted_reachability_class_admits_no_offload() -> None:
    route = _resolve(
        NetworkReachabilityView(),
        endpoint=_endpoint(reachability_class=ReachabilityClass.ROUTABLE),
    )
    assert _trusted(route) == []


def test_a_node_without_the_authenticated_transport_admits_no_offload() -> None:
    route = _resolve(NetworkReachabilityView(), endpoint=_endpoint(protocols=()))
    assert _trusted(route) == []


def test_a_listener_missing_the_authenticated_transport_admits_the_node_relay() -> None:
    route = _resolve(NetworkReachabilityView(), listener=_listener(protocols=()))
    assert _trusted(route) == ["node_relay"]


def test_a_node_without_a_target_leg_listener_admits_only_the_direct_dial() -> None:
    route = _resolve(NetworkReachabilityView(), endpoint=_endpoint(target_leg_url=""))
    assert _trusted(route) == ["worker_direct"]


def test_a_trusted_node_relay_enters_the_purpose_scoped_listener() -> None:
    route = _resolve(NetworkReachabilityView())
    node_relay = next(
        c for c in route.candidates if c.transport is Transport.NODE_RELAY
    )
    assert node_relay.hops[0].endpoint == "10.0.0.4:41100"


def test_an_inadmissible_node_relay_keeps_the_announced_endpoint() -> None:
    route = _resolve(NetworkReachabilityView(), trust=TargetLegTrustPolicy())
    node_relay = next(
        c for c in route.candidates if c.transport is Transport.NODE_RELAY
    )
    assert node_relay.hops[0].endpoint == "10.0.0.4:9101"


def test_the_root_class_governs_the_direct_dial_not_the_logical_origin() -> None:
    # The logical origin cannot reach a same-cluster endpoint, but it never dials one:
    # the root opens the target leg, so the root's class is what admits the candidate.
    route = _resolve(
        NetworkReachabilityView(),
        origin=_origin("worker", ReachabilityClass.ROUTABLE),
    )
    assert "worker_direct" in [c.transport.value for c in route.candidates]

    unreachable_root = _resolve(
        NetworkReachabilityView(),
        root=_origin("root", ReachabilityClass.ROUTABLE),
    )
    assert "worker_direct" not in [
        c.transport.value for c in unreachable_root.candidates
    ]


def test_target_leg_evidence_is_keyed_to_the_root_across_logical_origins() -> None:
    view = NetworkReachabilityView()
    root = _origin("root")
    view.observe(
        RouteObservation(
            origin_id=root.origin_id,
            policy_class=root.policy_class,
            target_node_id="nde-target",
            incarnation=1,
            listener_generation=3,
            transport=Transport.WORKER_DIRECT,
            outcome=RouteObservationOutcome.CONNECT_FAILURE,
        ),
        now=0.0,
    )
    # A different logical origin sees the same root-to-target demotion.
    route = _resolve(view, origin=_origin("other-worker"), root=root)
    assert "worker_direct" not in [c.transport.value for c in route.candidates]
