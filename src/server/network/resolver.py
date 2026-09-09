"""The pure network route resolver.

``resolve_route`` is a pure control-plane function: given the logical ``RouteOrigin``,
the root origin that opens any forward-dialed target leg, the target replica listener,
the target node's endpoint advertisement, and a snapshot of the derived reachability
view, it returns an ordered, expiry-bounded ``ResolvedRoute`` candidate ladder. It is
pure — it mutates nothing and permits no peer discovery; a deputy executes only the
candidates it returns.

The two origins are distinct sources. ``control_relay`` runs from the logical origin's
own outbound attachment, and the resolved route keeps that origin as its identity. The
forward-dial offloads are opened by the root, so they are graded and their reachability
keyed against the root's origin — the root is the physical target-leg bridge while the
logical origin remains the source.

Ladder rules:
- ``worker_direct`` is legal only when the listener is explicitly directly routable and
  the root's network class can reach the target endpoint's class. Shared-node placement
  alone is not sufficient.
- ``node_relay`` goes through the target node's announced endpoint and its node-local
  uplink; it is the initial same-node path as well as the normal cross-node path.
- ``control_relay`` is the universal reverse-rendezvous base: the root bridges between
  the origin and target reverse-relay attachments to the target's node-local sidecar
  delivery. Its feasibility is that both ends have a registered outbound attachment, not
  that either is inbound-reachable, so it is the guaranteed base whenever both
  attachments are present — including for an outbound-only node with no inbound URL.

Each forward-dial candidate carries whether the deployment's trust policy admits it as
an offload for claim-gated resident traffic; the reachability diagnostic still measures
one the policy does not admit.

Candidates a demotion has removed drop out; among those left, verified paths precede
untried ones, and within a rank the base preference is direct, then node relay, then
the control relay.
"""

from .reachability import NetworkReachabilityView
from .state import (
    NetworkEndpointAdvertisement,
    ReachabilityClass,
    ReachabilityState,
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    RouteOrigin,
    RouteTarget,
    TargetLegTrustPolicy,
    Transport,
)

_CLASS_LOCALITY: dict[ReachabilityClass, int] = {
    ReachabilityClass.SAME_NODE: 0,
    ReachabilityClass.SAME_CLUSTER: 1,
    ReachabilityClass.ROUTABLE: 2,
}


def _class_reachable(origin: ReachabilityClass, target: ReachabilityClass) -> bool:
    """Whether an origin at its locality can reach an endpoint exposed at ``target``.

    A more local origin reaches broader endpoints; a routable-only origin reaches only a
    routable endpoint.
    """
    return _CLASS_LOCALITY[origin] <= _CLASS_LOCALITY[target]


def _base_index(transport: Transport) -> int:
    return {
        Transport.WORKER_DIRECT: 0,
        Transport.NODE_RELAY: 1,
        Transport.CONTROL_RELAY: 2,
    }[transport]


def _admits(
    trust: TargetLegTrustPolicy,
    node_endpoint: NetworkEndpointAdvertisement,
    listener: RouteTarget,
    transport: Transport,
) -> bool:
    """Whether the trust policy admits this forward-dial candidate as an offload.

    The target must sit in the required trust domain and a trusted reachability class,
    and advertise the mutually authenticated transport — the node for both offloads, the
    listener itself for a direct dial, which also needs the node's purpose-scoped
    target-leg listener for a node relay.
    """
    if not trust.enabled:
        return False
    if node_endpoint.trust_domain != trust.trust_domain:
        return False
    if node_endpoint.reachability_class not in trust.classes:
        return False
    if trust.protocol not in node_endpoint.protocols:
        return False
    if transport is Transport.NODE_RELAY:
        return bool(node_endpoint.target_leg_url)
    return trust.protocol in listener.protocols


def resolve_route(
    origin: RouteOrigin,
    listener: RouteTarget,
    node_endpoint: NetworkEndpointAdvertisement | None,
    reachability: NetworkReachabilityView,
    *,
    target_leg_origin: RouteOrigin,
    trust: TargetLegTrustPolicy,
    now: float,
    route_epoch: int,
    expires_at: float | None = None,
) -> ResolvedRoute:
    """Resolve the ordered candidate ladder for one origin/target pair.

    The ``control_relay`` base is carried whenever the origin and the target node both
    advertise an outbound reverse-relay attachment; it names those attachments and the
    target's node-local sidecar delivery, and the root bridges between them.
    """
    graded: list[tuple[int, int, RouteCandidate]] = []

    def consider(
        transport: Transport,
        hops: tuple[RouteHop, ...],
        grading_origin: RouteOrigin,
        trusted: bool = False,
    ) -> None:
        state = reachability.state_for(
            grading_origin.origin_id,
            grading_origin.policy_class,
            listener.node_id,
            listener.incarnation,
            listener.listener_generation,
            transport,
            now=now,
        )
        # A control_relay is the guaranteed fallback and is never dropped; a demoted
        # direct or node path falls out until its backoff cools.
        if (
            state is ReachabilityState.DEMOTED
            and transport is not Transport.CONTROL_RELAY
        ):
            return
        rank = 0 if state is ReachabilityState.VERIFIED else 1
        graded.append(
            (
                rank,
                _base_index(transport),
                RouteCandidate(transport=transport, hops=hops, trusted=trusted),
            )
        )

    direct_route = listener.routes[0] if listener.routes else None
    if (
        listener.directly_routable
        and direct_route is not None
        and node_endpoint is not None
        and _class_reachable(
            target_leg_origin.reachability_class, node_endpoint.reachability_class
        )
    ):
        consider(
            Transport.WORKER_DIRECT,
            (
                RouteHop(
                    transport=Transport.WORKER_DIRECT,
                    endpoint=direct_route,
                    node_id=listener.node_id,
                ),
            ),
            target_leg_origin,
            _admits(trust, node_endpoint, listener, Transport.WORKER_DIRECT),
        )

    if node_endpoint is not None and direct_route is not None:
        trusted_node_relay = _admits(
            trust, node_endpoint, listener, Transport.NODE_RELAY
        )
        # A trusted offload enters the node's purpose-scoped target-leg listener; the
        # diagnostic path enters its announced endpoint.
        node_entry = (
            node_endpoint.target_leg_url if trusted_node_relay else node_endpoint.url
        )
        if node_entry:
            consider(
                Transport.NODE_RELAY,
                (
                    RouteHop(
                        transport=Transport.NODE_RELAY,
                        endpoint=node_entry,
                        node_id=node_endpoint.node_id,
                    ),
                    RouteHop(
                        transport=Transport.NODE_RELAY,
                        endpoint=direct_route,
                        node_id=listener.node_id,
                    ),
                ),
                target_leg_origin,
                trusted_node_relay,
            )

    # control_relay is the universal reverse-rendezvous base: the root bridges between
    # the origin and target reverse-relay attachments to the target's node-local sidecar
    # delivery. It needs both ends attached outward and a sidecar delivery route; it
    # does not need either end inbound-reachable, so it is the guaranteed base whenever
    # both attachments are present, including an outbound-only node with no inbound URL.
    target_attachment = (
        node_endpoint.relay_attachment_id if node_endpoint is not None else None
    )
    if (
        origin.relay_attachment_id is not None
        and target_attachment is not None
        and direct_route is not None
    ):
        consider(
            Transport.CONTROL_RELAY,
            (
                RouteHop(
                    transport=Transport.CONTROL_RELAY,
                    endpoint="",
                    node_id=origin.node_id,
                ),
                RouteHop(
                    transport=Transport.CONTROL_RELAY,
                    endpoint=direct_route,
                    node_id=listener.node_id,
                ),
            ),
            origin,
        )

    graded.sort(key=lambda item: (item[0], item[1]))
    return ResolvedRoute(
        origin_id=origin.origin_id,
        target_node_id=listener.node_id,
        listener_generation=listener.listener_generation,
        route_epoch=route_epoch,
        candidates=tuple(candidate for _, _, candidate in graded),
        expires_at=expires_at,
    )
