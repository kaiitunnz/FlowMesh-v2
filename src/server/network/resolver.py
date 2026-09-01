"""The pure network route resolver.

``resolve_route`` is a pure control-plane function: given a trusted ``RouteOrigin``, the
target replica listener, the target node's endpoint advertisement, and a snapshot of the
derived reachability view, it returns an ordered, expiry-bounded ``ResolvedRoute``
candidate ladder. It is pure — it mutates nothing and permits no peer discovery; the
deputy executes only the candidates it returns.

Ladder rules:
- ``worker_direct`` is legal only when the listener is explicitly directly routable and
  the origin's network class can reach the target endpoint's class. Shared-node
  placement alone is not sufficient.
- ``node_relay`` goes through the target node's announced endpoint and its node-local
  uplink; it is the initial same-node path as well as the normal cross-node path.
- ``control_relay`` is the always-available controlled fallback.

Candidates a demotion has removed drop out; among those left, verified paths precede
untried ones, and within a rank the base preference is direct, then node relay, then
the control relay.
"""

from .reachability import NetworkReachabilityView
from .state import (
    NetworkEndpointAdvertisement,
    ReachabilityClass,
    ReachabilityState,
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    RouteOrigin,
    Transport,
)

_CONTROL_RELAY_ENDPOINT = "control-plane"

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


def resolve_route(
    origin: RouteOrigin,
    listener: ReplicaListenerAdvertisement,
    node_endpoint: NetworkEndpointAdvertisement | None,
    reachability: NetworkReachabilityView,
    *,
    now: float,
    route_epoch: int,
    control_relay_endpoint: str | None = None,
    expires_at: float | None = None,
) -> ResolvedRoute:
    """Resolve the ordered candidate ladder for one origin/target pair.

    ``control_relay_endpoint`` is the deployment's controlled-fallback relay; when set,
    the control-relay candidate carries it plus the target sidecar as explicit hops.
    """
    graded: list[tuple[int, int, RouteCandidate]] = []

    def consider(transport: Transport, hops: tuple[RouteHop, ...]) -> None:
        state = reachability.state_for(
            origin.origin_id,
            origin.policy_class,
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
                RouteCandidate(transport=transport, hops=hops),
            )
        )

    direct_route = listener.routes[0] if listener.routes else None
    if (
        listener.directly_routable
        and direct_route is not None
        and node_endpoint is not None
        and _class_reachable(
            origin.reachability_class, node_endpoint.reachability_class
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
        )

    if node_endpoint is not None and direct_route is not None:
        consider(
            Transport.NODE_RELAY,
            (
                RouteHop(
                    transport=Transport.NODE_RELAY,
                    endpoint=node_endpoint.url,
                    node_id=node_endpoint.node_id,
                ),
                RouteHop(
                    transport=Transport.NODE_RELAY,
                    endpoint=direct_route,
                    node_id=listener.node_id,
                ),
            ),
        )

    # control_relay needs a target sidecar to relay to (``direct_route``); a listener
    # with no advertised route therefore yields no candidate at all, by design.
    if control_relay_endpoint is not None and direct_route is not None:
        control_hops: tuple[RouteHop, ...] = (
            RouteHop(
                transport=Transport.CONTROL_RELAY,
                endpoint=control_relay_endpoint,
                node_id=None,
            ),
            RouteHop(
                transport=Transport.CONTROL_RELAY,
                endpoint=direct_route,
                node_id=listener.node_id,
            ),
        )
    else:
        control_hops = (
            RouteHop(
                transport=Transport.CONTROL_RELAY,
                endpoint=_CONTROL_RELAY_ENDPOINT,
                node_id=None,
            ),
        )
    consider(Transport.CONTROL_RELAY, control_hops)

    graded.sort(key=lambda item: (item[0], item[1]))
    return ResolvedRoute(
        origin_id=origin.origin_id,
        target_node_id=listener.node_id,
        listener_generation=listener.listener_generation,
        route_epoch=route_epoch,
        candidates=tuple(candidate for _, _, candidate in graded),
        expires_at=expires_at,
    )
