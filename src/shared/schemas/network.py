"""Cross-plane network-plane advertisement schemas.

``NetworkEndpointAdvertisement`` crosses the server/supervisor boundary on node
registration, so it lives in the shared schemas rather than in the server-only network
package. The server-side directory, reachability view, and resolver build on it.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ReachabilityClass(StrEnum):
    """The operator-declared network class of an endpoint.

    An origin at its locality can reach a broader endpoint; a routable-only origin
    reaches only a routable endpoint. Shared-node placement alone does not make a direct
    path legal.
    """

    SAME_NODE = "same_node"
    SAME_CLUSTER = "same_cluster"
    ROUTABLE = "routable"


OFFLOAD_PROTOCOL = "resident-offload"
"""The transport capability a direct origin-to-target offload is carried over."""


class NetworkEndpointAdvertisement(BaseModel):
    """A node's (or registered ingress edge's) purpose-scoped network-plane endpoint.

    Operator-configured and identity/TLS-bound at the source, not a worker-supplied
    arbitrary URL and not the generic server-management endpoint. ``generation`` is the
    monotonic fence: re-registration mints a fresh generation so stale advertisements
    and route evidence keyed to an older one are never used, which is also the fence
    that invalidates the node's relay evidence.

    ``offload_url`` is the node's purpose-scoped listener for a directly dialed
    offload, which hands a session to the node's current local sidecar uplink; it is
    separate from the diagnostic ``url`` and is empty on a node that hosts no such
    listener.

    ``relay_attachment_id`` is the non-secret identity of this node's (or ingress
    edge's) outbound relay attachment to the root rendezvous. It proves the node can
    attach outward for the universal reverse relay; it is not an inbound URL a peer may
    dial, so a node advertises it even without an inbound endpoint URL. ``url`` is empty
    for such an outbound-only node.
    """

    model_config = ConfigDict(frozen=True)

    endpoint_id: str
    node_id: str | None = None
    url: str
    offload_url: str = ""
    generation: int
    trust_domain: str
    reachability_class: ReachabilityClass
    protocols: tuple[str, ...] = ()
    relay_attachment_id: str | None = None


class Transport(StrEnum):
    """A generic route transport candidate.

    ``worker_direct`` is origin-to-listener; ``node_relay`` goes through the replica
    node's endpoint and its node-local uplink; ``control_relay`` is the bounded
    through-root fallback. They are ordered into a candidate ladder by the resolver.
    """

    WORKER_DIRECT = "worker_direct"
    NODE_RELAY = "node_relay"
    CONTROL_RELAY = "control_relay"


class RouteObservationOutcome(StrEnum):
    """The classified outcome of one attempted route.

    Only network-path failures demote reachability. Authority, tenant, fence,
    application, and engine failures are not evidence about the path and do not demote.
    """

    VERIFIED = "verified"
    DNS_FAILURE = "dns_failure"
    CONNECT_FAILURE = "connect_failure"
    TLS_FAILURE = "tls_failure"
    ROUTE_FAILURE = "route_failure"
    TIMEOUT = "timeout"
    AUTHORITY_DENIED = "authority_denied"
    TENANT_DENIED = "tenant_denied"
    FENCE_INVALID = "fence_invalid"
    APPLICATION_ERROR = "application_error"
    ENGINE_ERROR = "engine_error"


_DEMOTING_OUTCOMES: frozenset[RouteObservationOutcome] = frozenset(
    {
        RouteObservationOutcome.DNS_FAILURE,
        RouteObservationOutcome.CONNECT_FAILURE,
        RouteObservationOutcome.TLS_FAILURE,
        RouteObservationOutcome.ROUTE_FAILURE,
        RouteObservationOutcome.TIMEOUT,
    }
)


def is_demoting(outcome: RouteObservationOutcome) -> bool:
    """Whether the outcome is a network-path failure that may demote reachability."""
    return outcome in _DEMOTING_OUTCOMES


__all__ = [
    "OFFLOAD_PROTOCOL",
    "NetworkEndpointAdvertisement",
    "ReachabilityClass",
    "RouteObservationOutcome",
    "Transport",
    "is_demoting",
]
