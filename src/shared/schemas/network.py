"""Cross-plane network-plane advertisement schemas.

``NetworkEndpointAdvertisement`` crosses the server/supervisor boundary on node
registration, so it lives in the shared schemas rather than in the server-only network
package. The server-side directory, reachability view, and resolver build on it.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

# The mutually authenticated transport a node and its replica listeners advertise when
# the deployment admits root-opened target-leg offloads.
TARGET_LEG_PROTOCOL = "resident-mtls"


class ReachabilityClass(StrEnum):
    """The operator-declared network class of an endpoint.

    An origin at its locality can reach a broader endpoint; a routable-only origin
    reaches only a routable endpoint. Shared-node placement alone does not make a direct
    path legal.
    """

    SAME_NODE = "same_node"
    SAME_CLUSTER = "same_cluster"
    ROUTABLE = "routable"


class NetworkEndpointAdvertisement(BaseModel):
    """A node's (or registered ingress edge's) purpose-scoped network-plane endpoint.

    Operator-configured and identity/TLS-bound at the source, not a worker-supplied
    arbitrary URL and not the generic server-management endpoint. ``generation`` is the
    monotonic fence: re-registration mints a fresh generation so stale advertisements
    and route evidence keyed to an older one are never used, which is also the fence
    that invalidates the node's relay evidence.

    ``target_leg_url`` is the node's purpose-scoped resident target-leg listener, which
    hands a session to the node's current local sidecar uplink; it is separate from the
    diagnostic ``url`` and is empty on a node that hosts no such listener.

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
    target_leg_url: str = ""
    generation: int
    trust_domain: str
    reachability_class: ReachabilityClass
    protocols: tuple[str, ...] = ()
    relay_attachment_id: str | None = None


__all__ = [
    "TARGET_LEG_PROTOCOL",
    "NetworkEndpointAdvertisement",
    "ReachabilityClass",
]
