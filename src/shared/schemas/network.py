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


class NetworkEndpointAdvertisement(BaseModel):
    """A node's (or registered ingress edge's) purpose-scoped network-plane endpoint.

    Operator-configured and identity/TLS-bound at the source, not a worker-supplied
    arbitrary URL and not the generic server-management endpoint. ``generation`` is the
    monotonic fence: re-registration mints a fresh generation so stale advertisements
    and route evidence keyed to an older one are never used.

    ``relay_attachment_id`` / ``relay_attachment_generation`` are the non-secret
    identity and monotonic fence of this node's (or ingress edge's) outbound relay
    attachment to the root rendezvous. They prove the node can attach outward for the
    universal reverse relay; they are not an inbound URL a peer may dial. A changed
    attachment generation invalidates route evidence and base relay candidates keyed to
    the older one.
    """

    model_config = ConfigDict(frozen=True)

    endpoint_id: str
    node_id: str | None = None
    url: str
    generation: int
    trust_domain: str
    reachability_class: ReachabilityClass
    protocols: tuple[str, ...] = ()
    relay_attachment_id: str | None = None
    relay_attachment_generation: int = 0


__all__ = ["NetworkEndpointAdvertisement", "ReachabilityClass"]
