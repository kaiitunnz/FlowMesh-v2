"""Schemas and enums for the topology-aware network plane.

These are the four separate facts the substrate resolves a route from: a
``NetworkEndpointAdvertisement`` (an operator-configured node/ingress endpoint), a
non-secret ``ReplicaListenerAdvertisement`` (the resident-facing sidecar capability of a
replica incarnation), classified ``RouteObservation``s feeding the derived reachability
view, and a control-bound ``RouteOrigin`` the pure resolver produces a ``ResolvedRoute``
for. None of these carries admission credit, semantic authority, or a route
authorization; a route observation is network evidence only.
"""

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from shared.schemas.network import NetworkEndpointAdvertisement, ReachabilityClass

from ..utils.time import now_iso

__all__ = [
    "NetworkEndpointAdvertisement",
    "NonresidentSidecarTarget",
    "PolicyClass",
    "ReachabilityClass",
    "ReachabilityEntry",
    "ReachabilityState",
    "ReplicaListenerAdvertisement",
    "ResolvedRoute",
    "RouteCandidate",
    "RouteHop",
    "RouteObservation",
    "RouteObservationOutcome",
    "RouteOrigin",
    "RouteTarget",
    "Transport",
    "is_demoting",
]


class PolicyClass(StrEnum):
    """The network/policy class scoping a route origin and its observations."""

    DEFAULT = "default"


class Transport(StrEnum):
    """A generic route transport candidate.

    ``worker_direct`` is caller-to-listener; ``node_relay`` goes through the
    replica-node endpoint and its node-local uplink; ``control_relay`` is the bounded
    controlled fallback. They are ordered into a candidate ladder by the resolver.
    """

    WORKER_DIRECT = "worker_direct"
    NODE_RELAY = "node_relay"
    CONTROL_RELAY = "control_relay"


class ReachabilityState(StrEnum):
    """The derived directional reachability of one (origin, target, transport) key."""

    UNKNOWN = "unknown"
    OPTIMISTIC = "optimistic"
    VERIFIED = "verified"
    DEMOTED = "demoted"


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


class RouteTarget(Protocol):
    """The route-target facts the pure resolver reads.

    Both the resident ``ReplicaListenerAdvertisement`` and the nonresident external-tool
    ``NonresidentSidecarTarget`` satisfy it, so one resolver serves both without either
    carrying the other's contract. ``incarnation`` and ``listener_generation`` are
    distinct reachability-view keys (a resident target moves them independently); a
    nonresident target mirrors both to its target generation.
    """

    node_id: str
    incarnation: int
    listener_generation: int
    routes: tuple[str, ...]
    directly_routable: bool


class ReplicaListenerAdvertisement(BaseModel):
    """The non-secret resident-facing listener of one replica incarnation.

    It names the sidecar/adapter capability and route endpoint(s), fenced by replica
    incarnation and listener generation. It is never the raw engine listener or an
    engine credential; a route resolves to this, not to a resident engine port.
    """

    model_config = ConfigDict(frozen=True)

    replica_id: str
    family: str
    incarnation: int
    listener_generation: int
    node_id: str
    worker_id: str | None = None
    routes: tuple[str, ...] = ()
    protocols: tuple[str, ...] = ()
    directly_routable: bool = False


class NonresidentSidecarTarget(BaseModel):
    """A control-issued, purpose-bound egress sidecar target reached over the network.

    Claim-free: it mints no admission credit and carries no resident replica/family/
    incarnation fact. ``target_id`` and ``target_generation`` are its audience fence — a
    bound sidecar rejects an operation naming another target id/generation.
    ``interfaces`` are the tool interfaces it serves and ``provider`` the egress
    provider it is provisioned for. ``incarnation`` and ``listener_generation`` mirror
    ``target_generation`` so the resolver's reachability keys stay stable per
    generation.
    """

    model_config = ConfigDict(frozen=True)

    target_id: str
    target_generation: int
    node_id: str
    worker_id: str | None = None
    incarnation: int
    listener_generation: int
    interfaces: tuple[str, ...] = ()
    provider: str = ""
    routes: tuple[str, ...] = ()
    protocols: tuple[str, ...] = ()
    directly_routable: bool = False


class RouteOrigin(BaseModel):
    """A trusted caller origin, bound by control to a registered source endpoint.

    ``origin_id`` is an unguessable control-bound token; the origin is the actual
    execution-network source (a worker-side deputy), not the logical caller, and it
    never discovers peers or owns admission policy.
    """

    model_config = ConfigDict(frozen=True)

    origin_id: str
    endpoint_id: str
    node_id: str | None = None
    reachability_class: ReachabilityClass
    policy_class: PolicyClass = PolicyClass.DEFAULT
    trust_domain: str
    relay_attachment_id: str | None = None


class RouteObservation(BaseModel):
    """Network evidence about one attempted directional route.

    It updates only the derived reachability view; it can never promote, release, or
    overwrite a ``ServiceClaim`` credit.
    """

    model_config = ConfigDict(frozen=True)

    origin_id: str
    policy_class: PolicyClass
    target_node_id: str
    incarnation: int
    listener_generation: int
    transport: Transport
    outcome: RouteObservationOutcome
    at: str = Field(default_factory=now_iso)


class ReachabilityEntry(BaseModel):
    """The derived state of one directional (origin, target, gen, transport) key.

    Keyed per transport so a demoted direct path does not taint the relay fallback.

    ``expires_at`` carries the positive/negative TTL; ``backoff_until`` and ``retries``
    bound optimistic re-attempts after a demotion.
    """

    origin_id: str
    policy_class: PolicyClass
    target_node_id: str
    incarnation: int
    listener_generation: int
    transport: Transport
    state: ReachabilityState = ReachabilityState.UNKNOWN
    expires_at: float | None = None
    backoff_until: float | None = None
    retries: int = 0
    updated_at: str = Field(default_factory=now_iso)


class RouteHop(BaseModel):
    """One hop of a resolved candidate path.

    For a forward-dial ``worker_direct`` / ``node_relay`` hop, ``endpoint`` is a
    dialable address. For a ``control_relay`` hop ``node_id`` is the attaching node the
    delivery routes by, and ``endpoint`` on the terminal hop is the node-local sidecar
    delivery route, never a peer-dialable address.
    """

    model_config = ConfigDict(frozen=True)

    transport: Transport
    endpoint: str
    node_id: str | None = None


class RouteCandidate(BaseModel):
    """One transport alternative in the ordered candidate ladder."""

    model_config = ConfigDict(frozen=True)

    transport: Transport
    hops: tuple[RouteHop, ...]


class ResolvedRoute(BaseModel):
    """An ordered, expiry-bounded candidate-path snapshot for a trusted origin.

    It is non-authoritative: it issues no authority, chooses no capacity, and mutates no
    ``ServiceClaim``. The deputy executes only these candidates in order and never scans
    for a peer.
    """

    model_config = ConfigDict(frozen=True)

    origin_id: str
    target_node_id: str
    listener_generation: int
    route_epoch: int
    candidates: tuple[RouteCandidate, ...]
    expires_at: float | None = None
