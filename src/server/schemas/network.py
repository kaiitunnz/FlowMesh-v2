"""REST schemas for the feature-gated network-plane echo and diagnostics."""

from pydantic import BaseModel, Field


class NetworkListenerBody(BaseModel):
    """The target listener an echo resolves a route to.

    It stands in for a resident-facing sidecar; this seam never fronts a real engine.
    ``routes`` are the sidecar addresses; ``directly_routable`` gates the direct path.
    """

    replica_id: str = Field(description="Target replica/listener id.")
    family: str = Field(default="echo-test", description="Target service family.")
    node_id: str = Field(description="Node hosting the target listener.")
    incarnation: int = Field(default=1, description="Replica incarnation fence.")
    listener_generation: int = Field(
        default=0, description="Listener generation fence."
    )
    routes: list[str] = Field(
        default_factory=list, description="Sidecar route endpoints (host:port)."
    )
    directly_routable: bool = Field(
        default=False, description="Whether a direct worker path is advertised."
    )


class NetworkEchoRequest(BaseModel):
    origin_node_id: str = Field(description="Node whose deputy executes the route.")
    listener: NetworkListenerBody = Field(description="Target listener to reach.")
    payload: str = Field(default="ping", description="Echo payload.")
    app_error: bool = Field(
        default=False, description="Ask the sidecar for an application error."
    )


class NetworkEchoResponse(BaseModel):
    selected_transport: str | None = Field(
        default=None, description="Transport that carried the echo, if any."
    )
    echoed: str | None = Field(default=None, description="Echoed payload, if verified.")
    route_epoch: int = Field(description="Resolved-route epoch.")
    candidates: list[str] = Field(description="Ordered candidate transports.")
    reachability: dict[str, str] = Field(
        description="Directional reachability state per transport after the attempt."
    )


class NetworkEndpointInfo(BaseModel):
    endpoint_id: str
    node_id: str | None = None
    url: str
    generation: int
    trust_domain: str
    reachability_class: str
    protocols: list[str] = Field(default_factory=list)


class NetworkReachabilityEntryInfo(BaseModel):
    origin_id: str
    target_node_id: str
    incarnation: int
    listener_generation: int
    transport: str
    state: str
    retries: int
