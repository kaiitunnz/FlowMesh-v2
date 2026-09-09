"""The worker-hosted forward serve ingress's control-plane messages.

A forward ingress authenticates nothing itself. It hands the presented credential and
the frozen request's bounded descriptor to control over its authenticated attachment and
serves the request only on control's admission. These are the shapes that cross that
boundary: the ingress-host advertisement a worker registers, the reserve control sends
back to bind one task's port, the bound evidence the worker returns, and the admission
request the ingress sends up. The admission request carries the descriptor, never the
raw body, which stays worker-private behind the digest.
"""

from pydantic import BaseModel, ConfigDict


class ServeIngressRequest(BaseModel):
    """What a forward ingress asks control to authenticate, authorize, and admit.

    It carries the frozen request's bounded descriptor and the presented credential,
    never the raw body: the body stays worker-private behind the digest, so control
    admits the request without the payload ever entering control state. The credential
    rides only this control message and reaches neither the data path, the logs, nor the
    engine. ``serve_task_id`` and the two generations name the port exposure the worker
    holds for the arrival port — control-assigned at reserve, never a client-supplied
    path — so control resolves the task from its own live exposure. ``path`` is the
    engine-native origin-form path, carried verbatim with no route prefix.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    serve_task_id: str
    binding_generation: int = 0
    exposure_generation: int = 0
    credential: str | None = None
    method: str
    path: str
    query: str = ""
    descriptor_digest: str
    body_bytes: int = 0


class ServeIngressAdvertisement(BaseModel):
    """A forward ingress host's registration: its public authority and port range.

    The worker reports the operator-configured public authority, the port range it may
    bind for per-task exposures, its TLS profile generation (``0`` for an explicit
    plaintext local-test host), and a monotonic generation the directory fences a
    superseding registration by. Control derives the ingress's route origin from the
    reporting worker's own node advertisement, so no address beyond the authority
    crosses.
    """

    model_config = ConfigDict(frozen=True)

    authority: str
    port_low: int
    port_high: int
    tls_profile_generation: int = 0
    generation: int = 0


class ServeIngressReserve(BaseModel):
    """Control's instruction to bind one task's public forward port.

    Sent to the ingress host worker when a forward binding is adopted (or when a host
    registers after adoption). The worker binds a listener on ``public_port`` mapped to
    this exposure and answers with a ``ServeIngressBound``. ``tls`` selects HTTPS from
    the host's operator TLS profile; a plaintext local-test host binds plain HTTP.
    """

    model_config = ConfigDict(frozen=True)

    serve_task_id: str
    binding_generation: int
    exposure_generation: int
    public_port: int
    tls: bool = False


class ServeIngressBound(BaseModel):
    """The ingress worker's ready evidence that a reserved port is bound and serving.

    Control commits the exposure ``LIVE`` and publishes its url only from this evidence:
    a bound socket alone is not enough, so the worker returns the listener generation it
    bound under and its ingress attachment generation, both fenced by the exposure
    generation the reserve carried.
    """

    model_config = ConfigDict(frozen=True)

    serve_task_id: str
    binding_generation: int
    exposure_generation: int
    listener_generation: int
    attachment_generation: int
