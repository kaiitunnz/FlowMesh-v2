"""The worker-hosted forward serve ingress's control-plane messages.

A forward ingress authenticates nothing itself. It hands the presented credential and
the frozen request's bounded descriptor to control over its authenticated attachment and
serves the request only on control's admission. These are the shapes that cross that
boundary: the admission request the ingress sends up, and the ingress advertisement it
registers. Both carry the descriptor, never the raw body, which stays worker-private
behind the digest.
"""

from pydantic import BaseModel, ConfigDict


class ServeIngressRequest(BaseModel):
    """What a forward ingress asks control to authenticate, authorize, and admit.

    It carries the frozen request's bounded descriptor and the presented credential,
    never the raw body: the body stays worker-private behind the digest, so control
    admits the request without the payload ever entering control state. The credential
    rides only this control message and reaches neither the data path, the logs, nor the
    engine.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    serve_task_id: str
    credential: str | None = None
    method: str
    path: str
    query: str = ""
    descriptor_digest: str
    body_bytes: int = 0


class ServeIngressAdvertisement(BaseModel):
    """A forward ingress's registration: the public base url clients reach it at.

    The worker reports the operator-configured public url and a monotonic generation the
    registry fences a superseding registration by. Control derives the ingress's route
    origin from the reporting worker's own node advertisement, so no address beyond the
    public url crosses here.
    """

    model_config = ConfigDict(frozen=True)

    public_url: str
    generation: int = 0
