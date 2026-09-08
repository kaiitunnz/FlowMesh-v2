"""The gated serve ingresses a deployment has registered.

A serve task's binding pins one gated HTTP exposure mode. ``proxy``, the default,
terminates at the root-local proxy ingress: it holds a root-internal rendezvous
attachment rather than a public listener, and because the root cannot dial a worker its
frames always ride the universal ``control_relay``. ``forward`` terminates at a
separately registered, externally reachable ingress hosted on a worker, whose own
network class may reach a replica sidecar directly and so may take a resolved
``worker_direct`` or ``node_relay`` offload with ``control_relay`` as the fallback.

Both are transport-only route origins over the same binding and claim path, and neither
is reachable except through central authentication and admission. A mode whose ingress a
deployment has not registered is unavailable: the request fails closed rather than
falling back to the other mode or exposing a raw listener.
"""

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse


class ServeAccessMode(StrEnum):
    """The gated HTTP exposure a serve task's binding pins."""

    PROXY = "proxy"
    FORWARD = "forward"


def is_public_base_url(url: str) -> bool:
    """Whether a url can serve as a public base a client resolves and dials."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


@dataclass(frozen=True)
class ServeIngress:
    """One registered gated ingress: the transport-only origin a mode terminates at.

    ``origin_id`` names the node the network plane derives the route origin from, so a
    forward ingress inherits its own node's reachability rather than advertising a
    second endpoint. ``worker_id`` addresses the control messages that reach it, and
    ``public_url`` is the base a client reaches it at — empty for the root-local proxy,
    which a client addresses at the server's own base url. ``generation`` fences the
    registration so a superseded one is refused.
    """

    mode: ServeAccessMode
    origin_id: str
    worker_id: str = ""
    public_url: str = ""
    generation: int = 0


class ServeIngressRegistry:
    """The gated ingresses this deployment has registered, one per mode.

    The root-local proxy ingress is internal to the root, so it is registered whenever a
    deployment permits it; an operator that refuses public serve exposure registers none
    and every proxy request fails closed. A forward ingress exists only where a
    deployment configured and registered one, so resolving that mode returns nothing
    until it has.
    """

    def __init__(self, proxy_origin_id: str | None) -> None:
        self._by_mode: dict[ServeAccessMode, ServeIngress] = {}
        if proxy_origin_id is not None:
            self._by_mode[ServeAccessMode.PROXY] = ServeIngress(
                mode=ServeAccessMode.PROXY, origin_id=proxy_origin_id
            )

    def register_forward(
        self,
        *,
        origin_id: str,
        worker_id: str,
        public_url: str,
        generation: int,
    ) -> bool:
        """Register (or re-register at a newer generation) the forward ingress.

        The url is the one an operator configured on that worker, reported once and
        fenced by its generation, so it is validated here rather than trusted: a
        registration that could not address the ingress is refused, as is one that a
        newer registration has already superseded.
        """
        if not is_public_base_url(public_url):
            return False
        current = self._by_mode.get(ServeAccessMode.FORWARD)
        if current is not None and generation < current.generation:
            return False
        self._by_mode[ServeAccessMode.FORWARD] = ServeIngress(
            mode=ServeAccessMode.FORWARD,
            origin_id=origin_id,
            worker_id=worker_id,
            public_url=public_url.rstrip("/"),
            generation=generation,
        )
        return True

    def withdraw_forward(self) -> None:
        """Drop the forward ingress, so requests pinned to it fail closed again."""
        self._by_mode.pop(ServeAccessMode.FORWARD, None)

    def live(self, mode: ServeAccessMode) -> ServeIngress | None:
        """The registered ingress for a mode, or None when none is registered."""
        return self._by_mode.get(mode)
