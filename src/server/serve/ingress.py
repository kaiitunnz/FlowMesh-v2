"""The gated serve access modes and the root-local proxy ingress registration.

A serve task's binding pins one gated HTTP exposure mode. ``proxy``, the default,
terminates at the root-local proxy ingress: it holds a root-internal rendezvous
attachment rather than a public listener, and because the root cannot dial a worker its
frames always ride the universal ``control_relay``. ``forward`` terminates at a
worker-hosted, per-task port exposure a deployment registers through the
``ForwardIngressDirectory``; this registry holds only the root-local proxy.

The proxy ingress is a transport-only route origin over the binding and claim path, and
is reachable only through central authentication and admission. A proxy request whose
ingress a deployment has not registered is unavailable: it fails closed rather than
exposing a raw listener.
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
    """The registered root-local proxy ingress: its transport-only route origin.

    ``origin_id`` names the node the network plane derives the route origin from. A
    client reaches the proxy at the server's own base url, so it advertises no address
    of its own.
    """

    mode: ServeAccessMode
    origin_id: str


class ServeIngressRegistry:
    """The root-local proxy ingress this deployment has registered.

    The proxy ingress is internal to the root, so it is registered whenever a deployment
    permits it; an operator that refuses public proxy serve exposure registers none and
    every proxy request fails closed. Forward exposure is not held here — each forward
    binding owns a port exposure in the ``ForwardIngressDirectory``.
    """

    def __init__(self, proxy_origin_id: str | None) -> None:
        self._by_mode: dict[ServeAccessMode, ServeIngress] = {}
        if proxy_origin_id is not None:
            self._by_mode[ServeAccessMode.PROXY] = ServeIngress(
                mode=ServeAccessMode.PROXY, origin_id=proxy_origin_id
            )

    def live(self, mode: ServeAccessMode) -> ServeIngress | None:
        """The registered ingress for a mode, or None when none is registered."""
        return self._by_mode.get(mode)
