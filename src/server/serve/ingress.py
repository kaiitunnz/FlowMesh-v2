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


class ServeAccessMode(StrEnum):
    """The gated HTTP exposure a serve task's binding pins."""

    PROXY = "proxy"
    FORWARD = "forward"


@dataclass(frozen=True)
class ServeIngress:
    """One registered gated ingress: the transport-only origin a mode terminates at.

    ``origin_id`` names the registered edge the network plane derives the route origin
    from, and ``generation`` fences it so a superseded registration is refused.
    """

    mode: ServeAccessMode
    origin_id: str
    generation: int = 0


class ServeIngressRegistry:
    """The gated ingresses this deployment has registered, one per mode.

    The root-local proxy ingress is registered whenever the gated serve surface is up,
    since it is internal to the root. A forward ingress exists only where a deployment
    configured and registered one, so resolving that mode returns nothing until it has.
    """

    def __init__(self, proxy_origin_id: str) -> None:
        self._by_mode: dict[ServeAccessMode, ServeIngress] = {
            ServeAccessMode.PROXY: ServeIngress(
                mode=ServeAccessMode.PROXY, origin_id=proxy_origin_id
            )
        }

    def register_forward(self, origin_id: str, generation: int) -> None:
        """Register (or re-register at a newer generation) the forward ingress."""
        current = self._by_mode.get(ServeAccessMode.FORWARD)
        if current is not None and generation < current.generation:
            return
        self._by_mode[ServeAccessMode.FORWARD] = ServeIngress(
            mode=ServeAccessMode.FORWARD, origin_id=origin_id, generation=generation
        )

    def withdraw_forward(self) -> None:
        """Drop the forward ingress, so requests pinned to it fail closed again."""
        self._by_mode.pop(ServeAccessMode.FORWARD, None)

    def live(self, mode: ServeAccessMode) -> ServeIngress | None:
        """The registered ingress for a mode, or None when none is registered."""
        return self._by_mode.get(mode)
