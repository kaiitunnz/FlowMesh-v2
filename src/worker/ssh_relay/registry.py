"""Loopback ports this worker's relay lane may connect to, keyed by endpoint id."""

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class SshEndpointRegistry:
    """The ports a relayed session may reach, keyed by the id its executor names.

    An executor publishes an endpoint once its listener is ready and withdraws it
    once it stops. Resolution is the only authority on what a relay may reach: an id
    never published, or already withdrawn, resolves to nothing.
    """

    def __init__(self) -> None:
        self._ports: dict[str, int] = {}
        self._lock = threading.Lock()
        self._on_withdraw: list[Callable[[str], None]] = []

    def add_withdraw_listener(self, listener: Callable[[str], None]) -> None:
        self._on_withdraw.append(listener)

    def publish(self, endpoint_id: str, port: int) -> None:
        with self._lock:
            previous = self._ports.get(endpoint_id)
            self._ports[endpoint_id] = port
        if previous is not None and previous != port:
            logger.warning(
                "Endpoint %s republished on port %s, replacing port %s",
                endpoint_id,
                port,
                previous,
            )

    def withdraw(self, endpoint_id: str) -> None:
        """Stop offering an endpoint and end every relay connected to it."""
        with self._lock:
            self._ports.pop(endpoint_id, None)
        for listener in self._on_withdraw:
            listener(endpoint_id)

    def resolve(self, endpoint_id: str) -> int | None:
        with self._lock:
            return self._ports.get(endpoint_id)
