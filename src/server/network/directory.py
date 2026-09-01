"""The network endpoint directory.

An in-memory index of the current ``NetworkEndpointAdvertisement`` per endpoint, rebuilt
from the node registry on startup (the node record is the durable carrier, so the
directory holds no independent persistence and cannot drift from it). ``generation`` is
the fence: a regressing advertisement is rejected and an advancing one signals that
stale route evidence for that node should be invalidated.
"""

from .state import NetworkEndpointAdvertisement


class NetworkEndpointDirectory:
    """Latest endpoint advertisement per endpoint id, keyed for lookup by node."""

    def __init__(self) -> None:
        self._by_endpoint: dict[str, NetworkEndpointAdvertisement] = {}

    def upsert(self, adv: NetworkEndpointAdvertisement) -> bool:
        """Record ``adv`` unless a same-or-newer generation is already held.

        Returns True when the stored generation advanced (or the endpoint is new), so a
        caller can invalidate route evidence bound to the superseded generation.
        """
        current = self._by_endpoint.get(adv.endpoint_id)
        if current is not None and adv.generation <= current.generation:
            return False
        self._by_endpoint[adv.endpoint_id] = adv
        return True

    def get(self, endpoint_id: str) -> NetworkEndpointAdvertisement | None:
        return self._by_endpoint.get(endpoint_id)

    def by_node(self, node_id: str) -> NetworkEndpointAdvertisement | None:
        for adv in self._by_endpoint.values():
            if adv.node_id == node_id:
                return adv
        return None

    def remove(self, endpoint_id: str) -> None:
        self._by_endpoint.pop(endpoint_id, None)

    def remove_node(self, node_id: str) -> None:
        for endpoint_id in [
            eid for eid, adv in self._by_endpoint.items() if adv.node_id == node_id
        ]:
            self._by_endpoint.pop(endpoint_id, None)

    def all(self) -> list[NetworkEndpointAdvertisement]:
        return list(self._by_endpoint.values())

    def rebuild(self, advertisements: list[NetworkEndpointAdvertisement]) -> None:
        """Replace the index from the node registry's current advertisements."""
        self._by_endpoint = {adv.endpoint_id: adv for adv in advertisements}
