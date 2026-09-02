"""A network-plane node advertises so the resolver can carry the universal control_relay
base between it and any peer — including an outbound-only node with no inbound URL, for
which the server derives the relay attachment identity from the node."""

from server.config import NetworkPlaneConfig
from server.supervisor.supervisor import _endpoint_advertisement_provider


def test_carries_the_inbound_url_when_configured() -> None:
    provider = _endpoint_advertisement_provider(
        NetworkPlaneConfig(enabled=True, endpoint_url="127.0.0.1:9101")
    )
    first = provider()
    assert first is not None
    assert first.url == "127.0.0.1:9101"
    # A re-registration bumps the generation so the prior advertisement and its stale
    # relay evidence are superseded together.
    second = provider()
    assert second is not None
    assert second.generation > first.generation


def test_advertises_when_enabled_even_without_an_inbound_url() -> None:
    # An outbound-only node still attaches outward, so it advertises (empty url) and is
    # relay attach-eligible; the server derives its attachment identity from the node.
    provider = _endpoint_advertisement_provider(NetworkPlaneConfig(enabled=True))
    ad = provider()
    assert ad is not None
    assert ad.url == ""


def test_no_advertisement_when_the_plane_is_disabled() -> None:
    provider = _endpoint_advertisement_provider(NetworkPlaneConfig(enabled=False))
    assert provider() is None
