"""A network-plane node advertises its outbound relay attachment so the resolver can
carry the universal control_relay base between it and any peer."""

from server.config import NetworkPlaneConfig
from server.supervisor.supervisor import _endpoint_advertisement_provider


def test_advertises_a_relay_attachment_when_enabled() -> None:
    provider = _endpoint_advertisement_provider(
        NetworkPlaneConfig(enabled=True, endpoint_url="127.0.0.1:9101")
    )
    first = provider()
    assert first is not None
    assert first.relay_attachment_id == "127.0.0.1:9101"
    # The attachment generation tracks the endpoint generation, so a re-registration
    # supersedes the prior advertisement and its stale relay evidence together.
    assert first.relay_attachment_generation == first.generation
    second = provider()
    assert second is not None
    assert second.relay_attachment_generation > first.relay_attachment_generation


def test_no_advertisement_when_the_plane_is_disabled() -> None:
    provider = _endpoint_advertisement_provider(NetworkPlaneConfig(enabled=False))
    assert provider() is None
