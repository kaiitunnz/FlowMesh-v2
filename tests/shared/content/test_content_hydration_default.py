"""The worker content cache is on wherever the network plane is, unless set."""

import pytest

from shared.content import content_hydration_enabled


@pytest.mark.parametrize(
    ("env", "enabled"),
    [
        ({}, True),
        ({"CONTENT_HYDRATION_ENABLED": ""}, True),
        ({"NETWORK_PLANE_ENABLED": "false"}, False),
        ({"NETWORK_PLANE_ENABLED": "false", "CONTENT_HYDRATION_ENABLED": ""}, False),
        ({"CONTENT_HYDRATION_ENABLED": "false"}, False),
        ({"CONTENT_HYDRATION_ENABLED": "true"}, True),
        ({"NETWORK_PLANE_ENABLED": "true", "CONTENT_HYDRATION_ENABLED": "1"}, True),
    ],
)
def test_the_cache_follows_its_setting_or_the_network_plane(
    env: dict[str, str], enabled: bool
) -> None:
    assert content_hydration_enabled(env) is enabled


def test_the_cache_turned_on_without_the_network_plane_is_refused() -> None:
    with pytest.raises(ValueError, match="requires NETWORK_PLANE_ENABLED"):
        content_hydration_enabled(
            {"NETWORK_PLANE_ENABLED": "false", "CONTENT_HYDRATION_ENABLED": "true"}
        )
