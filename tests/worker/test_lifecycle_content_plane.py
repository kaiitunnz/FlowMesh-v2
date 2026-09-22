"""The worker lifecycle owns its content plane from startup to shutdown."""

from typing import Any, cast
from unittest import mock

from worker.content import WorkerContentPlane
from worker.lifecycle import Lifecycle


def test_the_lifecycle_starts_the_plane_and_stops_it_before_unregistering(
    tmp_path,
) -> None:
    events: list[str] = []
    client = mock.Mock()
    client.unregister.side_effect = lambda **_: events.append("unregister")
    plane = mock.Mock()
    plane.start.side_effect = lambda: events.append("start")
    plane.stop.side_effect = lambda: events.append("stop")
    lifecycle = Lifecycle(
        cast(Any, client),
        hb_sec=5,
        hb_ttl_sec=15,
        hb_file=tmp_path / "hb",
        cost_per_hour=0.0,
    )

    lifecycle.start_content_plane(cast(WorkerContentPlane, plane))
    assert lifecycle.content_plane is plane
    lifecycle.shutdown()

    assert events == ["start", "stop", "unregister"]
