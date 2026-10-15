"""The worker lifecycle owns its content plane from startup to shutdown."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest

from worker.content import WorkerContentCache, WorkerContentPlane
from worker.lifecycle import Lifecycle


def test_the_lifecycle_starts_the_plane_and_stops_it_before_unregistering(
    tmp_path,
) -> None:
    events: list[str] = []
    client = mock.Mock()
    client.unregister.side_effect = lambda *_, **__: events.append("unregister")
    plane = mock.Mock()
    plane.start.side_effect = lambda: events.append("start")
    plane.stop.side_effect = lambda *_: events.append("stop")
    lifecycle = Lifecycle(
        cast(Any, client),
        hb_sec=5,
        hb_ttl_sec=15,
        hb_file=tmp_path / "hb",
        cost_per_hour=0.0,
    )

    lifecycle.start_content_plane(cast(WorkerContentPlane, plane))
    assert lifecycle.content_plane is plane
    lifecycle.shutdown(graceful=True)

    assert events == ["start", "stop", "unregister"]


@pytest.mark.parametrize("report", ["set_succeeded", "set_failed", "set_cancelled"])
def test_the_copies_a_task_kept_are_reported_before_the_task_ends(
    tmp_path: Path, report: str
) -> None:
    sent: list[str] = []
    client = mock.Mock()
    client.task_succeeded.side_effect = lambda *_, **__: sent.append("task")
    client.task_failed.side_effect = lambda *_, **__: sent.append("task")
    client.task_cancelled.side_effect = lambda *_, **__: sent.append("task")
    cache = WorkerContentCache(
        tmp_path / "content",
        announce=lambda holding: sent.append(f"holding {holding.seq}"),
    )
    lifecycle = Lifecycle(
        cast(Any, client),
        hb_sec=5,
        hb_ttl_sec=15,
        hb_file=tmp_path / "hb",
        cost_per_hour=0.0,
    )
    lifecycle.content_plane = WorkerContentPlane(
        cast(Any, SimpleNamespace(store=cache)), cast(Any, mock.Mock())
    )
    try:
        cache.write("local", b"artifact")
        getattr(lifecycle, report)("tsk-1")
    finally:
        cache.stop_reporting()

    assert sent == ["holding 1", "task"]
