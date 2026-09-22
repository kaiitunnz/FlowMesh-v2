"""What a worker tells control it holds, at startup and on its keep-alive cadence."""

import pytest

from worker.content import ContentLaneHost, WorkerContentCache


class _Reports:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, held) -> None:  # noqa: ANN001 - test sink
        self.calls.append(list(held))

    @property
    def reported(self) -> set[tuple[str, str]]:
        return {item for call in self.calls for item in call}


def _lane(tmp_path, reports: _Reports, **kwargs) -> ContentLaneHost:
    return ContentLaneHost(
        store=WorkerContentCache(
            tmp_path / "content", retain_sec=kwargs.pop("retain", 900.0)
        ),
        push_frame=lambda frame: None,
        request_grant=lambda reference, task_id: None,
        worker_id="wkr-1",
        generation=2,
        announce=reports,
        **kwargs,
    )


def test_a_restarted_worker_reports_the_copies_still_on_its_disk(tmp_path) -> None:
    first = WorkerContentCache(tmp_path / "content", retain_sec=900.0)
    reference = first.write("local", b"prepared", media_type="application/json")

    reports = _Reports()
    lane = _lane(tmp_path, reports)
    lane.start()
    try:
        assert lane.report_held() == 1
    finally:
        lane.stop()
    assert reports.reported == {("local", reference.content_digest)}


def test_one_report_carries_every_object_rather_than_one_message_each(
    tmp_path,
) -> None:
    cache = WorkerContentCache(tmp_path / "content", retain_sec=900.0)
    for index in range(5):
        cache.write("local", f"object-{index}".encode())

    reports = _Reports()
    lane = _lane(tmp_path, reports)
    lane.start()
    try:
        assert lane.report_held() == 5
    finally:
        lane.stop()
    assert len(reports.calls) == 1
    assert len(reports.calls[0]) == 5


@pytest.mark.parametrize("ttl", [15.0, 60.0, 300.0, 3600.0])
def test_the_keep_alive_reports_several_times_per_record_lifetime(
    tmp_path, ttl: float
) -> None:
    # A record lapses unless a report refreshes it, so the loop must come round well
    # inside one lifetime however the deployment sets it.
    lane = _lane(tmp_path, _Reports(), holder_report_ttl_sec=ttl)
    assert lane.housekeeping_interval_sec <= ttl / 3
