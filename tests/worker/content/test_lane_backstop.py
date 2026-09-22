"""The lane host's backstop on a peer read: it bounds a stall, never a slow transfer."""

import asyncio
import time
from typing import cast

import pytest

from shared.content import ContentHydrationError, ContentReference, reference_for
from worker.content import ContentHydrationClient, WorkerContentCache
from worker.content.lane_host import ContentLaneHost

_STALL = 0.2


class _Client:
    """A requester that takes a while, reporting progress as it goes or not at all."""

    def __init__(self, duration_sec: float, *, progressing: bool) -> None:
        self._duration_sec = duration_sec
        self._progressing = progressing
        self.last_progress = time.monotonic()

    async def hydrate(self, reference: ContentReference, task_id: str) -> bytes:
        deadline = time.monotonic() + self._duration_sec
        while time.monotonic() < deadline:
            await asyncio.sleep(_STALL / 4)
            if self._progressing:
                self.last_progress = time.monotonic()
        return b"body"


def _lane(tmp_path, client: _Client) -> ContentLaneHost:
    lane = ContentLaneHost(
        store=WorkerContentCache(tmp_path / "content"),
        push_frame=lambda frame: None,
        request_grant=lambda reference, task_id: None,
        worker_id="wkr-1",
        generation=1,
        transfer_timeout_sec=_STALL,
    )
    lane.start()
    lane._client = cast(ContentHydrationClient, client)
    return lane


def test_a_peer_read_that_keeps_progressing_is_not_cut_off(tmp_path) -> None:
    lane = _lane(tmp_path, _Client(_STALL * 6, progressing=True))
    try:
        assert lane.hydrate(reference_for("local", b"body"), "tsk-1") == b"body"
    finally:
        lane.stop()


def test_a_peer_read_that_stops_progressing_fails_typed(tmp_path) -> None:
    lane = _lane(tmp_path, _Client(60.0, progressing=False))
    try:
        started = time.monotonic()
        with pytest.raises(ContentHydrationError, match="stopped making progress"):
            lane.hydrate(reference_for("local", b"body"), "tsk-1")
        assert time.monotonic() - started < _STALL * 10
    finally:
        lane.stop()
