"""The lane host's peer reads: the backstop bounds a stall, never a slow transfer, and
what the cache does with a verified read never fails it; a worker that dials no peer
never asks for a read only a peer transport may carry."""

import asyncio
import time
from pathlib import Path
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

    async def hydrate_to_file(
        self,
        reference: ContentReference,
        task_id: str,
        staging: Path,
        peer_only: bool = False,
    ) -> Path:
        deadline = time.monotonic() + self._duration_sec
        while time.monotonic() < deadline:
            await asyncio.sleep(_STALL / 4)
            if self._progressing:
                self.last_progress = time.monotonic()
        landed = staging / "transfer"
        landed.write_bytes(b"body")
        return landed


def _lane(tmp_path, client: _Client) -> ContentLaneHost:
    lane = ContentLaneHost(
        store=WorkerContentCache(tmp_path / "content"),
        push_frame=lambda frame: None,
        request_grant=lambda reference, task_id, peer_only: None,
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
        assert lane.read_peer(reference_for("local", b"body"), "tsk-1") == b"body"
    finally:
        lane.stop()


def test_a_peer_read_that_stops_progressing_fails_typed(tmp_path) -> None:
    lane = _lane(tmp_path, _Client(60.0, progressing=False))
    try:
        started = time.monotonic()
        with pytest.raises(ContentHydrationError, match="stopped making progress"):
            lane.read_peer(reference_for("local", b"body"), "tsk-1")
        assert time.monotonic() - started < _STALL * 10
    finally:
        lane.stop()


def test_a_verified_peer_read_survives_a_copy_the_cache_cannot_take(
    tmp_path, monkeypatch
) -> None:
    lane = _lane(tmp_path, _Client(0.0, progressing=True))

    def refuse(reference: ContentReference, verified: Path) -> None:
        raise OSError("cache disk full")

    monkeypatch.setattr(lane._store, "adopt", refuse)
    try:
        assert lane.read_peer(reference_for("local", b"body"), "tsk-1") == b"body"
        assert list(lane._store.staging_dir("local").iterdir()) == []
    finally:
        lane.stop()


def test_a_worker_without_peer_transports_never_asks_for_a_peer_only_read(
    tmp_path,
) -> None:
    asked: list[bool] = []
    lane = ContentLaneHost(
        store=WorkerContentCache(tmp_path / "content"),
        push_frame=lambda frame: None,
        request_grant=lambda reference, task_id, peer_only: asked.append(peer_only),
        worker_id="wkr-1",
        generation=1,
        transfer_timeout_sec=_STALL,
    )
    lane.start()
    try:
        with pytest.raises(ContentHydrationError, match="no peer transport"):
            lane.read_peer(reference_for("local", b"body"), "tsk-1", peer_only=True)
    finally:
        lane.stop()
    assert asked == []
