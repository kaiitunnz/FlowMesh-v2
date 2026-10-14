"""A worker running the content cache under a control that does not is refused at
once, so it reads the store rather than waiting out a transfer nobody grants."""

import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from server.content import HydrationDenial
from shared.content import BACKEND_FILESYSTEM, ContentReference, ObjectStoreConfig
from shared.schemas.command import MediatedOpMessage
from shared.schemas.event import WorkerEvent
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import FakeRegistry
from tests.server.task.test_v2_orchestration import _runtime as _real_runtime
from tests.worker.content.test_store_access import _access
from worker.content import (
    ContentAccessRegistry,
    ContentLaneHost,
    WorkerContentCache,
    WorkerContentPlane,
)


def _request(request_id: int) -> WorkerEvent:
    return WorkerEvent(
        type="CONTENT_HYDRATION_REQUEST",
        worker_id="wkr-1",
        payload={
            "task_id": "tsk-1",
            "reference": {
                "authorization_scope": "tenant-a",
                "content_digest": "a" * 64,
                "size_bytes": 1,
            },
            "request_id": request_id,
            "peer_only": False,
        },
    )


def _refusals(registry: MagicMock) -> list[MediatedOpMessage]:
    return [call.args[1] for call in registry.publish_mediated_op.call_args_list]


def test_control_without_a_cache_refuses_each_request_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    monitor = _monitor(_real_runtime(FakeRegistry()))
    registry: Any = monitor._worker_registry

    with caplog.at_level(logging.WARNING):
        monitor._handle_worker_event(_request(7))
        monitor._handle_worker_event(_request(8))

    refusals = _refusals(registry)
    assert [(r.frame_kind, r.payload) for r in refusals] == [
        (
            "content_grant_denied",
            {"request_id": n, "reason": HydrationDenial.CACHE_OFF.value},
        )
        for n in (7, 8)
    ]
    assert sum("runs the content cache" in r.message for r in caplog.records) == 1


def test_a_refused_worker_reads_the_store_without_waiting(tmp_path: Path) -> None:
    monitor = _monitor(_real_runtime(FakeRegistry()))
    registry: Any = monitor._worker_registry
    cfg = ObjectStoreConfig(backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path / "s")
    access = ContentAccessRegistry(cfg, arrival_wait_sec=0.1)
    access.accept(_access("tsk-1"))
    reference = access.store_for("tsk-1", "tenant-a").write("tenant-a", b"input")
    lane: ContentLaneHost

    def request_grant(
        reference: ContentReference, task_id: str, peer_only: bool, request_id: int
    ) -> None:
        monitor._handle_worker_event(_request(request_id))
        for refusal in _refusals(registry)[-1:]:
            lane.route(refusal.frame_kind, refusal.payload)

    lane = ContentLaneHost(
        store=WorkerContentCache(tmp_path / "cache"),
        push_frame=lambda frame: None,
        request_grant=request_grant,
        worker_id="wkr-1",
        generation=1,
        transfer_timeout_sec=10.0,
    )
    lane.start()
    try:
        started = time.monotonic()
        assert WorkerContentPlane(lane, access).hydrate("tsk-1", reference) == b"input"
        assert time.monotonic() - started < 5.0
    finally:
        lane.stop()
