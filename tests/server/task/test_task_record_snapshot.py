"""Callers outside the runtime read its tasks through detached snapshots."""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.services.port_forward import PortForwardService
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime, fanout
from shared.content import (
    OCTET_STREAM,
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
)
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_store_redrive import _runtime as _paused_runtime
from tests.server.task.test_v2_orchestration import (
    _TS,
    AUTORESEARCH,
    FakeRegistry,
    _planned,
    _register,
    _worker,
    _WorkerRegistryStub,
)


class _GatedStore(FabricObjectStore):
    """A store whose reads wait until a gate opens, like a slow GET."""

    def __init__(self, store: FabricObjectStore) -> None:
        self._store = store
        self.gate = threading.Event()
        self.reading = threading.Event()

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._store.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        self.reading.set()
        self.gate.wait(10)
        return self._store.fetch(reference)


@pytest.mark.anyio
async def test_session_restore_reads_a_snapshot_while_the_redrive_adds_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fanout, "_FANOUT_READ_BACKOFF_SEC", 0.0)
    registry = FakeRegistry()
    runtime, flaky, _scheds, _clock = _paused_runtime(registry)
    _, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    payload = _planned(runtime, planner, ["h1", "h2", "h3"])
    # The planner settles while the store is away, so its fan-out waits for the
    # re-drive a restart starts.
    flaky.error = ContentUnavailable("store down")
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", payload, _TS)

    gated = _GatedStore(flaky._store)
    reader = runtime._results
    reader._store = gated  # type: ignore[attr-defined]
    restored = TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        reader,
        logging.getLogger("commit-then-act"),
        credential_vault=InMemoryCredentialVault(),
    )
    try:
        assert await restored.rehydrate() == 1
        assert gated.reading.wait(5)
        restored._tasks["tsk-ssh-forward"] = restored._tasks[planner].model_copy(
            update={
                "task_id": "tsk-ssh-forward",
                "status": TaskStatus.DISPATCHED,
                "assigned_worker": "wkr-1",
                "latest_update": {"ssh": {"mode": "forward", "port": 2222}},
            }
        )
        size_before = len(restored._tasks)
        registered: list[str] = []

        async def register(task_id: str, *_args: Any) -> None:
            # Binding the port awaits while the store answers the re-drive.
            gated.gate.set()
            deadline = time.monotonic() + 5
            while len(restored._tasks) == size_before and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            registered.append(task_id)

        service = SimpleNamespace(
            _register_task_async=register, _logger=logging.getLogger("pf")
        )
        await PortForwardService.restore_sessions(
            cast(Any, service), restored.task_records()
        )

        assert len(restored._tasks) > size_before
        assert registered == ["tsk-ssh-forward"]
    finally:
        restored._redrive.stop()


@pytest.mark.anyio
async def test_a_snapshot_keeps_its_rows_after_the_runtime_changes_them() -> None:
    registry = FakeRegistry()
    runtime, _flaky, _scheds, _clock = _paused_runtime(registry)
    _, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    (snapshot,) = [r for r in runtime.task_records() if r.task_id == planner]

    await asyncio.sleep(0)
    live = runtime._tasks[planner]
    live.failed_workers.append("wkr-9")
    live.latest_update = {"progress": 1}
    live.status = TaskStatus.DISPATCHED

    assert snapshot.failed_workers == []
    assert snapshot.latest_update is None
    assert snapshot.status is TaskStatus.PENDING
    assert snapshot is not live
