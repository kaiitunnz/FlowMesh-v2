"""EventMonitor adopts a serve task on its endpoint report and drains it on terminal.

A public serve or dev_model task is adopted as a standing resident allocation once it
reports an endpoint; an internal resident backing task and a non-serve task are never
adopted; a serve task's terminal drains its binding.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from server.services.monitoring import EventMonitor
from shared.tasks import TaskType


class _GatedServe:
    def __init__(self) -> None:
        self.adopted: list[str] = []
        self.drained: list[str] = []

    def adopt(self, task_id: str) -> None:
        self.adopted.append(task_id)

    def drain(self, task_id: str) -> None:
        self.drained.append(task_id)


def _monitor(runtime: MagicMock, gated_serve: _GatedServe | None) -> EventMonitor:
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("test.monitoring.serve_adopt"),
        runtime=runtime,
        dispatcher=MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=MagicMock(),
        gated_serve=gated_serve,  # type: ignore[arg-type]
    )


def _record(task_type: TaskType, *, resident: bool = False, port: int | None = 8123):
    serve = {"model": "m", "_host": "127.0.0.1", "_port": port} if port else {}
    return SimpleNamespace(
        task_type=task_type, resident=resident, latest_update={"serve": serve}
    )


def test_serve_and_dev_model_endpoints_are_adopted() -> None:
    for task_type in (TaskType.SERVE, TaskType.DEV_MODEL):
        runtime = MagicMock()
        runtime.get_record.return_value = _record(task_type)
        gated = _GatedServe()
        _monitor(runtime, gated)._maybe_adopt_serve("tsk-1")
        assert gated.adopted == ["tsk-1"]


def test_internal_resident_backing_task_is_not_adopted() -> None:
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.SERVE, resident=True)
    gated = _GatedServe()
    _monitor(runtime, gated)._maybe_adopt_serve("tsk-1")
    assert gated.adopted == []


def test_non_serve_task_is_not_adopted() -> None:
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.AGENT)
    gated = _GatedServe()
    _monitor(runtime, gated)._maybe_adopt_serve("tsk-1")
    assert gated.adopted == []


def test_serve_task_without_an_endpoint_yet_is_not_adopted() -> None:
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.SERVE, port=None)
    gated = _GatedServe()
    _monitor(runtime, gated)._maybe_adopt_serve("tsk-1")
    assert gated.adopted == []


def test_serve_task_terminal_drains_the_binding() -> None:
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.DEV_MODEL)
    gated = _GatedServe()
    _monitor(runtime, gated)._maybe_drain_serve("tsk-1")
    assert gated.drained == ["tsk-1"]


def test_non_serve_terminal_does_not_drain() -> None:
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.AGENT)
    gated = _GatedServe()
    _monitor(runtime, gated)._maybe_drain_serve("tsk-1")
    assert gated.drained == []


def test_adopt_and_drain_are_noops_without_a_gated_serve() -> None:
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.SERVE)
    monitor = _monitor(runtime, None)
    monitor._maybe_adopt_serve("tsk-1")
    monitor._maybe_drain_serve("tsk-1")  # no raise
