"""EventMonitor adopts a serve task on its endpoint report and drains it on terminal.

A public serve or dev_model task is adopted as a standing resident allocation once it
reports an endpoint; an internal resident backing task and a non-serve task are never
adopted; a serve task's terminal drains its binding.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from server.serve.ingress import ServeAccessMode
from server.services.monitoring import EventMonitor
from shared.tasks import TaskType


class _GatedServe:
    def __init__(self) -> None:
        self.adopted: list[tuple[str, ServeAccessMode]] = []
        self.drained: list[str] = []

    def adopt(self, task_id: str, access_mode: ServeAccessMode) -> None:
        self.adopted.append((task_id, access_mode))

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


def _record(
    task_type: TaskType,
    *,
    resident: bool = False,
    port: int | None = 8123,
    access_mode: str | None = None,
):
    serve = {"model": "m", "_host": "127.0.0.1", "_port": port} if port else {}
    return SimpleNamespace(
        task_type=task_type,
        resident=resident,
        latest_update={"serve": serve},
        task=SimpleNamespace(spec=SimpleNamespace(accessMode=access_mode)),
    )


def test_serve_and_dev_model_endpoints_are_adopted() -> None:
    for task_type in (TaskType.SERVE, TaskType.DEV_MODEL):
        runtime = MagicMock()
        runtime.get_record.return_value = _record(task_type)
        gated = _GatedServe()
        _monitor(runtime, gated)._maybe_adopt_serve("tsk-1")
        assert gated.adopted == [("tsk-1", ServeAccessMode.PROXY)]


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


def test_the_tasks_pinned_access_mode_reaches_adoption() -> None:
    # The binding must pin the mode the user declared, so a forward task is never
    # adopted as a proxy one and silently exposed over the root instead.
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.SERVE, access_mode="forward")
    gated = _GatedServe()
    _monitor(runtime, gated)._maybe_adopt_serve("tsk-1")
    assert gated.adopted == [("tsk-1", ServeAccessMode.FORWARD)]


class _Ingresses:
    def __init__(self, ingress: object | None) -> None:
        self._ingress = ingress

    def live(self, _mode: ServeAccessMode) -> object | None:
        return self._ingress


class _GatedWithIngresses(_GatedServe):
    def __init__(self, ingress: object | None) -> None:
        super().__init__()
        self.ingresses = _Ingresses(ingress)


def _advertise(access_mode: str | None, ingress: object | None) -> dict:
    runtime = MagicMock()
    runtime.get_record.return_value = _record(TaskType.SERVE, access_mode=access_mode)
    monitor = _monitor(runtime, _GatedWithIngresses(ingress))
    monitor._server_base_url = "http://root.example:8000"
    return monitor._handle_serve_task_update(
        "tsk-1", "wrk-1", {"serve": {"model": "m", "_host": "h", "_port": 8123}}
    )["serve"]


def test_a_proxy_task_is_advertised_at_the_server_base_url() -> None:
    ingress = SimpleNamespace(public_url="")
    assert (
        _advertise("proxy", ingress)["url"]
        == "http://root.example:8000/api/v1/serve/tasks/tsk-1"
    )


def test_a_forward_task_is_advertised_at_its_registered_ingress() -> None:
    # The base comes from what the ingress registered, never the root's own url.
    ingress = SimpleNamespace(public_url="http://ingress.example:8100")
    assert (
        _advertise("forward", ingress)["url"]
        == "http://ingress.example:8100/api/v1/serve/tasks/tsk-1"
    )


def test_a_task_whose_ingress_is_unregistered_is_advertised_with_no_url() -> None:
    # Publishing the root url for a forward-pinned task would hand out an address that
    # can only fail closed, so none is published at all.
    assert "url" not in _advertise("forward", None)
