"""A worker shutting down finishes the boundaries it holds for suspended agents: it
unregisters only once control has committed each one's outcome, or at its deadline."""

import threading
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from shared.grpc.supervisor.v1 import supervisor_pb2
from shared.tools.search.schema import SEARCH_INTERFACE
from tests.worker.factories import no_mediated_op
from tests.worker.test_runner_mediated_dispatch import _permit
from tests.worker.test_runner_shutdown import _Echo, _runner
from tests.worker.test_supervisor_client_dispatch_id import _client
from worker import lifecycle as lifecycle_module
from worker import runner as runner_module
from worker import supervisor_client as supervisor_module
from worker.lifecycle import Lifecycle
from worker.main import run_until_exit
from worker.resident.lane_host import ResidentLaneHost
from worker.runner import Runner

_HELD = ("tsk-agent", "call-1")
_RESIDENT_HELD = ("tsk-leaf", "resident-model/tsk-leaf")


def _mediated_op(kind: str) -> supervisor_pb2.DispatchMessage:
    message = supervisor_pb2.DispatchMessage()
    message.mediated_op.kind = kind
    return message


def test_a_stopped_client_relays_mediated_operations_until_shutdown() -> None:
    client = _client()
    client._channel = cast(Any, object())
    client._shutdown.clear()

    def stream(*_: Any, **__: Any) -> Any:
        yield _mediated_op("permit")
        client.stop()
        yield _mediated_op("permit")
        yield _mediated_op("reap")
        client._shutdown.set()

    stub = MagicMock()
    stub.StreamTasks.side_effect = stream
    client._stub = cast(Any, stub)

    with patch.object(supervisor_module.grpc, "channel_ready_future"):
        client._run_task_stream()

    relayed = []
    while (op := client.next_mediated_op(0)) is not None:
        relayed.append(op[0])
    assert relayed == ["permit", "permit", "reap"]


def _draining_runner(
    tmp_path: Path,
    reap_after_sec: float | None,
    order: list[str],
    executor_cls: type[_Echo] = _Echo,
    resident: bool = False,
) -> tuple[Runner, MagicMock]:
    """A runner whose one task stops the worker while it holds a boundary, an egress
    request or a resident one; control reaps the boundary ``reap_after_sec`` into the
    shutdown, or never."""
    runner: Runner

    def stop_while_running(_task_id: str) -> None:
        runner.stop()

    runner = _runner(tmp_path, executor_cls(on_run=stop_while_running), "tsk-1")
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    cast(MagicMock, lifecycle.client).worker_id = runner.lifecycle.worker_id
    if resident:
        lifecycle.resident_requests.put(*_RESIDENT_HELD, '{"messages": []}')
        reap = (
            "resident_reap",
            {"task_id": _RESIDENT_HELD[0], "call_correlation": _RESIDENT_HELD[1]},
        )
    else:
        lifecycle.pending_egress_requests.put(*_HELD, cast(Any, object()))
        reap = ("reap", {"agent_task_id": _HELD[0], "call_correlation": _HELD[1]})
    runner.lifecycle = lifecycle
    reap_at: list[float] = []

    def mediated_op(timeout: float) -> tuple[str, dict[str, Any]] | None:
        if reap_after_sec is None or not runner.shutdown_requested:
            no_mediated_op(timeout)
            return None
        if not reap_at:
            reap_at.append(time.monotonic() + reap_after_sec)
        if reap_at[0] > time.monotonic() or "reaped" in order:
            no_mediated_op(timeout)
            return None
        order.append("reaped")
        return reap

    client = cast(MagicMock, lifecycle.client)
    client.iter_interrupts.return_value = []
    client.iter_stops.return_value = []
    client.next_mediated_op.side_effect = mediated_op
    client.create_task_log_emitter.return_value = None
    client.unregister.side_effect = lambda *_, **__: order.append("unregistered")
    return runner, client


@pytest.mark.parametrize("resident", [False, True])
def test_a_drained_worker_unregisters_once_its_boundaries_are_reaped(
    tmp_path: Path, resident: bool
) -> None:
    order: list[str] = []
    runner, _ = _draining_runner(tmp_path, 0.3, order, resident=resident)

    with (
        patch.object(runner_module, "SearchEgress"),
        patch.object(runner_module, "ModelEgress"),
    ):
        run_until_exit(runner, runner.lifecycle, MagicMock())

    assert order == ["reaped", "unregistered"]
    assert runner.lifecycle.held_boundaries() == []


def test_a_boundary_that_cannot_finish_leaves_at_the_deadline(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    runner, client = _draining_runner(tmp_path, None, order)
    started = time.monotonic()

    with patch.object(runner_module, "_BOUNDARY_DRAIN_SEC", 0.5):
        run_until_exit(runner, runner.lifecycle, MagicMock())

    assert order == ["unregistered"]
    assert time.monotonic() - started < 5.0
    client.unregister.assert_called_once()
    assert client.unregister.call_args.args == (True,)


def test_a_crashed_worker_unregisters_without_waiting(tmp_path: Path) -> None:
    order: list[str] = []
    runner, client = _draining_runner(tmp_path, None, order)
    started = time.monotonic()

    with patch.object(runner, "_resolve_output_dir", side_effect=OSError(28, "full")):
        try:
            run_until_exit(runner, runner.lifecycle, MagicMock())
        except OSError:
            pass

    assert order == ["unregistered"]
    assert time.monotonic() - started < 5.0
    assert client.unregister.call_args.args == (False,)


def test_a_shutdown_bounds_its_teardown_by_the_stop_budget(tmp_path: Path) -> None:
    order: list[str] = []
    runner, client = _draining_runner(tmp_path, None, order)
    runner._resident_host = cast(Any, MagicMock())
    runner._responses_facade = cast(Any, MagicMock())
    plane = MagicMock()
    runner.lifecycle.start_content_plane(plane)
    started = time.monotonic()

    with (
        patch.object(runner_module, "_STOP_BUDGET_SEC", 2.0),
        patch.object(runner_module, "_BOUNDARY_DRAIN_SEC", 1.5),
    ):
        run_until_exit(runner, runner.lifecycle, MagicMock())

    budget_end = started + 2.0
    (facade_timeout,) = cast(MagicMock, runner._responses_facade).stop.call_args.args
    (resident_timeout,) = cast(MagicMock, runner._resident_host).stop.call_args.args
    (plane_timeout,) = plane.stop.call_args.args
    assert 0.0 <= facade_timeout <= 0.6
    assert 0.0 <= resident_timeout <= 0.6
    assert 0.0 <= plane_timeout <= 0.6
    unregister_timeout = client.unregister.call_args.kwargs["timeout"]
    assert unregister_timeout == lifecycle_module._UNREGISTER_FLOOR_SEC
    assert time.monotonic() < budget_end + 0.5


def test_an_unregister_waits_for_the_event_stream_only_until_its_timeout() -> None:
    client = _client()
    client._event_ready.clear()
    finished = threading.Event()

    def unregister() -> None:
        try:
            client.unregister(True, timeout=0.2)
        except RuntimeError:
            pass
        finished.set()

    threading.Thread(target=unregister, daemon=True).start()

    assert finished.wait(timeout=2.0)


def test_a_permit_after_the_boundary_drain_runs_nothing(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _Echo())
    runner._shut_down()
    sidecar = MagicMock()
    permit = _permit("search/v1")

    with patch.object(runner, "_ensure_mediated_sidecar", return_value=sidecar):
        runner._route_mediated_op("permit", permit.model_dump(mode="json"))

    sidecar.submit_permit.assert_not_called()


def test_a_held_model_turn_does_not_hold_the_boundary_drain(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _Echo())
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    runner.lifecycle = lifecycle
    started = time.monotonic()

    rendezvous = runner._model_turn_rendezvous
    rendezvous.reopen(_HELD[0], "episode-1")

    def stash() -> None:
        lifecycle.pending_egress_requests.put(*_HELD, cast(Any, object()))

    with rendezvous.register(*_HELD, "episode-1", stash):
        runner._finish_held_boundaries(started + 5.0)

    assert time.monotonic() - started < 1.0


class _SlowCleanup(_Echo):
    def cleanup_after_run(self) -> None:
        time.sleep(3.0)


def test_a_slow_executor_cleanup_still_unregisters_inside_the_stop_budget(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    runner, client = _draining_runner(tmp_path, None, order, _SlowCleanup)
    unregistered: list[float] = []
    client.unregister.side_effect = lambda *_, **__: unregistered.append(
        time.monotonic()
    )
    started = time.monotonic()

    with (
        patch.object(runner_module, "_STOP_BUDGET_SEC", 2.0),
        patch.object(runner_module, "_BOUNDARY_DRAIN_SEC", 0.5),
    ):
        run_until_exit(runner, runner.lifecycle, MagicMock())

    assert unregistered and unregistered[0] - started < 2.5


def test_a_resident_frame_after_the_boundary_drain_builds_no_lanes(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _Echo())
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    cast(MagicMock, lifecycle.client).worker_id = "wkr-1"
    runner.lifecycle = lifecycle
    runner._shut_down()

    runner._route_mediated_op(
        "resident_reap", {"task_id": "tsk-leaf", "call_correlation": "c"}
    )

    host = runner._resident_host
    if host is not None:
        host.stop(1.0)
    assert host is None


def _race_the_boundary_drain(runner: Runner, building: threading.Event) -> None:
    """Close the boundaries while a frame is building a lane, then let the build end."""
    assert building.wait(5)
    shutdown = threading.Thread(target=runner._shut_down)
    shutdown.start()
    time.sleep(0.2)
    building.clear()
    shutdown.join(5)


def test_a_resident_host_built_across_the_boundary_drain_is_stopped(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _Echo())
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    cast(MagicMock, lifecycle.client).worker_id = "wkr-1"
    runner.lifecycle = lifecycle
    building = threading.Event()
    real_start = ResidentLaneHost.start

    def slow_start(host: ResidentLaneHost) -> None:
        real_start(host)
        building.set()
        while building.is_set():
            time.sleep(0.01)

    with patch.object(ResidentLaneHost, "start", slow_start):
        frame = threading.Thread(
            target=runner._route_mediated_op,
            args=("resident_reap", {"task_id": "tsk-leaf", "call_correlation": "c"}),
        )
        frame.start()
        _race_the_boundary_drain(runner, building)
        frame.join(5)

    host = runner._resident_host
    alive = host is not None and host._thread.is_alive()
    if alive:
        cast(ResidentLaneHost, host).stop(1.0)
    assert not alive


def test_a_permit_sidecar_built_across_the_boundary_drain_is_stopped(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _Echo())
    building = threading.Event()
    sidecar = MagicMock()

    def slow_build(**_: Any) -> MagicMock:
        building.set()
        while building.is_set():
            time.sleep(0.01)
        return sidecar

    with patch.object(runner_module, "MediatedEgressSidecar", side_effect=slow_build):
        frame = threading.Thread(
            target=runner._route_mediated_op,
            args=("permit", _permit(SEARCH_INTERFACE).model_dump(mode="json")),
        )
        frame.start()
        _race_the_boundary_drain(runner, building)
        frame.join(5)

    assert runner._mediated_sidecar is sidecar
    sidecar.stop.assert_called_once()


def test_a_spent_stop_budget_still_gives_the_unregister_its_floor(
    tmp_path: Path,
) -> None:
    client = _client()
    client._stub = cast(Any, object())
    client._event_ready.clear()
    reconnect = threading.Timer(0.3, client._event_ready.set)
    reconnect.start()
    lifecycle = Lifecycle(client, 5, 15, tmp_path / "hb", 0.0)

    with patch.object(client, "shutdown"):
        lifecycle.shutdown(True, deadline=time.monotonic() - 1.0)

    reconnect.join()
    assert not client._event_queue.empty()
