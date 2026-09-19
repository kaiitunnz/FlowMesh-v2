"""``_current_spans_path`` as a ``ContextVar``: off-lane threads resolve no path.

``MediatedEgressSidecar`` and ``ResidentLaneHost`` run their own spans on threads
that are not the executor's own pool (no ``contextvars.copy_context()`` propagation),
concurrently with whichever task the main loop picks up next. A plain module global
would append their spans to that next task's ``spans.jsonl``; a ``ContextVar`` excludes
them by construction because a bare ``threading.Thread`` starts with no inherited
context.
"""

import json
import threading
from pathlib import Path

from shared.telemetry.config import TelemetryLevel
from tests.worker.otel_support import fresh_worker_provider, worker_telemetry
from worker.telemetry import otel


def _read_span_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["name"] for line in path.read_text().splitlines() if line}


def test_off_lane_thread_span_does_not_land_in_the_active_tasks_spans_jsonl(
    tmp_path: Path,
) -> None:
    task_a_path = tmp_path / "task-a" / "spans.jsonl"
    entered = threading.Event()
    off_lane_done = threading.Event()

    def off_lane_work() -> None:
        entered.wait(timeout=5)
        with otel.get_tracer().start_as_current_span("flowmesh.egress"):
            pass
        off_lane_done.set()

    with fresh_worker_provider(worker_telemetry(TelemetryLevel.COARSE)):
        thread = threading.Thread(target=off_lane_work)
        thread.start()
        try:
            with otel.workflow_trace_context("wfl-task-a"):
                with otel.task_trace_context("wfl-task-a", task_a_path):
                    with otel.get_tracer().start_as_current_span("task"):
                        entered.set()
                        assert off_lane_done.wait(timeout=5)
        finally:
            thread.join(timeout=5)

    names = _read_span_names(task_a_path)
    assert names == {"task"}
    assert "flowmesh.egress" not in names


def test_off_lane_thread_resolves_no_spans_path_by_default() -> None:
    """A thread with no ``task_trace_context`` active resolves ``None`` directly."""
    resolved: dict[str, object] = {}

    def check() -> None:
        resolved["path"] = otel._resolve_path()

    thread = threading.Thread(target=check)
    thread.start()
    thread.join(timeout=5)

    assert resolved["path"] is None
