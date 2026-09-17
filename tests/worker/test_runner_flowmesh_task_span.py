"""The runner's ``flowmesh.task`` span: unification gate.

Every task type gets one, entering the propagated context from the dispatch envelope,
without perturbing the shipped span tree the governance analyzer reads. These tests
drive a real ``Runner.start()`` loop (as ``test_runner_traceparent.py`` does) rather
than calling the span helpers directly, so they exercise the exact call site the
runner wraps.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from opentelemetry import trace

from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.telemetry.ids import workflow_to_trace_id_int
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.executors.base_executor import Executor
from worker.executors.mixins.governance import GovernanceMixin
from worker.runner import Runner

_WORKFLOW_ID = "wfl-fbad6be5c4434181a2d394eac830dea1"
_INBOUND_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
_INBOUND_TRACE_ID = int("4bf92f3577b34da6a3ce929d0e0e4736", 16)

_OFF = TelemetryConfig(
    level=TelemetryLevel.OFF,
    traces_enabled=True,
    metrics_enabled=True,
    sample_ratio=1.0,
    otlp_endpoint=None,
)
_COARSE = TelemetryConfig(
    level=TelemetryLevel.COARSE,
    traces_enabled=True,
    metrics_enabled=True,
    sample_ratio=1.0,
    otlp_endpoint=None,
)


class _ShippedLikeExecutor(GovernanceMixin, Executor):
    """Mimics one of the six shipped executors: opens its own ``task`` span."""

    name = "echo"

    def __init__(self) -> None:  # noqa: D107
        # Deliberately skips Executor.__init__ (it requires a WorkerConfig this
        # test has no use for); _task_span sets every attribute it needs itself.
        self._task_id: str | None = None
        self._task_out_dir: Path | None = None
        self._current_batch_id: str | None = None
        self._task_owner_id = ""

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        with self._task_span(
            task.task_id, task.workflow_id, out_dir, owner_id=task.owner_id
        ):
            self._log_event("queuing for execution", data_id=task.task_id)
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        return None

    def stop(self, task_id: str) -> None:
        return None

    def cleanup_after_run(self) -> None:
        return None


class _SpanCapturingExecutor(Executor):
    """Mimics ``agent_episode`` / ``service_leaf``: opens no span of its own."""

    name = "echo"
    captured: dict[str, Any]

    def __init__(self) -> None:  # noqa: D107
        self.captured = {}

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        span = trace.get_current_span()
        context = span.get_span_context()
        self.captured["is_recording"] = span.is_recording()
        self.captured["trace_id"] = context.trace_id
        self.captured["name"] = getattr(span, "name", None)
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        return None


def _spans_path(tmp_path: Path, task_id: str) -> Path:
    return tmp_path / task_id / "logs" / "spans.jsonl"


def _read_span_names(tmp_path: Path, task_id: str) -> set[str]:
    path = _spans_path(tmp_path, task_id)
    if not path.exists():
        return set()
    return {json.loads(line)["name"] for line in path.read_text().splitlines() if line}


def _run(
    tmp_path: Path,
    executor: Executor,
    telemetry: TelemetryConfig,
    *,
    traceparent: str | None = None,
    task_id: str = "tsk-1",
) -> None:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    msg = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id=task_id,
        workflow_id=_WORKFLOW_ID,
        traceparent=traceparent,
    )
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path,
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
        telemetry=telemetry,
    )
    runner.start()


def test_shipped_span_tree_unchanged_when_telemetry_off(tmp_path: Path) -> None:
    """Gate: spans.jsonl is still produced at ``off``, with no ``flowmesh.task``."""
    executor = _ShippedLikeExecutor()

    _run(tmp_path, executor, _OFF, task_id="tsk-off")

    names = _read_span_names(tmp_path, "tsk-off")
    assert names == {"task", "queuing for execution"}


def test_shipped_span_tree_excludes_flowmesh_task_when_telemetry_on(
    tmp_path: Path,
) -> None:
    """Gate: the unification span never lands in spans.jsonl, at any level.

    This is the mechanism behind the byte-identical ``ProfileSummary`` acceptance
    test: ``analyze()`` is a pure function of spans.jsonl's rows, and this proves
    those rows are exactly the shipped set whether or not ``flowmesh.task`` opened
    around the run.
    """
    executor = _ShippedLikeExecutor()

    _run(tmp_path, executor, _COARSE, task_id="tsk-on")

    names = _read_span_names(tmp_path, "tsk-on")
    assert names == {"task", "queuing for execution"}
    assert "flowmesh.task" not in names


def test_every_task_type_gets_a_flowmesh_task_span(tmp_path: Path) -> None:
    """Gate: an executor that opens no span of its own (agent_episode, service_leaf)
    still runs inside ``flowmesh.task`` when telemetry is on."""
    executor = _SpanCapturingExecutor()

    _run(tmp_path, executor, _COARSE, task_id="tsk-bare")

    assert executor.captured["is_recording"] is True
    assert executor.captured["name"] == "flowmesh.task"


def test_no_span_at_all_when_telemetry_off(tmp_path: Path) -> None:
    """Gate: ``off`` opens none of the new spans, including for a bare executor."""
    executor = _SpanCapturingExecutor()

    _run(tmp_path, executor, _OFF, task_id="tsk-bare-off")

    assert executor.captured["is_recording"] is False


def test_no_inbound_traceparent_derives_trace_id_from_workflow(
    tmp_path: Path,
) -> None:
    """Gate: with no inbound traceparent, the task still lands in its workflow's
    trace rather than rooting a random one."""
    executor = _SpanCapturingExecutor()

    _run(tmp_path, executor, _COARSE, traceparent=None, task_id="tsk-root")

    assert executor.captured["trace_id"] == workflow_to_trace_id_int(_WORKFLOW_ID)


def test_inbound_traceparent_is_entered_as_parent_context(tmp_path: Path) -> None:
    """Gate: a dispatched traceparent is entered, not re-derived from workflow_id."""
    executor = _SpanCapturingExecutor()

    _run(
        tmp_path,
        executor,
        _COARSE,
        traceparent=_INBOUND_TRACEPARENT,
        task_id="tsk-inbound",
    )

    assert executor.captured["trace_id"] == _INBOUND_TRACE_ID
