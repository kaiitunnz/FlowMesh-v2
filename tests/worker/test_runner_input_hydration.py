"""The runner hydrates a task's referenced inputs before anything reads them."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from shared.content import (
    ContentUnavailable,
    SharedFilesystemObjectStore,
)
from shared.inference import canonical_contract
from shared.schemas.event import TaskFailureKind
from shared.schemas.result import RESULT_MEDIA_TYPE, BaseExecutorResult, ResultEnvelope
from shared.schemas.result.catalog import InferenceResult
from shared.schemas.result.payloads import InferenceItem
from shared.tasks.result_binding import ResultBinding
from shared.tasks.specs import InferenceSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    FakeContentPlane,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.executors.base_executor import Executor
from worker.runner import Runner


class _Recording(Executor):
    name = "echo"

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        self.seen.append(
            {
                stage: result.model_dump(mode="json")
                for stage, result in (task.spec.upstreamResults or {}).items()
            }
        )
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        return None


def _stored(plane: FakeContentPlane, task_id: str, result: Any) -> ResultBinding:
    envelope = ResultEnvelope.model_validate({"task_id": task_id, "result": result})
    reference = plane.store.write(
        "org-a",
        envelope.model_dump_json(indent=2).encode(),
        media_type=RESULT_MEDIA_TYPE,
    )
    return ResultBinding(task_id=task_id, reference=reference)


def _runner(
    tmp_path: Path, plane: FakeContentPlane, spec: dict[str, Any], **message: Any
) -> tuple[Runner, MagicMock, "_Recording"]:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.content_plane = plane
    msg = make_worker_task_message(
        spec, task_id="tsk-1", content_scope="org-a", **message
    )
    executor = _Recording()
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    )
    return runner, lifecycle, executor


def _run(
    tmp_path: Path, plane: FakeContentPlane, spec: dict[str, Any], **message: Any
) -> Any:
    runner, lifecycle, executor = _runner(tmp_path, plane, spec, **message)
    runner.start()
    return lifecycle, executor


def test_the_executor_sees_the_hydrated_upstream_results(tmp_path: Path) -> None:
    plane = FakeContentPlane(SharedFilesystemObjectStore(tmp_path / "cas"))
    upstream = _stored(plane, "tsk-p", {"items": ["alpha"]})

    lifecycle, executor = _run(
        tmp_path,
        plane,
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        upstream_results={"p": upstream},
    )

    lifecycle.set_failed.assert_not_called()
    assert executor.seen == [{"p": {"ok": True, "items": ["alpha"]}}]


def test_an_unreachable_store_reports_the_inputs_unavailable(tmp_path: Path) -> None:
    plane = FakeContentPlane(SharedFilesystemObjectStore(tmp_path / "cas"))
    upstream = _stored(plane, "tsk-p", {"items": ["alpha"]})
    plane.error = ContentUnavailable("store down")

    lifecycle, executor = _run(
        tmp_path,
        plane,
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        upstream_results={"p": upstream},
    )

    assert executor.seen == []
    kwargs = lifecycle.set_failed.call_args.kwargs
    assert kwargs["retryable"] is True
    assert kwargs["failure_kind"] is TaskFailureKind.INPUT_UNAVAILABLE
    assert kwargs["unavailable_inputs"] == (upstream.reference,)


def test_a_cancel_landing_during_hydration_stops_before_execution(
    tmp_path: Path,
) -> None:
    plane = FakeContentPlane(SharedFilesystemObjectStore(tmp_path / "cas"))
    upstream = _stored(plane, "tsk-p", {"items": ["alpha"]})
    runner, lifecycle, executor = _runner(
        tmp_path,
        plane,
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        upstream_results={"p": upstream},
    )
    plane.on_read = lambda: runner._pending_cancels.add("tsk-1")

    runner.start()

    assert executor.seen == []
    lifecycle.set_cancelled.assert_called_once()
    lifecycle.set_failed.assert_not_called()
    assert "tsk-1" not in runner._pending_cancels


def test_a_preparation_resolves_its_hydrated_upstream(tmp_path: Path) -> None:
    plane = FakeContentPlane(SharedFilesystemObjectStore(tmp_path / "cas"))
    produced = InferenceResult(
        model="up/model",
        items=[InferenceItem(index=0, prompt="p", output="from upstream")],
    )
    upstream = _stored(plane, "tsk-up", produced.model_dump(mode="json"))
    spec = InferenceSpecStrict.model_validate(
        {
            "taskType": "inference",
            "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
            "data": {"type": "list", "expr": "up.items.output"},
        }
    )

    lifecycle, _ = _run(
        tmp_path,
        plane,
        spec.model_dump(by_alias=True),
        upstream_results={"up": upstream},
        declared_contract=canonical_contract(spec),
        input_preparation=True,
    )

    lifecycle.set_failed.assert_not_called()
    metadata = lifecycle.set_succeeded.call_args.kwargs["metadata"]
    assert metadata["input_materialization"]["binding"]["cardinality"] == 1
