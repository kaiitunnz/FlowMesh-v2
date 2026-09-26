"""The runner hydrates a task's referenced inputs before anything reads them."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from shared.content import (
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
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
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.executors.base_executor import Executor
from worker.runner import Runner


class _Plane:
    def __init__(self, store: FabricObjectStore) -> None:
        self.store = store
        self.error: Exception | None = None

    def for_task(self, task_id: str) -> FabricObjectStore:
        return self.store

    def hydrate(self, task_id: str, reference: ContentReference) -> bytes:
        if self.error is not None:
            raise self.error
        return self.store.hydrate(reference)


class _Recording(Executor):
    name = "echo"

    def __init__(self) -> None:  # noqa: D107
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


def _stored(plane: _Plane, task_id: str, result: Any) -> ResultBinding:
    envelope = ResultEnvelope.model_validate({"task_id": task_id, "result": result})
    reference = plane.store.write(
        "org-a",
        envelope.model_dump_json(indent=2).encode(),
        media_type=RESULT_MEDIA_TYPE,
    )
    return ResultBinding(task_id=task_id, reference=reference)


def _run(tmp_path: Path, plane: _Plane, spec: dict[str, Any], **message: Any) -> Any:
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
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    ).start()
    return lifecycle, executor


def test_the_executor_sees_the_hydrated_upstream_results(tmp_path: Path) -> None:
    plane = _Plane(SharedFilesystemObjectStore(tmp_path / "cas"))
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
    plane = _Plane(SharedFilesystemObjectStore(tmp_path / "cas"))
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


def test_a_preparation_resolves_its_hydrated_upstream(tmp_path: Path) -> None:
    plane = _Plane(SharedFilesystemObjectStore(tmp_path / "cas"))
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
