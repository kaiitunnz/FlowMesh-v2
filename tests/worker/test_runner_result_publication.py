"""The runner stores a task's result in the shared store before reporting success."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.content import (
    ContentReference,
    ContentStoreError,
    FabricObjectStore,
    SharedFilesystemObjectStore,
)
from shared.harness.adapter import HarnessResult, HarnessResultKind
from shared.schemas.result import BaseExecutorResult
from shared.tasks import MergedChildTaskStrict
from shared.tasks.executor_key import ExecutorKey
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.executors.base_executor import Executor
from worker.executors.episode_support import EpisodeStepResult
from worker.runner import Runner


class _Plane:
    def __init__(self, store: FabricObjectStore) -> None:
        self.store = store
        self.tasks: list[str] = []

    def for_task(self, task_id: str) -> FabricObjectStore:
        self.tasks.append(task_id)
        return self.store


class _FailingStore(SharedFilesystemObjectStore):
    def write(self, scope: str, data: bytes, *, media_type: str = "") -> Any:
        raise ContentStoreError("store unreachable")


class _Returning(Executor):
    name = "echo"

    def __init__(self, result: BaseExecutorResult) -> None:  # noqa: D107
        self.result = result

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        return self.result

    def cancel(self, task_id: str) -> None:
        return None


def _run(
    tmp_path: Path,
    result: BaseExecutorResult,
    store: FabricObjectStore,
    **message: Any,
) -> tuple[MagicMock, _Plane]:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    plane = _Plane(store)
    lifecycle.content_plane = plane
    msg = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id="tsk-1",
        content_scope="org-a",
        **message,
    )
    executor = _Returning(result)
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={ExecutorKey.ECHO: executor, ExecutorKey.DEFAULT: executor},
        default_executor=executor,
        logger=MagicMock(),
    ).start()
    return lifecycle, plane


def _metadata(lifecycle: MagicMock) -> dict[str, Any]:
    lifecycle.set_failed.assert_not_called()
    return lifecycle.set_succeeded.call_args.kwargs["metadata"]


def test_success_reports_the_stored_envelope(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    lifecycle, _ = _run(tmp_path, BaseExecutorResult(), store)

    reference = ContentReference.model_validate(
        _metadata(lifecycle)["result_reference"]
    )
    local = (tmp_path / "out" / "tsk-1" / "results.json").read_bytes()
    assert reference.authorization_scope == "org-a"
    assert reference.media_type == "application/json"
    assert store.hydrate(reference) == local


def test_a_store_failure_fails_the_task_retryably(tmp_path: Path) -> None:
    lifecycle, _ = _run(tmp_path, BaseExecutorResult(), _FailingStore(tmp_path / "cas"))

    lifecycle.set_succeeded.assert_not_called()
    assert lifecycle.set_failed.call_args.kwargs["retryable"] is True


@pytest.mark.parametrize(
    ("kind", "stored"),
    [(HarnessResultKind.YIELD, False), (HarnessResultKind.COMPLETION, True)],
)
def test_only_an_episode_completion_is_stored(
    tmp_path: Path, kind: HarnessResultKind, stored: bool
) -> None:
    step = EpisodeStepResult(harness_result=HarnessResult(kind=kind), value="v")
    lifecycle, _ = _run(tmp_path, step, SharedFilesystemObjectStore(tmp_path / "cas"))

    assert ("result_reference" in _metadata(lifecycle)) is stored
    assert (tmp_path / "out" / "tsk-1" / "results.json").exists()


def test_merged_children_store_under_the_parent_task(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    result = BaseExecutorResult()
    result.children = {
        "tsk-c": BaseExecutorResult(),
        "tsk-unknown": BaseExecutorResult(),
    }
    child = MergedChildTaskStrict(
        task_id="tsk-c",
        owner_id="usr-test",
        workflow_id="wfl-test",
        spec=make_worker_task_message(
            {"taskType": "echo"}, task_type=TaskType.ECHO
        ).spec,
    )
    lifecycle, plane = _run(tmp_path, result, store, merged_children=[child])

    children = _metadata(lifecycle)["child_result_references"]
    assert set(children) == {"tsk-c"}
    child_bytes = store.hydrate(ContentReference.model_validate(children["tsk-c"]))
    assert child_bytes == (tmp_path / "out" / "tsk-c" / "results.json").read_bytes()
    assert set(plane.tasks) == {"tsk-1"}
