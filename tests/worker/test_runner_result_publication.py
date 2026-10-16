"""The runner stores a task's result in the shared store before reporting success."""

import errno
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.content import (
    ContentHydrationError,
    ContentReference,
    ContentStoreError,
    FabricObjectStore,
    SharedFilesystemObjectStore,
)
from shared.harness.adapter import HarnessResult, HarnessResultKind
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.tasks import MergedChildTaskStrict
from shared.tasks.artifact_upload import ARTIFACTS_PENDING
from shared.tasks.task_type import TaskType
from shared.utils.task_output import detach_shared_files
from tests.worker.factories import (
    make_worker_hardware,
    make_worker_task_message,
    no_mediated_op,
)
from worker.content.artifacts import checkpoint_spool_dir, spool_checkpoint
from worker.content.custody import ArtifactCustody, UploadJournal
from worker.content.uploads import ArtifactUploader
from worker.executors.base_executor import Executor
from worker.executors.episode_support import EpisodeStepResult
from worker.runner import Runner


class _Plane:
    def __init__(self, store: FabricObjectStore) -> None:
        self.store = store
        self.tasks: list[str] = []

    def for_task(self, task_id: str, scope: str) -> FabricObjectStore:
        self.tasks.append(task_id)
        return self.store


class _FailingStore(SharedFilesystemObjectStore):
    def write(self, scope: str, data: bytes, *, media_type: str = "") -> Any:
        raise ContentStoreError("store unreachable")


class _Returning(Executor):
    name = "echo"

    def __init__(self, result: BaseExecutorResult) -> None:
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
    return _run_executor(tmp_path, _Returning(result), store, **message)


def _run_with(tmp_path: Path, executor: Executor, **message: Any) -> MagicMock:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    return _run_executor(tmp_path, executor, store, **message)[0]


def _run_executor(
    tmp_path: Path,
    executor: Executor,
    store: FabricObjectStore,
    **message: Any,
) -> tuple[MagicMock, _Plane]:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
    plane = _Plane(store)
    lifecycle.content_plane = plane
    msg = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id="tsk-1",
        content_scope="org-a",
        **message,
    )
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
        owner="w1",
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


def test_a_merged_parent_stores_only_its_own_result(tmp_path: Path) -> None:
    def _stored(out: Path, children: dict[str, BaseExecutorResult]) -> dict[str, Any]:
        store = SharedFilesystemObjectStore(out / "cas")
        result = BaseExecutorResult.model_validate({"value": "parent"})
        result.children = children
        child = MergedChildTaskStrict(
            task_id="tsk-c",
            owner_id="usr-other",
            workflow_id="wfl-other",
            spec=make_worker_task_message(
                {"taskType": "echo"}, task_type=TaskType.ECHO
            ).spec,
        )
        lifecycle, _ = _run(out, result, store, merged_children=[child])
        metadata = _metadata(lifecycle)
        stored = {
            task_id: json.loads(store.hydrate(ContentReference.model_validate(ref)))
            for task_id, ref in {
                "tsk-1": metadata["result_reference"],
                **metadata.get("child_result_references", {}),
            }.items()
        }
        for envelope in stored.values():
            envelope["result"].pop("_artifacts", None)
        return stored

    merged = _stored(
        tmp_path / "merged",
        {"tsk-c": BaseExecutorResult.model_validate({"value": "child"})},
    )
    alone = _stored(tmp_path / "alone", {})

    assert merged["tsk-1"]["result"] == alone["tsk-1"]["result"]
    assert merged["tsk-c"]["result"]["value"] == "child"


class _Producing(_Returning):
    """Leaves artifacts in the task's output, as a trainer does."""

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        model = out_dir / "artifacts" / "final_model"
        model.mkdir(parents=True, exist_ok=True)
        (model / "weights.bin").write_bytes(b"weights")
        return self.result


def _run_producing(
    tmp_path: Path, store: FabricObjectStore, cleanup: bool = False
) -> MagicMock:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
    lifecycle.content_plane = _Plane(store)
    msg = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id="tsk-1",
        content_scope="org-a",
    )
    executor = _Producing(BaseExecutorResult())
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
        cleanup_after_publish=cleanup,
    ).start()
    return lifecycle


@pytest.mark.parametrize("cleanup", [False, True])
def test_success_reports_the_artifacts_it_published(
    tmp_path: Path, cleanup: bool
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")

    metadata = _metadata(_run_producing(tmp_path, store, cleanup))

    (binding,) = metadata["artifacts"]
    assert binding["name"] == "final_model" and binding["kind"] == "tree"
    assert store.hydrate(ContentReference.model_validate(binding["reference"]))
    local = tmp_path / "out" / "tsk-1" / "artifacts" / "final_model"
    assert local.exists() is not cleanup


def test_artifacts_the_store_refuses_fail_the_task_retryably(tmp_path: Path) -> None:
    class _NoFiles(SharedFilesystemObjectStore):
        def write_file(self, scope: str, path: Path, **kwargs: Any) -> Any:
            raise ContentStoreError("store unreachable")

    lifecycle = _run_producing(tmp_path, _NoFiles(tmp_path / "cas"))

    lifecycle.set_succeeded.assert_not_called()
    assert lifecycle.set_failed.call_args.kwargs["retryable"] is True


class _Training(_Returning):
    """Saves a checkpoint mid-run, as a trainer's save callback hands it over."""

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        checkpoint = out_dir / "artifacts" / "checkpoints" / "checkpoint-5"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "model.bin").write_bytes(b"step five")
        spool_checkpoint(checkpoint, checkpoint_spool_dir(out_dir), 5)
        return self.result


def test_a_training_task_reports_its_checkpoints_as_it_runs_and_at_success(
    tmp_path: Path,
) -> None:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
    lifecycle.content_plane = _Plane(SharedFilesystemObjectStore(tmp_path / "cas"))
    msg = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id="tsk-1",
        content_scope="org-a",
    ).model_copy(update={"task_type": TaskType.SFT.value})
    executor = _Training(BaseExecutorResult())
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"sft": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    ).start()

    revisions = {(b["name"], b["revision"]) for b in _metadata(lifecycle)["artifacts"]}
    assert ("checkpoints", "step-5") in revisions
    envelope = ResultEnvelope.model_validate_json(
        lifecycle.content_plane.store.hydrate(
            ContentReference.model_validate(_metadata(lifecycle)["result_reference"])
        )
    )
    assert envelope.result.artifacts_ is not None
    assert envelope.result.artifacts_.revisions == {"checkpoints": "step-5"}
    updates = [
        call.args[1]
        for call in lifecycle.notify_task_update.call_args_list
        if "artifact_published" in call.args[1]
    ]
    assert [u["artifact_published"][0]["revision"] for u in updates] == ["step-5"]
    assert not (tmp_path / "out" / "tsk-1" / ".checkpoint-spool").exists()


def test_a_starting_worker_clears_the_working_dirs_a_crash_left(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept").write_bytes(b"not the worker's")
    left = tmp_path / "out" / "tsk-old"
    for name in (".artifact-inputs", ".ssh-inputs", ".checkpoint-spool"):
        (left / name / "x").mkdir(parents=True)
    (left / ".export-staging").symlink_to(outside)
    (left / "artifacts").mkdir()
    (left / ".working-owner").write_text("w1")

    _run(tmp_path, BaseExecutorResult(), SharedFilesystemObjectStore(tmp_path / "cas"))

    assert sorted(p.name for p in left.iterdir()) == [".working-owner", "artifacts"]
    assert (outside / "kept").read_bytes() == b"not the worker's"


def _run_staged(
    tmp_path: Path, store: FabricObjectStore, executor: Executor, **message: Any
) -> tuple[MagicMock, ArtifactCustody, list[list[UploadJournal]]]:
    custody = ArtifactCustody(tmp_path / "content")
    uploads = ArtifactUploader(
        custody,
        "w1",
        identity=lambda: ("wrk-test", 1),
        upload=MagicMock(),
        resend_success=MagicMock(),
        report=MagicMock(),
    )
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
    lifecycle.content_plane = _Plane(store)
    lifecycle.artifact_uploads = uploads
    owed_at_success: list[list[UploadJournal]] = []
    lifecycle.set_succeeded.side_effect = lambda *_, **__: owed_at_success.append(
        list(custody.journals())
    )
    msg = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id="tsk-1",
        content_scope="org-a",
        dispatch_id="dsp-1",
        **message,
    )
    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    ).start()
    return lifecycle, custody, owed_at_success


def test_a_staged_success_owes_its_upload_before_it_is_reported(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")

    lifecycle, custody, owed = _run_staged(
        tmp_path, store, _Producing(BaseExecutorResult())
    )

    metadata = _metadata(lifecycle)
    assert metadata[ARTIFACTS_PENDING] is True
    (journal,) = owed[0]
    assert journal.task_id == "tsk-1" and journal.success == metadata
    (member,) = journal.tasks["tsk-1"].members
    assert custody.source(journal, member).read_bytes() == b"weights"
    with pytest.raises(ContentHydrationError):
        store.stored_size(member)
    (binding,) = metadata["artifacts"]
    assert store.hydrate(ContentReference.model_validate(binding["reference"]))


class _ProducingMerged(_Returning):
    """Leaves an artifact in its own output and in a merged child's."""

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        for task_dir in (out_dir, out_dir.parent / "tsk-c"):
            (task_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            (task_dir / "artifacts" / "model").write_bytes(task_dir.name.encode())
        return self.result


def test_a_staged_success_holds_each_output_file_itself_in_custody(
    tmp_path: Path,
) -> None:
    result = BaseExecutorResult()
    result.children = {"tsk-c": BaseExecutorResult()}

    _, custody, owed = _run_staged(
        tmp_path,
        SharedFilesystemObjectStore(tmp_path / "cas"),
        _ProducingMerged(result),
        merged_children=[_child()],
    )

    (journal,) = owed[0]
    for task_id in ("tsk-1", "tsk-c"):
        (member,) = journal.tasks[task_id].members
        output = tmp_path / "out" / task_id / "artifacts" / "model"
        assert os.path.samestat(custody.source(journal, member).stat(), output.stat())


def test_a_success_with_nothing_staged_owes_no_upload(tmp_path: Path) -> None:
    lifecycle, custody, owed = _run_staged(
        tmp_path,
        SharedFilesystemObjectStore(tmp_path / "cas"),
        _Returning(BaseExecutorResult()),
    )

    assert ARTIFACTS_PENDING not in _metadata(lifecycle)
    assert owed == [[]]
    assert not any((tmp_path / "content").rglob("journal.json"))


def test_a_task_failing_after_staging_ends_its_custody(tmp_path: Path) -> None:
    class _NoManifests(SharedFilesystemObjectStore):
        def write(self, scope: str, data: bytes, *, media_type: str = "") -> Any:
            raise ContentStoreError("store unreachable")

    lifecycle, custody, _ = _run_staged(
        tmp_path,
        _NoManifests(tmp_path / "cas"),
        _Producing(BaseExecutorResult()),
    )

    lifecycle.set_succeeded.assert_not_called()
    assert lifecycle.set_failed.call_args.kwargs["retryable"] is True
    assert list(custody.journals()) == []
    assert not custody.publication_dir("org-a", "dsp-1").exists()


class _Rewriting(_Returning):
    """Rewrites in place the outputs an earlier attempt left, its own and a merged
    child's."""

    ran = False

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        self.ran = True
        for task_dir in (out_dir, out_dir.parent / "tsk-c"):
            with open(task_dir / "artifacts" / "model", "r+b") as model:
                model.write(b"rewritten")
        return self.result


def _published(tmp_path: Path, task_id: str) -> Path:
    model = tmp_path / "out" / task_id / "artifacts" / "model"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"accepted")
    held = tmp_path / f"custody-{task_id}"
    os.link(model, held)
    return held


def _child() -> MergedChildTaskStrict:
    return MergedChildTaskStrict(
        task_id="tsk-c",
        owner_id="usr-test",
        workflow_id="wfl-test",
        spec=make_worker_task_message(
            {"taskType": "echo"}, task_type=TaskType.ECHO
        ).spec,
    )


def test_a_rerun_writing_its_outputs_never_changes_what_was_published(
    tmp_path: Path,
) -> None:
    held = [_published(tmp_path, task_id) for task_id in ("tsk-1", "tsk-c")]
    result = BaseExecutorResult()
    result.children = {"tsk-c": BaseExecutorResult()}
    executor = _Rewriting(result)

    _run_with(tmp_path, executor, merged_children=[_child()])

    assert executor.ran
    assert [path.read_bytes() for path in held] == [b"accepted", b"accepted"]


def test_an_output_that_cannot_be_detached_fails_before_the_executor_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _published(tmp_path, "tsk-1")
    monkeypatch.setattr(
        "worker.runner.detach_shared_files",
        MagicMock(side_effect=OSError(errno.ENOSPC, "full")),
    )
    executor = _Rewriting(BaseExecutorResult())

    lifecycle = _run_with(tmp_path, executor)

    assert not executor.ran
    lifecycle.set_succeeded.assert_not_called()
    assert lifecycle.set_failed.call_args.kwargs["retryable"] is True


def test_a_cancel_landing_during_detachment_ends_the_run_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = _published(tmp_path, "tsk-1")
    cancelled = False
    ended = Runner._ended_before_execution_locked

    def ended_or_cancelled(runner: Runner, task_id: str) -> str | None:
        return "cancelled" if cancelled else ended(runner, task_id)

    def detaching(top: Path, raise_if_cancelled: Any) -> int:
        nonlocal cancelled
        cancelled = True
        return detach_shared_files(top, raise_if_cancelled)

    monkeypatch.setattr(Runner, "_ended_before_execution_locked", ended_or_cancelled)
    monkeypatch.setattr("worker.runner.detach_shared_files", detaching)
    executor = _Rewriting(BaseExecutorResult())

    lifecycle = _run_with(tmp_path, executor)

    assert not executor.ran
    lifecycle.set_cancelled.assert_called_once()
    lifecycle.set_failed.assert_not_called()
    model = tmp_path / "out" / "tsk-1" / "artifacts" / "model"
    assert os.path.samestat(model.stat(), held.stat())
