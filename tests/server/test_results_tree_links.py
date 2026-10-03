"""The server never follows a link below the results root: a link planted in a task's
results is never chmodded, written through, read through, or archived as content."""

import io
import logging
import stat
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from server.routers.v1 import results as results_router
from server.routers.v1 import traces as traces_router
from server.services.log_archiver import TaskLogArchiver
from shared.utils.manifest import prepare_output_dir, scratch_dir, sync_manifest

_LOGGER = logging.getLogger("test.results_tree_links")
_SECRET = b"root:x:0:0\n"


class _Outside:
    """A directory and a file outside every task's results, planted as link targets."""

    def __init__(self, tmp_path: Path) -> None:
        self.dir = tmp_path / "outside"
        self.dir.mkdir()
        self.file = self.dir / "secret.jsonl"
        self.file.write_bytes(_SECRET)
        self.dir.chmod(0o700)
        self.file.chmod(0o600)

    def assert_untouched(self) -> None:
        assert stat.S_IMODE(self.dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(self.file.stat().st_mode) == 0o600
        assert sorted(p.name for p in self.dir.iterdir()) == ["secret.jsonl"]
        assert self.file.read_bytes() == _SECRET


@pytest.fixture
def outside(tmp_path: Path) -> _Outside:
    return _Outside(tmp_path)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    path = tmp_path / "results"
    path.mkdir()
    return path


def _task(results: Path, *dirs: str) -> Path:
    task_dir = results / "tsk-1"
    task_dir.mkdir()
    for name in dirs:
        (task_dir / name).mkdir()
    return task_dir


@pytest.mark.parametrize("linked", ["", "logs", "artifacts"])
def test_preparing_a_task_directory_never_re_modes_a_link(
    results: Path, outside: _Outside, linked: str
) -> None:
    task_dir = results / "tsk-1"
    if linked:
        task_dir.mkdir()
    (task_dir / linked if linked else task_dir).symlink_to(outside.dir)

    with pytest.raises(OSError):
        prepare_output_dir(task_dir)

    outside.assert_untouched()


def test_a_scratch_directory_is_never_a_link(results: Path, outside: _Outside) -> None:
    task_dir = _task(results)
    (task_dir / "scratch").symlink_to(outside.dir)

    with pytest.raises(OSError):
        scratch_dir(task_dir)

    outside.assert_untouched()


def test_a_manifest_reads_and_writes_no_link(results: Path, outside: _Outside) -> None:
    task_dir = _task(results, "logs", "artifacts")
    (task_dir / "artifacts" / "out.bin").symlink_to(outside.file)
    (task_dir / "artifacts" / "nested").symlink_to(outside.dir)
    (task_dir / "leak").symlink_to(outside.dir)
    (task_dir / "manifest.json").symlink_to(outside.file)

    manifest = sync_manifest(task_dir, "tsk-1", ["artifacts/out.bin", "nested/x"])

    entries = {entry["path"]: entry for entry in manifest["entries"]}
    assert entries["artifacts/out.bin"]["status"] == "missing"
    assert entries["artifacts"]["file_count"] == 0
    assert "leak" not in entries
    assert not (task_dir / "manifest.json").is_symlink()
    outside.assert_untouched()


@pytest.mark.parametrize("linked", ["logs", "logs/logs.jsonl"])
def test_the_log_archiver_appends_through_no_link(
    results: Path, outside: _Outside, linked: str
) -> None:
    task_dir = _task(results, "logs", "artifacts")
    if linked == "logs":
        (task_dir / "logs").rmdir()
        (task_dir / "logs").symlink_to(outside.dir)
    else:
        (task_dir / linked).symlink_to(outside.file)
    redis = MagicMock()
    redis.get.return_value = None
    runtime = MagicMock()
    runtime.get_record.return_value = None
    archiver = TaskLogArchiver(redis, runtime, results, _LOGGER)
    archiver._ensure_task("tsk-1", 0.0)

    archiver._flush_task("tsk-1", [("1-0", {"payload": '{"message": "hi"}'})])
    archiver._finalize_manifest("tsk-1")

    outside.assert_untouched()
    assert archiver._archived("tsk-1")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("linked", "filename"),
    [("", "out.txt"), ("artifacts", "out.txt")],
)
async def test_an_artifact_upload_writes_through_no_link(
    results: Path, outside: _Outside, linked: str, filename: str
) -> None:
    if linked == "":
        (results / "tsk-1").symlink_to(outside.dir)
    else:
        (_task(results) / "artifacts").symlink_to(outside.dir)

    with pytest.raises(HTTPException) as exc:
        await results_router.upload_result_file(
            task_id="tsk-1",
            file=UploadFile(file=io.BytesIO(b"x"), filename=filename),
            runtime=cast(Any, SimpleNamespace(get_record=lambda _task_id: None)),
            principal=cast(Any, None),
            results_dir=results,
            logger=_LOGGER,
        )

    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    outside.assert_untouched()


@pytest.mark.anyio
async def test_an_artifact_upload_replaces_a_linked_file(
    results: Path, outside: _Outside
) -> None:
    task_dir = _task(results, "logs", "artifacts")
    (task_dir / "artifacts" / "out.txt").symlink_to(outside.file)

    await results_router.upload_result_file(
        task_id="tsk-1",
        file=UploadFile(file=io.BytesIO(b"x"), filename="out.txt"),
        runtime=cast(Any, SimpleNamespace(get_record=lambda _task_id: None)),
        principal=cast(Any, None),
        results_dir=results,
        logger=_LOGGER,
    )

    written = task_dir / "artifacts" / "out.txt"
    assert not written.is_symlink() and written.read_bytes() == b"x"
    outside.assert_untouched()


def test_an_artifact_store_refuses_a_directory_swapped_for_a_link(
    results: Path, outside: _Outside
) -> None:
    task_dir = _task(results, "logs", "artifacts")
    (task_dir / "artifacts" / "sub").symlink_to(outside.dir)

    with pytest.raises(OSError):
        results_router._store_artifact(
            UploadFile(file=io.BytesIO(b"x"), filename="sub/out.txt"),
            task_dir,
            Path("artifacts/sub/out.txt"),
        )

    outside.assert_untouched()


@pytest.mark.anyio
async def test_a_trace_upload_refuses_a_linked_logs_directory(
    results: Path, outside: _Outside
) -> None:
    (_task(results) / "logs").symlink_to(outside.dir)

    with pytest.raises(HTTPException) as exc:
        await traces_router.upload_task_trace(
            task_id="tsk-1",
            trace_type="spans",
            file=UploadFile(file=io.BytesIO(b"{}\n"), filename="spans.jsonl"),
            results_dir=results,
        )

    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
    outside.assert_untouched()


@pytest.mark.anyio
async def test_a_trace_upload_replaces_a_linked_file(
    results: Path, outside: _Outside
) -> None:
    task_dir = _task(results, "logs")
    (task_dir / "logs" / "spans.jsonl").symlink_to(outside.file)

    await traces_router.upload_task_trace(
        task_id="tsk-1",
        trace_type="spans",
        file=UploadFile(file=io.BytesIO(b"{}\n"), filename="spans.jsonl"),
        results_dir=results,
    )

    written = task_dir / "logs" / "spans.jsonl"
    assert not written.is_symlink() and written.read_bytes() == b"{}\n"
    outside.assert_untouched()


@pytest.mark.parametrize("linked", ["logs", "logs/spans.jsonl"])
def test_trace_reads_follow_no_link(
    results: Path, outside: _Outside, linked: str
) -> None:
    outside.file.chmod(0o644)
    outside.file.write_bytes(b'{"leaked": true}\n')
    outside.file.chmod(0o600)
    task_dir = _task(results, "logs")
    if linked == "logs":
        (task_dir / "logs").rmdir()
        (task_dir / "logs").symlink_to(outside.dir)
        (outside.dir / "spans.jsonl").symlink_to(outside.file)
    else:
        (task_dir / linked).symlink_to(outside.file)

    rows = list(traces_router._WorkflowRows(results, ["tsk-1"], "spans.jsonl"))

    assert rows == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("linked", "filename"),
    [
        ("artifacts/out.jsonl", "out.jsonl"),
        ("artifacts", "secret.jsonl"),
        ("out.jsonl", "out.jsonl"),
    ],
)
async def test_an_artifact_download_reads_through_no_link(
    results: Path, outside: _Outside, linked: str, filename: str
) -> None:
    task_dir = _task(results, "artifacts" if linked != "artifacts" else "logs")
    if linked == "artifacts":
        (task_dir / "artifacts").symlink_to(outside.dir)
    else:
        (task_dir / linked).symlink_to(outside.file)

    with pytest.raises(HTTPException) as exc:
        await results_router.download_result_file(
            task_id="tsk-1",
            filename=filename,
            principal=cast(Any, None),
            runtime=cast(Any, SimpleNamespace()),
            results_dir=results,
            logger=_LOGGER,
        )

    assert exc.value.status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }
    outside.assert_untouched()


@pytest.mark.anyio
@pytest.mark.parametrize("linked", ["", "logs", "logs/logs.jsonl"])
async def test_a_logs_download_reads_through_no_link(
    results: Path, outside: _Outside, linked: str
) -> None:
    if linked == "":
        (results / "tsk-1").symlink_to(outside.dir)
        (outside.dir / "logs").mkdir()
        (outside.dir / "logs" / "logs.jsonl").write_bytes(_SECRET)
    elif linked == "logs":
        (_task(results) / "logs").symlink_to(outside.dir)
        (outside.dir / "logs.jsonl").write_bytes(_SECRET)
    else:
        (_task(results, "logs") / linked).symlink_to(outside.file)

    with pytest.raises(HTTPException) as exc:
        await results_router.download_task_logs(
            task_id="tsk-1",
            principal=cast(Any, None),
            results_dir=results,
            logger=_LOGGER,
        )

    assert exc.value.status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }


@pytest.mark.anyio
async def test_a_bundle_of_a_linked_task_directory_holds_nothing_of_its_target(
    results: Path, outside: _Outside
) -> None:
    (results / "tsk-1").symlink_to(outside.dir)
    (outside.dir / "artifacts").mkdir()
    (outside.dir / "artifacts" / "secret.bin").write_bytes(_SECRET)
    stub = SimpleNamespace(
        get_record=lambda _task_id: None,
        read_result_bytes=lambda _task_id: b"{}",
    )

    response = await results_router.download_result_bundle(
        task_id="tsk-1",
        background_tasks=BackgroundTasks(),
        include=[],
        principal=cast(Any, None),
        runtime=cast(Any, stub),
        results_dir=results,
        logger=_LOGGER,
    )

    assert isinstance(response, FileResponse)
    try:
        with tarfile.open(response.path, mode="r:gz") as archive:
            names = archive.getnames()
    finally:
        Path(response.path).unlink()
    assert names == ["tsk-1/results.json"]
