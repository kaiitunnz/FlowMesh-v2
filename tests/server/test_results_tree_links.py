"""The server never follows a link below the results root: a link planted in a task's
results is never chmodded, written through, read through, or archived as content."""

import gzip
import io
import logging
import stat
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, status

from server.routers.v1 import results as results_router
from server.services.log_archiver import TaskLogArchiver
from shared.utils.manifest import prepare_output_dir, scratch_dir
from tests.server.asgi_responses import serve

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

    archiver._flush_task("tsk-1", [("1-0", {"payload": '{"message": "hi"}'})], 0.0)
    archiver._finalize_logs("tsk-1", archiver._states["tsk-1"], 0.0)

    outside.assert_untouched()
    assert archiver.archived("tsk-1")


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
@pytest.mark.parametrize("linked", ["", "logs", "logs/logs.jsonl"])
async def test_a_bundle_holds_no_log_read_through_a_link(
    results: Path, outside: _Outside, linked: str
) -> None:
    if linked == "":
        (results / "tsk-1").symlink_to(outside.dir)
    elif linked == "logs":
        (_task(results) / "logs").symlink_to(outside.dir)
    else:
        (_task(results, "logs") / linked).symlink_to(outside.file)
    stub = SimpleNamespace(
        get_record=lambda _task_id: None,
        read_result_bytes=lambda _task_id: b"{}",
        artifact_bindings=lambda _task_id: {},
        result_reader=None,
    )

    response = await results_router.download_result_bundle(
        task_id="tsk-1",
        include=["results", "logs"],
        revision="final",
        principal=cast(Any, None),
        runtime=cast(Any, stub),
        results_dir=results,
        logger=_LOGGER,
    )

    body = (await serve(response)).body
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(body))) as archive:
        names = archive.getnames()
    assert names == ["tsk-1/results.json"]
    outside.assert_untouched()
