import contextlib
import io
import logging
import os
import stat
import tarfile
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from lumid_hooks import PrincipalContext, ResourceRef

from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import results as results_router
from server.task.results import ResultReader
from shared.content import (
    ContentReference,
    ContentUnavailable,
    SharedFilesystemObjectStore,
)
from shared.schemas.result import RESULT_MEDIA_TYPE, ResultEnvelope
from shared.tasks.result_binding import ResultBinding
from shared.utils import atomic


class _UnreachableStore(SharedFilesystemObjectStore):
    def fetch(self, reference: ContentReference) -> bytes:
        raise ContentUnavailable("store down")


class _RuntimeStub:
    """Resolves task results from bindings over a local shared store."""

    def __init__(self, root: Path | None = None) -> None:
        self.store = SharedFilesystemObjectStore((root or Path("/nonexistent")) / "cas")
        self.reader = ResultReader(self.store)
        self.bindings: dict[str, ResultBinding] = {}

    def bind(self, task_id: str, result: dict[str, Any]) -> bytes:
        data = (
            ResultEnvelope.model_validate({"task_id": task_id, "result": result})
            .model_dump_json(indent=2)
            .encode("utf-8")
        )
        reference = self.store.write("org", data, media_type=RESULT_MEDIA_TYPE)
        self.bindings[task_id] = ResultBinding(task_id=task_id, reference=reference)
        return data

    def unreachable(self) -> None:
        self.reader = ResultReader(_UnreachableStore(Path("/nonexistent")))

    def corrupt(self, task_id: str) -> None:
        reference = self.bindings[task_id].reference
        assert reference is not None
        self.bindings[task_id] = ResultBinding(
            task_id=task_id,
            reference=reference.model_copy(update={"content_digest": "0" * 64}),
        )

    def read_result(self, task_id: str) -> ResultEnvelope | None:
        binding = self.bindings.get(task_id)
        return self.reader.read(binding) if binding is not None else None

    def read_result_bytes(self, task_id: str) -> bytes | None:
        binding = self.bindings.get(task_id)
        return self.reader.read_bytes(binding) if binding is not None else None


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.results_router")


def _principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="p-1",
        org_id="org",
        external_id="ext",
        principal_type="user",
        scopes=[],
    )


class _DenyAllChecker:
    name = "deny-all"

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="denied")

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return frozenset[str]()


@pytest.fixture
def deny_all_permissions() -> Iterator[None]:
    PERMISSION_CHECKERS.append(_DenyAllChecker())
    try:
        yield
    finally:
        PERMISSION_CHECKERS.clear()


def test_download_result_file_route_uses_path_converter() -> None:
    route = next(
        route
        for route in results_router.router.routes
        if isinstance(route, APIRoute)
        and route.path == "/results/{task_id}/files/{filename:path}"
    )
    route = cast(APIRoute, route)
    assert route.path == "/results/{task_id}/files/{filename:path}"


@pytest.mark.anyio
async def test_download_result_file_resolves_flat_name_under_artifacts(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task-1"
    artifacts_dir = task_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    artifact_path = artifacts_dir / "result.json"
    artifact_path.write_text('{"ok":true}', encoding="utf-8")

    response = await results_router.download_result_file(
        task_id="task-1",
        filename="result.json",
        runtime=cast(Any, _RuntimeStub()),
        results_dir=tmp_path,
    )

    assert await _body(response) == b'{"ok":true}'
    assert response.media_type == "application/json"


@pytest.mark.anyio
async def test_download_result_file_falls_back_to_task_root_for_flat_filename(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task-1"
    task_dir.mkdir(parents=True)
    root_file = task_dir / "result.json"
    root_file.write_text('{"ok":true}', encoding="utf-8")

    response = await results_router.download_result_file(
        task_id="task-1",
        filename="result.json",
        runtime=cast(Any, _RuntimeStub()),
        results_dir=tmp_path,
    )

    assert await _body(response) == b'{"ok":true}'


async def _body(response: Response) -> bytes:
    assert isinstance(response, StreamingResponse)
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        assert isinstance(chunk, bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def test_resolve_artifact_relative_path_scopes_nested_paths_to_artifacts() -> None:
    assert results_router._resolve_artifact_path("result.json") == Path(
        "artifacts/result.json"
    )
    assert results_router._resolve_artifact_path("nested/result.json") == Path(
        "artifacts/nested/result.json"
    )
    assert results_router._resolve_artifact_path("artifacts/result.json") == Path(
        "artifacts/artifacts/result.json"
    )
    assert results_router._resolve_artifact_path(
        "artifacts/nested/result.json"
    ) == Path("artifacts/artifacts/nested/result.json")


def test_resolve_artifact_relative_path_rejects_invalid_paths() -> None:
    with pytest.raises(Exception):
        results_router._resolve_artifact_path("../result.json")


@pytest.mark.parametrize("filename", [".fm-tmp-result.json", ".fm-tmp-dir/out.bin"])
def test_an_artifact_named_as_an_in_flight_write_is_refused(filename: str) -> None:
    with pytest.raises(HTTPException) as exc:
        results_router._resolve_artifact_path(filename)
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.anyio
async def test_upload_result_file_denied_without_permission(
    deny_all_permissions: None, logger: logging.Logger
) -> None:
    with pytest.raises(HTTPException) as exc:
        await results_router.upload_result_file(
            task_id="t-1", principal=_principal(), logger=logger
        )
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.anyio
async def test_upload_result_file_shares_the_task_directories_before_writing(
    tmp_path: Path, logger: logging.Logger
) -> None:
    task_dir = tmp_path / "task-1"
    write = results_router.atomic_write_stream
    modes_at_write: dict[str, int] = {}

    def _write(target: Path, source: Any, dir_fd: int) -> None:
        for directory in (task_dir, task_dir / "artifacts", task_dir / "logs"):
            if directory.is_dir():
                modes_at_write[directory.name] = stat.S_IMODE(directory.stat().st_mode)
        write(target, source, dir_fd=dir_fd)

    with patch.object(results_router, "atomic_write_stream", _write):
        await results_router.upload_result_file(
            task_id="task-1",
            file=UploadFile(file=io.BytesIO(b"x"), filename="out.txt"),
            runtime=cast(Any, SimpleNamespace(get_record=lambda _task_id: None)),
            principal=_principal(),
            results_dir=tmp_path,
            logger=logger,
        )

    assert modes_at_write == {"task-1": 0o777, "artifacts": 0o777, "logs": 0o777}
    assert (task_dir / "artifacts" / "out.txt").read_bytes() == b"x"


@pytest.mark.anyio
async def test_get_result_returns_the_bare_stored_result(
    tmp_path: Path, logger: logging.Logger
) -> None:
    runtime = _RuntimeStub(tmp_path)
    runtime.bind("t-1", {"task_type": "echo", "items": [{"output": "a"}]})

    result = await results_router.get_result(
        task_id=" t-1 ",
        principal=_principal(),
        runtime=cast(Any, runtime),
        logger=logger,
    )

    assert not isinstance(result, ResultEnvelope)
    assert result.model_dump()["items"] == [{"output": "a"}]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("task_id", "setup", "code"),
    [
        ("  ", None, status.HTTP_400_BAD_REQUEST),
        ("t-unbound", None, status.HTTP_404_NOT_FOUND),
        ("t-1", "corrupt", status.HTTP_500_INTERNAL_SERVER_ERROR),
        ("t-1", "unreachable", status.HTTP_500_INTERNAL_SERVER_ERROR),
    ],
)
async def test_get_result_error_contract(
    tmp_path: Path,
    logger: logging.Logger,
    task_id: str,
    setup: str | None,
    code: int,
) -> None:
    runtime = _RuntimeStub(tmp_path)
    runtime.bind("t-1", {"items": []})
    if setup == "corrupt":
        runtime.corrupt("t-1")
    if setup == "unreachable":
        runtime.unreachable()

    with pytest.raises(HTTPException) as exc:
        await results_router.get_result(
            task_id=task_id,
            principal=_principal(),
            runtime=cast(Any, runtime),
            logger=logger,
        )
    assert exc.value.status_code == code


@pytest.mark.anyio
async def test_download_results_json_reads_the_stored_envelope(
    tmp_path: Path, logger: logging.Logger
) -> None:
    runtime = _RuntimeStub(tmp_path)
    stored = runtime.bind("task-1", {"items": []})

    response = await results_router.download_result_file(
        task_id="task-1",
        filename="results.json",
        principal=_principal(),
        runtime=cast(Any, runtime),
        results_dir=tmp_path,
        logger=logger,
    )

    assert response.body == stored
    assert response.media_type == RESULT_MEDIA_TYPE


@pytest.mark.anyio
async def test_upload_result_file_copies_in_chunks_off_the_event_loop(
    tmp_path: Path, logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    copies: list[tuple[int, int]] = []
    copy = atomic.shutil.copyfileobj

    def _recording(source: Any, target: Any, length: int = 0) -> None:
        copies.append((threading.get_ident(), length))
        copy(source, target, length)

    async def _whole_read(*args: Any) -> bytes:
        raise AssertionError("the upload was read whole")

    monkeypatch.setattr(atomic.shutil, "copyfileobj", _recording)
    upload = UploadFile(file=io.BytesIO(b"x" * 10), filename="out.bin")
    monkeypatch.setattr(upload, "read", _whole_read)

    await results_router.upload_result_file(
        task_id="task-1",
        file=upload,
        runtime=cast(Any, SimpleNamespace(get_record=lambda _task_id: None)),
        principal=_principal(),
        results_dir=tmp_path,
        logger=logger,
    )

    assert copies == [(copies[0][0], 1 << 20)]
    assert copies[0][0] != threading.get_ident()
    assert (tmp_path / "task-1" / "artifacts" / "out.bin").read_bytes() == b"x" * 10


@pytest.mark.anyio
async def test_download_result_bundle_builds_off_the_event_loop(
    tmp_path: Path, logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    build = results_router._create_result_bundle_archive

    def _recording(*args: Any, **kwargs: Any) -> Path:
        threads.append(threading.get_ident())
        return build(*args, **kwargs)

    monkeypatch.setattr(results_router, "_create_result_bundle_archive", _recording)
    stub = SimpleNamespace(
        get_record=lambda _task_id: None,
        read_result_bytes=lambda _task_id: b"{}",
    )

    response = await results_router.download_result_bundle(
        task_id="t-1",
        background_tasks=BackgroundTasks(),
        include=[],
        principal=_principal(),
        runtime=cast(Any, stub),
        results_dir=tmp_path,
        logger=logger,
    )

    assert isinstance(response, FileResponse)
    assert threads and threads[0] != threading.get_ident()
    Path(response.path).unlink()


def test_a_bundle_leaves_out_in_flight_writes_and_files_removed_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "nested").mkdir(parents=True)
    (artifacts / "nested" / "kept.bin").write_bytes(b"x")
    in_flight = Path(tempfile.mkstemp(prefix=".fm-tmp-", dir=artifacts)[1])
    assert atomic.is_atomic_temp(in_flight.name)
    fwalk = os.fwalk

    def _with_removed(top: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        for dirpath, dirs, files, dirfd in fwalk(top, *args, **kwargs):
            yield dirpath, dirs, [*files, "gone.bin"], dirfd

    monkeypatch.setattr(results_router.os, "fwalk", _with_removed)

    assert _bundle_members(tmp_path) == [
        ("t-1/artifacts", tarfile.DIRTYPE, ""),
        ("t-1/artifacts/nested", tarfile.DIRTYPE, ""),
        ("t-1/artifacts/nested/kept.bin", tarfile.REGTYPE, ""),
    ]


def test_a_bundle_archives_a_symlinked_section_root_as_the_link(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "passwd").write_text("root:x:0:0")
    task = tmp_path / "task"
    task.mkdir()
    (task / "logs").symlink_to(secret)

    members = _bundle_members(task, ("logs",))

    assert members == [("t-1/logs", tarfile.SYMTYPE, str(secret))]
    assert members == _reference_members(task, ("logs",))


def test_a_bundle_archives_a_symlinked_nested_directory_as_the_link(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "passwd").write_text("root:x:0:0")
    artifacts = tmp_path / "task" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "out").symlink_to(secret)

    members = _bundle_members(tmp_path / "task")

    assert members == [
        ("t-1/artifacts", tarfile.DIRTYPE, ""),
        ("t-1/artifacts/out", tarfile.SYMTYPE, str(secret)),
    ]
    assert members == _reference_members(tmp_path / "task")


def test_a_bundle_reads_a_directory_swapped_for_a_link_from_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "kept.bin").write_bytes(b"secret")
    artifacts = tmp_path / "task" / "artifacts"
    (artifacts / "nested").mkdir(parents=True)
    (artifacts / "nested" / "kept.bin").write_bytes(b"x")
    fwalk = os.fwalk

    def _swapping(top: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        for entry in fwalk(top, *args, **kwargs):
            if entry[0].endswith("nested"):
                (artifacts / "nested").rename(tmp_path / "moved")
                (artifacts / "nested").symlink_to(secret)
            yield entry

    monkeypatch.setattr(results_router.os, "fwalk", _swapping)

    with _open_bundle(tmp_path / "task") as archive:
        member = archive.extractfile("t-1/artifacts/nested/kept.bin")
        assert member is not None and member.read() == b"x"


def test_a_bundle_keeps_links_fifos_empty_directories_and_modes(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "empty").mkdir(parents=True)
    (artifacts / "loop").symlink_to("loop")
    os.mkfifo(artifacts / "pipe")
    script = artifacts / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o750)
    (artifacts / "empty").chmod(0o700)

    members = _bundle_members(tmp_path)

    assert sorted(members) == _reference_members(tmp_path)
    with _open_bundle(tmp_path) as archive:
        modes = {member.name: member.mode for member in archive.getmembers()}
    assert modes["t-1/artifacts/run.sh"] == 0o750
    assert modes["t-1/artifacts/empty"] == 0o700


def _bundle_members(
    base_dir: Path, sections: tuple[str, ...] = ("artifacts",)
) -> list[tuple[str, bytes, str]]:
    with _open_bundle(base_dir, sections) as archive:
        return [
            (member.name, member.type, member.linkname)
            for member in archive.getmembers()
        ]


@contextlib.contextmanager
def _open_bundle(
    base_dir: Path, sections: tuple[str, ...] = ("artifacts",)
) -> Iterator[tarfile.TarFile]:
    bundle = results_router._create_result_bundle_archive(
        "t-1", base_dir, None, sections
    )
    try:
        with tarfile.open(bundle, mode="r:gz") as archive:
            yield archive
    finally:
        bundle.unlink()


def _reference_members(
    base_dir: Path, sections: tuple[str, ...] = ("artifacts",)
) -> list[tuple[str, bytes, str]]:
    """The members ``tarfile`` itself archives for each section, in walk order."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for section in sections:
            archive.add(base_dir / section, arcname=f"t-1/{section}")
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r") as archive:
        members = [(m.name, m.type, m.linkname) for m in archive.getmembers()]
    return sorted(members)
