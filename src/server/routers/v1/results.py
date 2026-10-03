import asyncio
import gzip
import io
import logging
import mimetypes
import os
import stat
import tarfile
import tempfile
import time
from pathlib import Path
from typing import BinaryIO

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response
from starlette.types import Receive, Scope, Send

from shared.schemas.result import RESULT_MEDIA_TYPE, AnyExecutorResult, result_file_path
from shared.utils.atomic import atomic_write_stream, is_atomic_temp
from shared.utils.manifest import (
    ARTIFACTS_DIR,
    LOGS_DIR,
    RESULTS_NAME,
    prepare_output_dir,
    sync_manifest,
)
from shared.utils.nofollow import (
    PathRefused,
    is_plain_segment,
    join_relative,
    open_below,
    open_dir,
    open_dir_at,
    open_regular,
    walk,
)

from ...app_state import (
    get_logger,
    get_results_dir,
    get_runtime,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...schemas.common import PathResponse
from ...task.models import TERMINAL_TASK_STATUSES
from ...task.results import ResultUnavailable, ResultUnreadable
from ...task.runtime import TaskRuntime

# Sections the bundle endpoint can include.
_BUNDLE_SECTIONS_CONCRETE = ("results", "artifacts", "logs")
_BUNDLE_SECTIONS_ACCEPTED = (*_BUNDLE_SECTIONS_CONCRETE, "all")
_BUNDLE_SECTIONS_DEFAULT = ("results", "artifacts")
_LOGS_NAME = "logs.jsonl"
_DOWNLOAD_CHUNK_BYTES = 1 << 20

router = APIRouter(prefix="/results", tags=["Results"])


def _resolve_artifact_path(filename: str) -> Path:
    segments = filename.split("/")
    if not all(
        is_plain_segment(part) and not is_atomic_temp(part) for part in segments
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
        )
    return Path(ARTIFACTS_DIR, *segments)


@router.get(
    "/{task_id}",
    summary="Get a result",
    description="Get a task result by task ID.",
    response_description="Task result",
)
async def get_result(
    task_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> AnyExecutorResult:
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="task_id is required"
        )
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    try:
        envelope = await asyncio.to_thread(runtime.read_result, task_id)
    except (ResultUnreadable, ResultUnavailable) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read result: {exc}",
        ) from exc
    if envelope is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="result not found"
        )
    return envelope.result


@router.post(
    "/{task_id}/files",
    summary="Upload a result artifact",
    description="Upload an artifact file for a task result.",
    response_description="Upload status",
)
async def upload_result_file(
    task_id: str,
    file: UploadFile = File(...),
    runtime: TaskRuntime = Depends(get_runtime),
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> PathResponse:
    await require_permission(
        principal, ResourceKind.RESULT, None, ResourceAction.WRITE, logger
    )
    base_dir = result_file_path(results_dir, task_id).parent
    relative_path = _resolve_artifact_path(file.filename or "")
    target_path = base_dir / relative_path

    record = runtime.get_record(task_id)
    expected_artifacts = record.task.spec.get_artifacts() if record else []
    try:
        await asyncio.to_thread(_store_artifact, file, base_dir, relative_path)
    except PathRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store artifact: {exc}",
        ) from exc
    await asyncio.to_thread(sync_manifest, base_dir, task_id, expected_artifacts)
    return PathResponse(ok=True, path=str(target_path))


def _store_artifact(file: UploadFile, base_dir: Path, relative_path: Path) -> None:
    prepare_output_dir(base_dir)
    with open_dir(base_dir, *relative_path.parent.parts, create=True) as dir_fd:
        atomic_write_stream(Path(relative_path.name), file.file, dir_fd=dir_fd)


@router.get(
    "/{task_id}/files/{filename:path}",
    summary="Download a result artifact",
    description="Download an artifact file for a task result.",
    response_description="Result file",
    response_class=FileResponse,
)
async def download_result_file(
    task_id: str,
    filename: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    sanitized = Path(filename)
    base_dir = result_file_path(results_dir, task_id).parent
    relative_path = _resolve_artifact_path(filename)

    opened = await asyncio.to_thread(open_below, base_dir, relative_path)
    if opened is None:
        if len(sanitized.parts) != 1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
            )
        if sanitized.name == RESULTS_NAME:
            return Response(
                await _read_result_bytes(runtime, task_id),
                media_type=RESULT_MEDIA_TYPE,
            )
        opened = await asyncio.to_thread(open_below, base_dir, sanitized)
        if opened is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
            )
    return _file_response(opened, sanitized.name)


@router.get(
    "/{task_id}/bundle",
    summary="Download a full result bundle",
    description="Download a tar archive containing the full task result directory.",
    response_description="Result bundle archive",
    response_class=FileResponse,
)
async def download_result_bundle(
    task_id: str,
    background_tasks: BackgroundTasks,
    include: list[str] = Query(default_factory=list),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> FileResponse:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    sections = _resolve_bundle_sections(include)

    record = runtime.get_record(task_id)
    if record is not None and record.status not in TERMINAL_TASK_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"task {task_id} is not in a terminal state "
                f"(status={record.status}); bundle unavailable"
            ),
        )

    base_dir = result_file_path(results_dir, task_id).parent
    has_dir = await asyncio.to_thread(_has_dir, base_dir)
    try:
        result = (
            await asyncio.to_thread(runtime.read_result_bytes, task_id)
            if "results" in sections
            else None
        )
    except (ResultUnreadable, ResultUnavailable) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare result bundle: {exc}",
        ) from exc
    if result is None and not has_dir:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="result bundle not found"
        )

    try:
        bundle_path = await asyncio.to_thread(
            _create_result_bundle_archive, task_id, base_dir, result, sections
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare result bundle: {exc}",
        ) from exc

    background_tasks.add_task(_cleanup_bundle_file, bundle_path)
    return FileResponse(
        bundle_path,
        media_type="application/x-tar",
        filename=f"{task_id}.tar.gz",
        headers={"Content-Encoding": "gzip"},
    )


@router.get(
    "/{task_id}/logs",
    summary="Download archived task logs",
    description="Download archived logs.jsonl for a task result.",
    response_description="Task log file",
    response_class=FileResponse,
)
async def download_task_logs(
    task_id: str,
    principal: PrincipalContext = Depends(authenticate_connection),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    base_dir = result_file_path(results_dir, task_id).parent
    opened = await asyncio.to_thread(open_below, base_dir, Path(LOGS_DIR) / _LOGS_NAME)
    if opened is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="logs not found"
        )
    return _file_response(opened, _LOGS_NAME)


def _file_response(opened: BinaryIO, name: str) -> "_OpenFileResponse":
    return _OpenFileResponse(
        opened,
        media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
    )


class _OpenFileResponse(FileResponse):
    """A file response served from an already-opened file rather than a path, with
    ``FileResponse``'s single ranges and validators.

    The body is read through ``/dev/fd``, which reopens the opened file itself, and
    the opened file is closed once the response ends, sent or aborted. A request
    naming several ranges is answered whole: ``FileResponse`` never finishes sending
    several ranges of a file truncated under it.
    """

    def __init__(self, opened: BinaryIO, media_type: str) -> None:
        self._opened = opened
        super().__init__(
            f"/dev/fd/{opened.fileno()}",
            media_type=media_type,
            stat_result=os.fstat(opened.fileno()),
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        extensions = {
            key: value
            for key, value in scope.get("extensions", {}).items()
            if key != "http.response.pathsend"
        }
        headers = [
            (name, value)
            for name, value in scope.get("headers", [])
            if name.lower() != b"range" or not _names_several_ranges(value)
        ]
        try:
            await super().__call__(
                {**scope, "extensions": extensions, "headers": headers}, receive, send
            )
        finally:
            self._opened.close()


def _names_several_ranges(value: bytes) -> bool:
    """Whether a ``Range`` header names more than one range. Every non-empty part
    counts, including one ``FileResponse`` would ignore, so such a request is answered
    whole rather than as a single range."""
    _, _, ranges = value.decode("latin-1").partition("=")
    return sum(1 for part in ranges.split(",") if part.strip() not in {"", "-"}) > 1


def _has_dir(base_dir: Path) -> bool:
    try:
        with open_dir(base_dir):
            return True
    except (FileNotFoundError, PathRefused):
        return False


def _resolve_bundle_sections(include: list[str]) -> tuple[str, ...]:
    if not include:
        return _BUNDLE_SECTIONS_DEFAULT
    invalid = sorted({v for v in include if v not in _BUNDLE_SECTIONS_ACCEPTED})
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown include values: {invalid}. "
                f"Accepted: {list(_BUNDLE_SECTIONS_ACCEPTED)}"
            ),
        )
    requested = set(include)
    if "all" in requested:
        return _BUNDLE_SECTIONS_CONCRETE
    ordered = tuple(s for s in _BUNDLE_SECTIONS_CONCRETE if s in requested)
    return ordered or _BUNDLE_SECTIONS_DEFAULT


def _create_result_bundle_archive(
    task_id: str,
    base_dir: Path,
    result: bytes | None,
    sections: tuple[str, ...] = _BUNDLE_SECTIONS_DEFAULT,
) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"flowmesh-result-{task_id}-",
        suffix=".tar.gz",
        delete=False,
    ) as tmp:
        bundle_path = Path(tmp.name)

    try:
        with (
            gzip.open(bundle_path, mode="wb") as fileobj,
            tarfile.open(fileobj=fileobj, mode="w") as archive,
        ):
            for section in sections:
                if section == "results":
                    if result is not None:
                        info = tarfile.TarInfo(f"{task_id}/{RESULTS_NAME}")
                        info.size = len(result)
                        info.mode = 0o644
                        info.mtime = int(time.time())
                        archive.addfile(info, io.BytesIO(result))
                    continue
                name = _bundle_section_name(section)
                if name is not None:
                    _add_section(archive, base_dir, name, f"{task_id}/{name}")
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise

    return bundle_path


def _add_section(
    archive: tarfile.TarFile, base_dir: Path, name: str, arcname: str
) -> None:
    """Add the section ``name`` of a task's results and everything under it without
    following a link.

    The walk holds each directory open and reads every name relative to it, so a
    link, or a directory swapped for one, is archived as the link. A task directory
    that is a link has no sections. In-flight atomic writes and names removed or
    replaced during the walk are left out.
    """
    try:
        with open_dir(base_dir) as base_fd:
            if not _add_entry(archive, base_fd, name, arcname):
                return
            with open_dir_at(base_fd, name) as top_fd:
                for rel_dir, dirs, others, dirfd in walk(top_fd):
                    dirs[:] = [entry for entry in dirs if not is_atomic_temp(entry)]
                    for entry in sorted([*dirs, *others]):
                        if not is_atomic_temp(entry):
                            _add_entry(
                                archive,
                                dirfd,
                                entry,
                                f"{arcname}/{join_relative(rel_dir, entry)}",
                            )
    except (FileNotFoundError, PathRefused):
        return


def _add_entry(archive: tarfile.TarFile, dirfd: int, name: str, arcname: str) -> bool:
    """Archive ``name`` in ``dirfd`` without following it; return whether it is a
    directory to walk. A special file, and a name removed or replaced while it is
    read, is left out."""
    try:
        st = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
        if stat.S_ISREG(st.st_mode):
            if (opened := open_regular(dirfd, name)) is not None:
                with opened as fh:
                    info = archive.gettarinfo(arcname=arcname, fileobj=fh)
                    if info is not None and info.isreg():
                        archive.addfile(info, fh)
                    elif info is not None and info.islnk():
                        archive.addfile(info)
            return False
        info = tarfile.TarInfo(arcname)
        if stat.S_ISDIR(st.st_mode):
            info.type = tarfile.DIRTYPE
        elif stat.S_ISLNK(st.st_mode):
            info.type = tarfile.SYMTYPE
            info.linkname = os.readlink(name, dir_fd=dirfd)
        else:
            return False
    except FileNotFoundError:
        return False
    info.mode = stat.S_IMODE(st.st_mode)
    info.uid, info.gid, info.mtime = st.st_uid, st.st_gid, int(st.st_mtime)
    archive.addfile(info)
    return info.isdir()


def _bundle_section_name(section: str) -> str | None:
    if section == "artifacts":
        return ARTIFACTS_DIR
    if section == "logs":
        return LOGS_DIR
    return None


async def _read_result_bytes(runtime: TaskRuntime, task_id: str) -> bytes:
    try:
        result = await asyncio.to_thread(runtime.read_result_bytes, task_id)
    except (ResultUnreadable, ResultUnavailable) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read result: {exc}",
        ) from exc
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
        )
    return result


def _cleanup_bundle_file(path: Path) -> None:
    path.unlink(missing_ok=True)
