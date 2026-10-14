import asyncio
import itertools
import logging
import mimetypes
import os
import threading
from collections.abc import Awaitable, Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

from anyio import CancelScope, create_task_group, to_thread
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import Message, Receive, Scope, Send

from shared.schemas.result import RESULT_MEDIA_TYPE, AnyExecutorResult, result_file_path
from shared.tasks.artifact_binding import (
    FINAL_REVISION,
    REVISION_PATTERN,
    ArtifactBinding,
    ArtifactBindings,
    ArtifactKind,
    is_revision,
    revision_step,
)
from shared.utils.atomic import is_atomic_temp
from shared.utils.manifest import (
    ARTIFACTS_DIR,
    LOGS_DIR,
    RESULTS_NAME,
)
from shared.utils.nofollow import (
    PathRefused,
    is_plain_segment,
    join_relative,
    open_below,
    open_dir,
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
from ...schemas.artifacts import PublishedArtifact, PublishedArtifactList
from ...task.artifact_reads import (
    ArtifactDownload,
    ArtifactNotFound,
    BundleFile,
    BundleStream,
    artifact_bundle_files,
    resolve_download,
)
from ...task.models import TERMINAL_TASK_STATUSES
from ...task.results import ResultUnavailable, ResultUnreadable
from ...task.runtime import TaskRuntime

# Sections the bundle endpoint can include.
_BUNDLE_SECTIONS_CONCRETE = ("results", "artifacts", "logs")
_BUNDLE_SECTIONS_ACCEPTED = (*_BUNDLE_SECTIONS_CONCRETE, "all")
_BUNDLE_SECTIONS_DEFAULT = ("results", "artifacts")
_LOGS_NAME = "logs.jsonl"
# Bundles streaming at once; each holds a thread and a compressor.
_STREAM_SLOTS = 4
_RETRY_AFTER_SEC = 5
# What a download gathers before each send.
_SEND_BYTES = 1 << 20

router = APIRouter(prefix="/results", tags=["Results"])


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


def _header(description: str) -> dict[str, Any]:
    return {"description": description, "schema": {"type": "string"}}


_BINARY = {"schema": {"type": "string", "format": "binary"}}
_RETRY_AFTER = {"Retry-After": _header("Seconds to wait before retrying.")}
_OBJECT_HEADERS = {
    "ETag": _header("The object's content digest."),
    "Repr-Digest": _header("The SHA-256 digest of the whole object."),
    "Last-Modified": _header("When the artifact was published."),
    "Accept-Ranges": _header("Always bytes."),
}
_DOWNLOAD_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "The whole object.",
        "content": {"application/octet-stream": _BINARY},
        "headers": _OBJECT_HEADERS,
    },
    206: {
        "description": "The one range the request named.",
        "content": {"application/octet-stream": _BINARY},
        "headers": {
            **_OBJECT_HEADERS,
            "Content-Range": _header("The slice served, of the whole object."),
        },
    },
    416: {
        "description": "The range lies outside the object.",
        "headers": {"Content-Range": _header("The whole object's size.")},
    },
    503: {
        "description": "The content store is unreachable.",
        "headers": _RETRY_AFTER,
    },
}
_BUNDLE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "A gzip-compressed tar of the bundled sections.",
        "content": {"application/gzip": _BINARY},
    },
    503: {
        "description": "The content store is unreachable, or too many bundles stream.",
        "headers": _RETRY_AFTER,
    },
}


@router.get(
    "/{task_id}/artifacts",
    summary="List a result's artifacts",
    description="List the artifacts a task published, optionally at one revision.",
    response_description="Published artifacts",
)
async def list_result_artifacts(
    task_id: str,
    revision: str | None = Query(
        default=None,
        description="Only this revision: final or step-<n>.",
        json_schema_extra={"pattern": REVISION_PATTERN},
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> PublishedArtifactList:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    if revision is not None:
        _check_revision(revision)
    bindings = await _artifact_bindings(runtime, task_id)
    return PublishedArtifactList(
        task_id=task_id,
        artifacts=[
            _published(binding)
            for (rev, _), binding in sorted(bindings.items(), key=_listing_order)
            if revision is None or rev == revision
        ],
    )


@router.get(
    "/{task_id}/files/{filename:path}",
    summary="Download a result artifact",
    description=(
        "Download a published artifact, a file inside a published directory, or the "
        "task's results.json."
    ),
    response_class=Response,
    responses=_DOWNLOAD_RESPONSES,
)
async def download_result_file(
    task_id: str,
    filename: str,
    revision: str = Query(
        default=FINAL_REVISION,
        description="Revision: final or step-<n>.",
        json_schema_extra={"pattern": REVISION_PATTERN},
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    _check_artifact_path(filename)
    _check_revision(revision)
    if filename == RESULTS_NAME and revision == FINAL_REVISION:
        return Response(
            await _read_result_bytes(runtime, task_id), media_type=RESULT_MEDIA_TYPE
        )
    bindings = await _artifact_bindings(runtime, task_id)
    try:
        download = await asyncio.to_thread(
            resolve_download, runtime.result_reader, bindings, revision, filename
        )
    except ArtifactNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
        ) from exc
    except ResultUnavailable as exc:
        raise _store_unavailable(exc) from exc
    except ResultUnreadable as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read artifact: {exc}",
        ) from exc
    return ArtifactResponse(download)


@router.get(
    "/{task_id}/bundle",
    summary="Download a full result bundle",
    description=(
        "Download a gzip-compressed tar of a settled task's result, its published "
        "artifacts at one revision, and its logs."
    ),
    response_class=Response,
    responses=_BUNDLE_RESPONSES,
)
async def download_result_bundle(
    task_id: str,
    include: list[str] = Query(default_factory=list),
    revision: str = Query(
        default=FINAL_REVISION,
        description="Revision of the artifacts bundled.",
        json_schema_extra={"pattern": REVISION_PATTERN},
    ),
    principal: PrincipalContext = Depends(authenticate_connection),
    runtime: TaskRuntime = Depends(get_runtime),
    results_dir: Path = Depends(get_results_dir),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    await require_permission(
        principal, ResourceKind.RESULT, task_id, ResourceAction.READ, logger
    )
    sections = _resolve_bundle_sections(include)
    _check_revision(revision)

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
    try:
        result = (
            await asyncio.to_thread(runtime.read_result_bytes, task_id)
            if "results" in sections
            else None
        )
    except ResultUnavailable as exc:
        raise _store_unavailable(exc) from exc
    except ResultUnreadable as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare result bundle: {exc}",
        ) from exc
    bindings = (
        await asyncio.to_thread(runtime.artifact_bindings, task_id) or {}
        if "artifacts" in sections
        else {}
    )
    has_logs = "logs" in sections and await asyncio.to_thread(
        _has_dir, base_dir / LOGS_DIR
    )
    if (
        "artifacts" in sections
        and revision != FINAL_REVISION
        and not any(rev == revision for rev, _ in bindings)
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no artifacts published at revision {revision}",
        )
    if result is None and not bindings and not has_logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="result bundle not found"
        )
    reader = runtime.result_reader

    def files() -> Iterator[BundleFile]:
        for section in sections:
            if section == "results" and result is not None:
                yield BundleFile(
                    f"{task_id}/{RESULTS_NAME}", len(result), partial(iter, (result,))
                )
            elif section == "artifacts":
                yield from artifact_bundle_files(
                    reader, bindings, revision, f"{task_id}/{ARTIFACTS_DIR}"
                )
            elif section == "logs":
                yield from _disk_files(base_dir, LOGS_DIR, f"{task_id}/{LOGS_DIR}")

    chunks = BundleStream(files, task_id)
    try:
        release = _STREAMS.take()
    except HTTPException:
        chunks.close()
        raise
    try:
        first = await asyncio.to_thread(next, chunks, b"")
    except BaseException as exc:
        chunks.close()
        release()
        if isinstance(exc, ResultUnavailable):
            raise _store_unavailable(exc) from exc
        if isinstance(exc, ResultUnreadable):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to prepare result bundle: {exc}",
            ) from exc
        raise

    def finish() -> None:
        chunks.close()
        release()

    return _ReleasingStream(
        itertools.chain((first,), chunks),
        finish,
        media_type="application/x-tar",
        headers={
            "Content-Encoding": "gzip",
            "Content-Disposition": f'attachment; filename="{task_id}.tar.gz"',
        },
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


def _file_response(opened: BinaryIO, name: str) -> _OpenFileResponse:
    return _OpenFileResponse(
        opened,
        media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
    )


def _names_several_ranges(value: bytes) -> bool:
    """Whether a ``Range`` header names more than one range. Every non-empty part
    counts, including one ``FileResponse`` would ignore, so such a request is answered
    whole rather than as a single range."""
    _, _, ranges = value.decode("latin-1").partition("=")
    return len(_range_parts(ranges)) > 1


def _range_parts(ranges: str) -> list[str]:
    """The non-empty ranges of a ``Range`` header's value after its unit."""
    parts = (part.strip() for part in ranges.split(","))
    return [part for part in parts if part not in {"", "-"}]


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


async def _read_result_bytes(runtime: TaskRuntime, task_id: str) -> bytes:
    try:
        result = await asyncio.to_thread(runtime.read_result_bytes, task_id)
    except ResultUnavailable as exc:
        raise _store_unavailable(exc) from exc
    except ResultUnreadable as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read result: {exc}",
        ) from exc
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
        )
    return result


def _check_artifact_path(filename: str) -> None:
    if not all(
        is_plain_segment(part) and not is_atomic_temp(part)
        for part in filename.split("/")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid filename"
        )


def _check_revision(revision: str) -> None:
    if not is_revision(revision):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid revision {revision!r}: expected final or step-<n>",
        )


async def _artifact_bindings(runtime: TaskRuntime, task_id: str) -> ArtifactBindings:
    bindings = await asyncio.to_thread(runtime.artifact_bindings, task_id)
    if bindings is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="result not found"
        )
    return bindings


def _listing_order(item: tuple[tuple[str, str], ArtifactBinding]) -> tuple[Any, ...]:
    (revision, name), _ = item
    step = revision_step(revision)
    return (step is None, step or 0, name)


def _published(binding: ArtifactBinding) -> PublishedArtifact:
    sha256 = None
    if binding.kind is ArtifactKind.FILE:
        sha256 = binding.reference.content_digest
    elif binding.archive is not None:
        sha256 = binding.archive.sha256
    return PublishedArtifact(
        name=binding.name,
        revision=binding.revision,
        kind=binding.kind,
        size_bytes=binding.size_bytes,
        files=binding.files,
        sha256=sha256,
        archive_of=binding.of,
        published_at=binding.published_at,
    )


def _store_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"the content store is unreachable: {exc}",
        headers={"Retry-After": str(_RETRY_AFTER_SEC)},
    )


class _StreamSlots:
    """A fixed number of bundles the root streams at once.

    Each holds a thread and a compressor, so a request past the cap is refused with
    ``503`` and a ``Retry-After`` rather than queued behind them.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._held = 0
        self._lock = threading.Lock()

    def take(self) -> Callable[[], None]:
        """A slot's release, or a ``503`` when every slot is held."""
        with self._lock:
            if self._held >= self._capacity:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="too many bundle downloads in flight",
                    headers={"Retry-After": str(_RETRY_AFTER_SEC)},
                )
            self._held += 1
        released = threading.Event()

        def release() -> None:
            if not released.is_set():
                released.set()
                with self._lock:
                    self._held -= 1

        return release


_STREAMS = _StreamSlots(_STREAM_SLOTS)


class _ReleasingStream(StreamingResponse):
    """A streamed response that gives back its stream slot however it ends."""

    def __init__(
        self,
        content: Iterator[bytes],
        release: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(content, **kwargs)
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


class ArtifactResponse(Response):
    """A published object streamed from the store, whole or as one range.

    It carries the object's ``ETag``, its whole bytes' ``Repr-Digest`` and its
    publication time as ``Last-Modified``. A request naming one range, unconditioned
    or whose ``If-Range`` still matches, is answered ``206`` with that slice; several
    ranges are answered whole. A failure before the first byte answers ``503`` while
    the store is unreachable and ``500`` for a missing or corrupt object; a failure
    after it ends the response short.
    """

    def __init__(self, download: ArtifactDownload) -> None:
        super().__init__()
        self._download = download
        headers = {
            "etag": download.etag,
            "repr-digest": download.repr_digest,
            "accept-ranges": "bytes",
            "content-type": download.media_type
            or mimetypes.guess_type(download.name)[0]
            or "application/octet-stream",
        }
        if (modified := download.last_modified) is not None:
            headers["last-modified"] = modified
        self.init_headers(headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # A send after the client is gone does not fail, so the response stops on the
        # disconnect itself rather than reading the rest of the object for no one.
        try:
            async with create_task_group() as group:

                async def until_done(step: Callable[[], Awaitable[None]]) -> None:
                    await step()
                    group.cancel_scope.cancel()

                group.start_soon(
                    until_done, partial(self._respond, scope, receive, send)
                )
                await until_done(partial(_disconnected, receive))
        except BaseExceptionGroup as failed:
            if len(failed.exceptions) == 1:
                raise failed.exceptions[0] from None
            raise

    async def _respond(self, scope: Scope, receive: Receive, send: Send) -> None:
        download = self._download
        size = download.size_bytes
        requested = Headers(scope=scope)
        headers = MutableHeaders(raw=list(self.raw_headers))
        span = None
        if (header := requested.get("range")) is not None and self._range_applies(
            requested.get("if-range")
        ):
            try:
                span = _single_range(header, size)
            except _MalformedRange as exc:
                await PlainTextResponse(str(exc), status_code=400)(scope, receive, send)
                return
            except _UnsatisfiableRange:
                await PlainTextResponse(
                    "", status_code=416, headers={"content-range": f"bytes */{size}"}
                )(scope, receive, send)
                return
        if span is None:
            status_code, chunks = 200, download.whole()
            headers["content-length"] = str(size)
        else:
            start, end = span
            # A range spanning the whole object is read whole, so it is verified.
            status_code = 206
            chunks = (
                download.whole()
                if (start, end) == (0, size)
                else download.part(start, end - start)
            )
            headers["content-range"] = f"bytes {start}-{end - 1}/{size}"
            headers["content-length"] = str(end - start)
        try:
            await self._send(scope, send, status_code, headers, chunks)
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                with CancelScope(shield=True):
                    await to_thread.run_sync(close)

    def _range_applies(self, if_range: str | None) -> bool:
        return if_range is None or if_range in (
            self._download.etag,
            self._download.last_modified,
        )

    async def _send(
        self,
        scope: Scope,
        send: Send,
        status_code: int,
        headers: MutableHeaders,
        chunks: Iterator[bytes],
    ) -> None:
        try:
            body, done = await to_thread.run_sync(_gather, chunks)
        except ResultUnavailable as exc:
            await PlainTextResponse(
                f"the content store is unreachable: {exc}",
                status_code=503,
                headers={"retry-after": str(_RETRY_AFTER_SEC)},
            )(scope, _no_receive, send)
            return
        except ResultUnreadable as exc:
            await PlainTextResponse(f"Failed to read artifact: {exc}", status_code=500)(
                scope, _no_receive, send
            )
            return
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": headers.raw,
            }
        )
        while True:
            await send(
                {"type": "http.response.body", "body": body, "more_body": not done}
            )
            if done:
                return
            body, done = await to_thread.run_sync(_gather, chunks)


def _gather(chunks: Iterator[bytes]) -> tuple[bytes, bool]:
    """About ``_SEND_BYTES`` of ``chunks``, and whether they are done.

    The first gather reaches past an archive view's leading headers into its first
    stored member, so an unreachable store answers before the response starts.
    """
    gathered = bytearray()
    while len(gathered) < _SEND_BYTES:
        if (chunk := next(chunks, None)) is None:
            return bytes(gathered), True
        gathered += chunk
    return bytes(gathered), False


async def _no_receive() -> Message:
    return {"type": "http.disconnect"}


async def _disconnected(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


class _MalformedRange(ValueError):
    pass


class _UnsatisfiableRange(ValueError):
    pass


def _single_range(header: str, size: int) -> tuple[int, int] | None:
    """The one byte range ``[start, end)`` a ``Range`` header names, or None when it
    names several, which is answered whole."""
    units, sep, ranges = header.partition("=")
    if units.strip().lower() != "bytes" or not sep:
        raise _MalformedRange("Only support bytes range")
    parts = _range_parts(ranges)
    if len(parts) > 1:
        return None
    if not parts or "-" not in parts[0]:
        raise _MalformedRange("Range header: range must be requested")
    first, _, last = (piece.strip() for piece in parts[0].partition("-"))
    try:
        if first:
            start = int(first)
            end = min(int(last) + 1, size) if last else size
        else:
            start, end = max(size - int(last), 0), size
    except ValueError as exc:
        raise _MalformedRange("Range header: range must be requested") from exc
    if not 0 <= start < size:
        raise _UnsatisfiableRange()
    if start >= end:
        raise _MalformedRange("Range header: start must be less than end")
    return start, end


def _disk_files(base_dir: Path, section: str, top: str) -> Iterator[BundleFile]:
    """The regular files under a section of a task's results directory, read without
    following a link; a link, an in-flight write and a special file are left out."""
    try:
        with open_dir(base_dir, section) as top_fd:
            for rel_dir, dirs, others, dirfd in walk(top_fd):
                dirs[:] = [entry for entry in dirs if not is_atomic_temp(entry)]
                for entry in sorted(others):
                    if is_atomic_temp(entry):
                        continue
                    opened = open_regular(dirfd, entry)
                    if opened is None:
                        continue
                    with opened as fh:
                        st = os.fstat(fh.fileno())
                        yield BundleFile(
                            f"{top}/{join_relative(rel_dir, entry)}",
                            st.st_size,
                            partial(_read_exactly, fh, st.st_size),
                            int(st.st_mtime),
                        )
    except (FileNotFoundError, PathRefused):
        return


def _read_exactly(fh: BinaryIO, size: int) -> Iterator[bytes]:
    while size > 0:
        chunk = fh.read(min(size, 1 << 20))
        if not chunk:
            raise OSError("a log file shrank while it was bundled")
        size -= len(chunk)
        yield chunk
