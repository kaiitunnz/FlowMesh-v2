import gc
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from fastapi import HTTPException, status
from fastapi.responses import FileResponse, Response
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
from tests.server.asgi_responses import ClientGone, serve


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


async def _body(response: Response) -> bytes:
    return (await serve(response)).body


@pytest.mark.parametrize("filename", [".fm-tmp-result.json", ".fm-tmp-dir/out.bin"])
def test_an_artifact_named_as_an_in_flight_write_is_refused(filename: str) -> None:
    with pytest.raises(HTTPException) as exc:
        results_router._check_artifact_path(filename)
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


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
        revision="final",
        principal=_principal(),
        runtime=cast(Any, runtime),
        logger=logger,
    )

    assert response.body == stored
    assert response.media_type == RESULT_MEDIA_TYPE


def _conditional(etag: str, last_modified: str) -> list[dict[str, str]]:
    return [
        {},
        {"Range": "bytes=0-9"},
        {"Range": "bytes=10-"},
        {"Range": "bytes=-5"},
        {"Range": "bytes=999999-"},
        {"Range": "bytes=abc"},
        {"If-None-Match": etag},
        {"If-Modified-Since": last_modified},
        {"Range": "bytes=0-9", "If-Range": etag},
        {"Range": "bytes=0-9", "If-Range": '"stale"'},
    ]


@pytest.mark.anyio
async def test_a_logs_download_answers_as_a_file_response_does(tmp_path: Path) -> None:
    logs = tmp_path / "task-1" / "logs"
    logs.mkdir(parents=True)
    (logs / "logs.jsonl").write_bytes(b'{"message": "hi"}\n' * 5000)
    plain = await serve(FileResponse(logs / "logs.jsonl"))

    for request in _conditional(plain.headers["etag"], plain.headers["last-modified"]):
        response = await results_router.download_task_logs(
            task_id="task-1", results_dir=tmp_path
        )
        assert await serve(response, request) == await serve(
            FileResponse(logs / "logs.jsonl"), request
        ), request


@pytest.mark.anyio
async def test_an_aborted_download_closes_its_files_without_a_collection(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "task-1" / "logs"
    logs.mkdir(parents=True)
    (logs / "logs.jsonl").write_bytes(b"x" * (1 << 20))
    open_fds = len(os.listdir("/dev/fd"))
    gc.disable()
    try:
        for _ in range(5):
            response = await results_router.download_task_logs(
                task_id="task-1", results_dir=tmp_path
            )
            with pytest.raises(ClientGone):
                await serve(response, abort_after_chunks=1)
        assert len(os.listdir("/dev/fd")) == open_fds
    finally:
        gc.enable()


@pytest.mark.parametrize(
    "filename", ["a//b.txt", "./x.txt", "dir/", "/abs.txt", "x\0y.txt", "a/../b"]
)
def test_an_artifact_name_with_a_non_plain_segment_is_refused(filename: str) -> None:
    with pytest.raises(HTTPException) as exc:
        results_router._check_artifact_path(filename)
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.anyio
@pytest.mark.parametrize("ranges", ["bytes=0-9,20-29", "bytes=0-1, 5-", "bytes=-5,0-0"])
async def test_a_download_naming_several_ranges_is_answered_whole(
    tmp_path: Path, ranges: str
) -> None:
    logs = tmp_path / "task-1" / "logs"
    logs.mkdir(parents=True)
    (logs / "logs.jsonl").write_bytes(bytes(range(256)))

    response = await results_router.download_task_logs(
        task_id="task-1", results_dir=tmp_path
    )

    assert await serve(response, {"Range": ranges}) == await serve(
        FileResponse(logs / "logs.jsonl")
    )


@pytest.mark.anyio
@pytest.mark.parametrize("ranges", [None, "bytes=0-", "bytes=0-9,100-199"])
async def test_a_download_of_a_file_truncated_under_it_ends(
    tmp_path: Path, ranges: str | None
) -> None:
    logs = tmp_path / "task-1" / "logs"
    logs.mkdir(parents=True)
    (logs / "logs.jsonl").write_bytes(b"x" * (1 << 20))
    response = await results_router.download_task_logs(
        task_id="task-1", results_dir=tmp_path
    )
    os.truncate(logs / "logs.jsonl", 0)

    with anyio.fail_after(5):
        served = await serve(response, {"Range": ranges} if ranges else None)

    assert served.body == b""
