import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException, status
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from lumid_hooks import PrincipalContext, ResourceRef

from server.hooks import PERMISSION_CHECKERS
from server.routers.v1 import results as results_router
from server.task.results import ResultBinding, ResultReader
from shared.content import SharedFilesystemObjectStore
from shared.schemas.result import RESULT_MEDIA_TYPE, ResultEnvelope


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

    assert isinstance(response, FileResponse)
    assert Path(response.path) == artifact_path


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

    assert isinstance(response, FileResponse)
    assert Path(response.path) == root_file


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
