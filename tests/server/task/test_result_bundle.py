"""Result bundle sections."""

import gzip
import io
import logging
import tarfile
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException

from server.routers.v1.results import (
    _BUNDLE_SECTIONS_DEFAULT,
    _resolve_bundle_sections,
    download_result_bundle,
)
from tests.server.asgi_responses import serve
from tests.server.test_artifact_downloads import _publish, _Runtime

_RESULT = b'{"task_id": "tsk-1", "result": {}}'


async def _names(tmp_path: Path, include: list[str], result: bytes | None) -> set[str]:
    runtime = _Runtime(tmp_path)
    _publish(tmp_path, runtime)
    if result is None:
        runtime.results.clear()
    else:
        runtime.results["tsk-1"] = result
    logs = tmp_path / "results" / "tsk-1" / "logs"
    logs.mkdir(parents=True)
    (logs / "logs.jsonl").write_text("line\n", encoding="utf-8")
    response = await download_result_bundle(
        task_id="tsk-1",
        include=include,
        revision="final",
        principal=cast(Any, None),
        runtime=cast(Any, runtime),
        results_dir=tmp_path / "results",
        logger=logging.getLogger("test.result_bundle"),
    )
    body = (await serve(response)).body
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(body))) as archive:
        return set(archive.getnames())


@pytest.mark.anyio
async def test_default_bundle_contains_result_and_artifacts_not_logs(
    tmp_path: Path,
) -> None:
    names = await _names(tmp_path, [], _RESULT)

    assert "tsk-1/results.json" in names
    assert "tsk-1/artifacts/notes.txt" in names
    assert "tsk-1/artifacts/final_model/config.json" in names
    assert not any(name.endswith(".tar.gz") for name in names)
    assert "tsk-1/logs/logs.jsonl" not in names


@pytest.mark.anyio
async def test_result_only_bundle_has_single_file(tmp_path: Path) -> None:
    assert await _names(tmp_path, ["results"], _RESULT) == {"tsk-1/results.json"}


@pytest.mark.anyio
async def test_all_bundle_includes_every_concrete_section(tmp_path: Path) -> None:
    names = await _names(tmp_path, ["all"], _RESULT)

    assert "tsk-1/results.json" in names
    assert "tsk-1/artifacts/notes.txt" in names
    assert "tsk-1/logs/logs.jsonl" in names


@pytest.mark.anyio
async def test_bundle_without_a_result_omits_the_results_entry(tmp_path: Path) -> None:
    names = await _names(tmp_path, ["results", "artifacts"], None)

    assert "tsk-1/results.json" not in names
    assert "tsk-1/artifacts/notes.txt" in names


def test_resolve_bundle_sections_empty_uses_default() -> None:
    assert _resolve_bundle_sections([]) == _BUNDLE_SECTIONS_DEFAULT


def test_resolve_bundle_sections_all_expands_to_concrete_set() -> None:
    assert _resolve_bundle_sections(["all"]) == ("results", "artifacts", "logs")


def test_resolve_bundle_sections_dedupes_and_orders() -> None:
    assert _resolve_bundle_sections(["artifacts", "logs", "results"]) == (
        "results",
        "artifacts",
        "logs",
    )


def test_resolve_bundle_sections_rejects_unknown() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _resolve_bundle_sections(["artifacts", "bogus"])
    assert exc_info.value.status_code == 400
