"""Result resource operations."""

import json
import tarfile
import tempfile
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from ..models.artifacts import PublishedArtifactList
from ..models.result import AnyExecutorResult, ResultEnvelope
from ._base import AsyncResource, SyncResource

type BundleSection = Literal["results", "artifacts", "logs", "all"]

_DEFAULT_INCLUDE: tuple[BundleSection, ...] = ("results", "artifacts")

_RESULT_ADAPTER: TypeAdapter[AnyExecutorResult] = TypeAdapter(AnyExecutorResult)


class Results(SyncResource):
    """Synchronous result operations."""

    def retrieve(self, task_id: str) -> AnyExecutorResult:
        """Retrieve the result for a completed task."""
        data = self._client._request("GET", f"/results/{task_id}")
        return _RESULT_ADAPTER.validate_python(data)

    def list_artifacts(
        self, task_id: str, revision: str | None = None
    ) -> PublishedArtifactList:
        """List the artifacts a task published, at every revision or at one."""
        data = self._client._request(
            "GET", f"/results/{task_id}/artifacts", params=_revision_params(revision)
        )
        return PublishedArtifactList.model_validate(data)

    def get_bundle(
        self,
        task_id: str,
        output_path: Path,
        include: Iterable[BundleSection] | None = None,
        revision: str | None = None,
    ) -> None:
        """Download a tar bundle of the task result, with its artifacts at
        ``revision`` (the final ones by default)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client._download(_bundle_path(task_id, include, revision), output_path)

    def download_file(
        self,
        task_id: str,
        filename: str,
        output_path: Path,
        revision: str | None = None,
    ) -> None:
        """Download a published artifact, or a file inside a published directory,
        at ``revision`` (the final one by default)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client._download(_file_path(task_id, filename, revision), output_path)

    def download_logs(self, task_id: str, output_path: Path) -> None:
        """Download archived logs.jsonl for a task."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client._download(f"/results/{task_id}/logs", output_path)

    def materialize(
        self,
        task_id: str,
        output_dir: Path,
        include: Iterable[BundleSection] | None = None,
        revision: str | None = None,
    ) -> tuple[dict[str, Any], Path, list[Path]]:
        """Fetch the task bundle and extract it under `output_dir/<task_id>/`.

        `include` defaults to `("results", "artifacts")`, and the artifacts are
        those of `revision`, the final ones by default. Returns
        `(payload, json_path, extracted_paths)`."""
        sections = _normalize_include(include)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            prefix=f"flowmesh-bundle-{task_id}-", suffix=".tar", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._client._download(_bundle_path(task_id, sections, revision), tmp_path)
            extracted = _extract_bundle(tmp_path, output_dir)
        finally:
            tmp_path.unlink(missing_ok=True)

        return _finalize_materialize(output_dir, task_id, sections, extracted)

    def download_files(
        self,
        task_id: str,
        file_paths: list[str],
        output_dir: Path,
        revision: str | None = None,
    ) -> Iterable[Path]:
        """Download specific result files into an output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in file_paths:
            out_path = output_dir / Path(file_path).name
            self.download_file(task_id, file_path, out_path, revision)
            yield out_path


class AsyncResults(AsyncResource):
    """Asynchronous result operations."""

    async def retrieve(self, task_id: str) -> AnyExecutorResult:
        """Retrieve the result for a completed task."""
        data = await self._client._request("GET", f"/results/{task_id}")
        return _RESULT_ADAPTER.validate_python(data)

    async def list_artifacts(
        self, task_id: str, revision: str | None = None
    ) -> PublishedArtifactList:
        """List the artifacts a task published. See sync variant."""
        data = await self._client._request(
            "GET", f"/results/{task_id}/artifacts", params=_revision_params(revision)
        )
        return PublishedArtifactList.model_validate(data)

    async def get_bundle(
        self,
        task_id: str,
        output_path: Path,
        include: Iterable[BundleSection] | None = None,
        revision: str | None = None,
    ) -> None:
        """Download a tar bundle of the task result. See sync variant."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client._download(
            _bundle_path(task_id, include, revision), output_path
        )

    async def download_file(
        self,
        task_id: str,
        filename: str,
        output_path: Path,
        revision: str | None = None,
    ) -> None:
        """Download a published artifact. See sync variant."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client._download(
            _file_path(task_id, filename, revision), output_path
        )

    async def download_logs(self, task_id: str, output_path: Path) -> None:
        """Download archived logs.jsonl for a task."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client._download(f"/results/{task_id}/logs", output_path)

    async def materialize(
        self,
        task_id: str,
        output_dir: Path,
        include: Iterable[BundleSection] | None = None,
        revision: str | None = None,
    ) -> tuple[dict[str, Any], Path, list[Path]]:
        """Fetch the task bundle and extract it. See sync variant."""
        sections = _normalize_include(include)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            prefix=f"flowmesh-bundle-{task_id}-", suffix=".tar", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            await self._client._download(
                _bundle_path(task_id, sections, revision), tmp_path
            )
            extracted = _extract_bundle(tmp_path, output_dir)
        finally:
            tmp_path.unlink(missing_ok=True)

        return _finalize_materialize(output_dir, task_id, sections, extracted)

    async def download_files(
        self,
        task_id: str,
        file_paths: list[str],
        output_dir: Path,
        revision: str | None = None,
    ) -> AsyncIterable[Path]:
        """Download specific result files into an output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in file_paths:
            out_path = output_dir / Path(file_path).name
            await self.download_file(task_id, file_path, out_path, revision)
            yield out_path


def _normalize_include(
    include: Iterable[BundleSection] | None,
) -> tuple[BundleSection, ...]:
    return tuple(dict.fromkeys(include)) if include else _DEFAULT_INCLUDE


def _bundle_path(
    task_id: str,
    include: Iterable[BundleSection] | None,
    revision: str | None = None,
) -> str:
    sections = _normalize_include(include)
    query = "&".join(f"include={s}" for s in sections)
    if revision is not None:
        query += f"&revision={revision}"
    return f"/results/{task_id}/bundle?{query}"


def _file_path(task_id: str, filename: str, revision: str | None) -> str:
    path = f"/results/{task_id}/files/{filename}"
    return path if revision is None else f"{path}?revision={revision}"


def _revision_params(revision: str | None) -> dict[str, str] | None:
    return {"revision": revision} if revision is not None else None


def _extract_bundle(bundle_path: Path, output_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    dest_root = output_dir.resolve()
    with tarfile.open(bundle_path, mode="r:*") as archive:
        for member in archive:
            member_path = (dest_root / member.name).resolve()
            try:
                member_path.relative_to(dest_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe member path in result bundle: {member.name}"
                ) from exc
            archive.extract(member, dest_root, filter="data")
            if member.isfile():
                extracted.append(member_path)
    return extracted


def _finalize_materialize(
    output_dir: Path,
    task_id: str,
    sections: tuple[BundleSection, ...],
    extracted: list[Path],
) -> tuple[dict[str, Any], Path, list[Path]]:
    """Validate the envelope and point _artifacts at the local extracted dir."""
    json_path = output_dir / task_id / "results.json"
    if not json_path.is_file():
        return {}, json_path, extracted

    envelope = ResultEnvelope.model_validate_json(json_path.read_text())
    if _wants_artifacts(sections) and (ctx := envelope.result.artifacts_):
        ctx.base_dir = (output_dir / task_id).resolve().as_posix()
    payload = envelope.model_dump(mode="json")
    json_path.write_text(json.dumps(payload, indent=2))
    return payload, json_path, extracted


def _wants_artifacts(sections: tuple[BundleSection, ...]) -> bool:
    return "artifacts" in sections or "all" in sections
