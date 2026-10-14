import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse

import requests

from shared.schemas.artifact import ArtifactContext, artifact_download_path
from shared.schemas.result import BaseExecutorResult
from shared.schemas.result.binding import artifact_context
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.specs.common import ModelInferSpecStrict
from shared.utils.http import auth_headers

from ...utils.redaction import redact_urls
from ..base_executor import ExecutionError
from .expressions import element_paths, item_path, project_expression_paths


def artifact_ref_source(
    value: Any,
    context: Mapping[str, BaseExecutorResult] | None,
    node: str | None,
    path: Sequence[str | int] = (),
) -> tuple[str, str] | None:
    """The producer task and artifact path a `{path: ...}` ref names, for the value at
    ``path`` inside upstream ``node``; None for any other value."""
    if (found := _ref_context(value, context, node, path)) is None:
        return None
    task_id, _, rel_path = found
    return task_id, rel_path


def _ref_context(
    value: Any,
    context: Mapping[str, BaseExecutorResult] | None,
    node: str | None,
    path: Sequence[str | int],
) -> tuple[str, ArtifactContext, str] | None:
    if not isinstance(value, dict) or "path" not in value:
        return None
    rel_path = value.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise ExecutionError("Artifact ref must include a non-empty 'path' field")
    ctx = (
        artifact_context(node_result, path)
        if context and node and (node_result := context.get(node))
        else None
    )
    if ctx is None or ctx.task_id is None:
        raise ExecutionError(
            "Cannot resolve artifact ref: upstream _artifacts context is missing"
        )
    return ctx.task_id, ctx, rel_path


def hydrated_artifact_ref(
    value: Any,
    spec: TaskSpecStrictBase,
    context: Mapping[str, BaseExecutorResult] | None,
    node: str | None,
    path: Sequence[str | int] = (),
) -> Any:
    """A `{path: ...}` ref as the local path its artifact was hydrated to before the
    task ran; any other value as it is."""
    if (named := artifact_ref_source(value, context, node, path)) is None:
        return value
    if (local := spec.hydrated_artifact(*named)) is None:
        raise ExecutionError(
            f"input_unreadable: task {named[0]} published no artifact named "
            f"{named[1]!r}"
        )
    return local


def artifact_ref_text(
    value: Any,
    context: Mapping[str, BaseExecutorResult] | None,
    node: str | None,
    path: Sequence[str | int] = (),
) -> Any:
    """A `{path: ...}` ref as the text naming its published artifact; any other value
    as it is."""
    if (found := _ref_context(value, context, node, path)) is None:
        return value
    task_id, ctx, rel_path = found
    return artifact_download_path(task_id, rel_path, ctx.revision_of(rel_path))


def list_items_expression(data: Mapping[str, Any]) -> str | None:
    """The expression list data projects its items from, when it names no items."""
    if data.get("items") is not None:
        return None
    expr = data.get("expr")
    if not expr:
        node, path = data.get("node"), data.get("path")
        if node and path:
            expr = f"{node}.{path}"
    return expr.strip() if isinstance(expr, str) and expr.strip() else None


def image_embedding_expression(
    raw: Mapping[str, Any],
) -> tuple[str, str | None] | None:
    """The expression an image embedding projects, with the upstream node it reads."""
    expr = raw.get("expr")
    node: str | None = raw.get("node")
    if not expr and node and (path_hint := raw.get("path")):
        expr = f"{node}.{path_hint}"
    if not expr or not isinstance(expr, str):
        return None
    return expr.strip(), node or expr.split(".", 1)[0].strip() or None


def data_artifact_refs(spec: TaskSpecStrictBase) -> list[tuple[str, str]]:
    """The producer task and path of each artifact a spec's data loads the bytes of:
    the images a visual-embedding model reads from its projected list items, and an
    image embedding.

    A projection that fails names nothing here; the executor reports it when it runs.
    """
    if not isinstance(spec, ModelInferSpecStrict) or not spec.data:
        return []
    data, context = spec.data, spec.upstreamResults or {}
    named: list[tuple[str, str]] = []

    def collect(value: Any, node: str | None, path: Sequence[str | int]) -> None:
        try:
            found = artifact_ref_source(value, context, node, path)
        except ExecutionError:
            return
        if found is not None and found not in named:
            named.append(found)

    reads_images = spec.model is not None and spec.model.visual_embedding
    if (
        reads_images
        and data.get("type") == "list"
        and (expr := list_items_expression(data))
    ):
        try:
            items, paths = project_expression_paths(expr, context)
        except Exception:
            items, paths = None, ()
        node = expr.split(".", 1)[0] or None
        for index, item in enumerate(items if isinstance(items, list) else ()):
            if isinstance(item, list):
                for position, member in enumerate(item):
                    collect(
                        member, node, item_path(element_paths(paths, index), position)
                    )
            else:
                collect(item, node, item_path(paths, index))
    if (raw := data.get("image_embedding")) and (
        embedding := image_embedding_expression(raw)
    ):
        expr, node = embedding
        try:
            value, value_paths = project_expression_paths(expr, context)
        except Exception:
            return named
        collect(value, node, value_paths if isinstance(value_paths, tuple) else ())
    return named


_DEFAULT_PORT_BY_SCHEME = {"http": 80, "https": 443}


def _effective_port(parsed: ParseResult) -> int | None:
    if parsed.port is not None:
        return parsed.port
    return _DEFAULT_PORT_BY_SCHEME.get(parsed.scheme.lower())


def is_flowmesh_origin_url(url: str) -> bool:
    """Whether `url` targets this worker's configured FlowMesh server.

    Gates whether the worker's bearer token (`FLOWMESH_API_KEY`, attached via
    `auth_headers()`) may be sent: only to the worker's own FlowMesh origin
    (`FLOWMESH_BASE_URL`), never to an arbitrary or public URL a workflow
    happens to reference. Compares scheme, hostname, and effective port
    (defaulting http/https to 80/443) so `https://flowmesh` and
    `https://flowmesh:443` are recognized as the same origin.
    """
    base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
    if not base_url:
        return False
    base = urlparse(base_url)
    if not base.scheme or not base.netloc:
        return False
    target = urlparse(url)
    return (
        target.scheme.lower() == base.scheme.lower()
        and (target.hostname or "").lower() == (base.hostname or "").lower()
        and _effective_port(target) == _effective_port(base)
    )


def resolve_artifact(source: str, timeout: float = 1800) -> Path:
    """Download an artifact URL (or symlink a local path) into a tempfile.

    `source` is either an absolute HTTP/HTTPS URL or an absolute filesystem
    path, such as the one a `{path: ...}` ref was hydrated to.
    """
    if not isinstance(source, str) or not source:
        raise ExecutionError("Artifact source is missing for embedding resolution")

    parsed = urlparse(source)
    temp_dir = Path(tempfile.mkdtemp(prefix="flowmesh-artifact-"))
    filename = Path(parsed.path).name or "artifact"
    local_path = temp_dir / filename

    if not parsed.scheme:
        if not (src_path := Path(source)).is_file():
            raise ExecutionError(f"Artifact source path does not exist: {src_path}")
        try:
            os.symlink(src_path.resolve(), local_path)
        except Exception as exc:
            raise ExecutionError(
                f"Failed to link artifact from {src_path}: {exc}"
            ) from exc
        return local_path

    try:
        headers = auth_headers() if is_flowmesh_origin_url(source) else None
        with requests.get(source, headers=headers, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_path
    except Exception as exc:
        raise ExecutionError(
            redact_urls(f"Failed to resolve embedding artifact: {exc}", source),
            retryable=True,
        ) from exc
