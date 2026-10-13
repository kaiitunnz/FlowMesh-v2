import logging
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.tasks.specs import TaskSpecStrictBase
from shared.telemetry.propagation import inject_ambient_traceparent
from shared.utils.atomic import atomic_write_text
from shared.utils.http import add_auth_headers
from shared.utils.nofollow import regular_files
from shared.utils.parsing import parse_bool_env
from shared.utils.redact import redact_url

from ...utils.redaction import redact_urls
from ..base_executor import ExecutionError, TaskReference
from .artifacts import is_flowmesh_origin_url


@dataclass(slots=True)
class HTTPDestination:
    """Resolved HTTP destination for posting task results / uploading artifacts."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30
    ignore_error: bool = False


def determine_resume_path(
    spec: TaskSpecStrictBase,
    training_cfg: dict[str, Any],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> Path | None:
    checkpoint_cfg = getattr(spec, "checkpoint", None) or {}
    load_cfg = checkpoint_cfg.get("load") or {}

    if load_cfg:
        return resolve_checkpoint_load(load_cfg, out_dir, logger=logger)

    resume_from = training_cfg.get("resume_from_path")
    if resume_from:
        path = Path(str(resume_from)).expanduser()
        if not path.exists():
            raise ExecutionError(f"Checkpoint path {path} not found for resume")
        return path
    return None


def resolve_checkpoint_load(
    load_cfg: dict[str, Any], out_dir: Path, logger: logging.Logger | None = None
) -> Path:
    source_type = str(load_cfg.get("type", "local")).lower()
    # Local path hint (works for any source type)
    hinted_path = load_cfg.get("path")
    if hinted_path:
        resolved = Path(str(hinted_path)).expanduser()
        if resolved.exists():
            log = logger or load_cfg.get("_logger")
            if log:
                log.info("Resolved checkpoint from local path: %s", resolved)
            return resolved

    task_hint = load_cfg.get("taskId") or load_cfg.get("task_id")
    if task_hint:
        task_str = str(task_hint).strip()
        if task_str:
            base_dir = (
                Path(os.getenv("RESULTS_DIR", "").strip() or "./results")
                .expanduser()
                .resolve()
            )
            artifacts_base = base_dir / task_str / "artifacts"
            guesses = [
                artifacts_base / "final_model",
                artifacts_base / "final_model.tar.gz",
                artifacts_base / "final_model.bin",
                artifacts_base / "final_model.safetensors",
                artifacts_base / "final_model.pt",
                artifacts_base,
            ]
            for candidate in guesses:
                if candidate.exists():
                    log = logger or load_cfg.get("_logger")
                    if log:
                        log.info(
                            "Resolved checkpoint from local task cache: %s", candidate
                        )
                    return candidate

    if source_type == "local":
        local_path = load_cfg.get("path") or load_cfg.get("uri")
        if not local_path:
            raise ExecutionError(
                "checkpoint.load.path is required for local checkpoints"
            )
        resolved = Path(str(local_path)).expanduser()
        if not resolved.exists():
            raise ExecutionError(f"Checkpoint path {resolved} not found")
        log = logger or load_cfg.get("_logger")
        if log:
            log.info("Using local checkpoint path: %s", resolved)
        return resolved

    if source_type == "http":
        url = load_cfg.get("url")
        if not url:
            task_id = load_cfg.get("task_id")
            filename = load_cfg.get("filename")
            base_url = load_cfg.get("base_url") or os.getenv("FLOWMESH_BASE_URL")

            if base_url and task_id and filename:
                from urllib.parse import urljoin

                base = base_url.rstrip("/") + "/"
                artifact_path = f"api/v1/results/{task_id}/files/{filename}"
                load_cfg = dict(load_cfg)
                load_cfg["url"] = urljoin(base, artifact_path)
            else:
                raise ExecutionError(
                    "checkpoint.load.url is required for HTTP checkpoints (set "
                    "base_url or FLOWMESH_BASE_URL)"
                )
        log = logger or load_cfg.get("_logger")
        if log:
            log.info(
                "Downloading checkpoint from %s", redact_url(str(load_cfg.get("url")))
            )
        return download_and_unpack(load_cfg, out_dir)

    raise ExecutionError(f"Unsupported checkpoint load type '{source_type}'")


def download_and_unpack(load_cfg: dict[str, Any], out_dir: Path) -> Path:
    url = load_cfg.get("url")
    if not url:
        raise ExecutionError("checkpoint.load.url is required for HTTP checkpoints")

    headers = {str(k): str(v) for k, v in (load_cfg.get("headers") or {}).items()}
    if is_flowmesh_origin_url(url):
        add_auth_headers(headers)
        inject_ambient_traceparent(headers)
    timeout = float(load_cfg.get("timeoutSec", 60))
    dest_root = out_dir / "imported_checkpoint"
    if dest_root.exists():
        shutil.rmtree(dest_root, ignore_errors=True)
    dest_root.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ExecutionError(
            f"Unsupported checkpoint URL scheme '{parsed.scheme}'; "
            "only http and https are allowed"
        )

    provided_name = load_cfg.get("filename")
    if provided_name:
        temp_name = str(provided_name)
    else:
        parsed_name = Path(parsed.path).name
        temp_name = parsed_name or "checkpoint"

    temp_path = (dest_root / temp_name).resolve()
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with temp_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        raise ExecutionError(
            redact_urls(f"Failed to download checkpoint from {url}: {exc}", url),
            retryable=True,
        ) from exc

    archive_format = str(load_cfg.get("archive_format", "auto")).lower()
    if archive_format == "auto":
        if tarfile.is_tarfile(temp_path):
            fmt = "tar"
        else:
            fmt = "none"
    else:
        fmt = detect_archive_format(archive_format, temp_path.name)
    if fmt == "none":
        return temp_path

    extracted_dir = dest_root / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "zip":
            with zipfile.ZipFile(temp_path, "r") as zip_archive:
                _safe_zip_extract(zip_archive, extracted_dir)
        else:
            with tarfile.open(temp_path, "r:*") as tar_archive:
                tar_archive.extractall(extracted_dir, filter="data")
    except (tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ExecutionError(f"Failed to unpack checkpoint archive: {exc}") from exc

    subdir = load_cfg.get("subdir")
    candidate = select_extracted_subdir(extracted_dir, subdir)
    if not candidate.exists():
        raise ExecutionError(f"Extracted checkpoint path {candidate} does not exist")
    return candidate


def _safe_zip_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for info in archive.infolist():
        target = (dest_resolved / info.filename).resolve()
        if dest_resolved != target and dest_resolved not in target.parents:
            raise ExecutionError(
                f"Refusing to extract zip entry escaping destination: {info.filename}"
            )
        archive.extract(info, dest_resolved)


def detect_archive_format(requested: str, filename: str) -> str:
    if requested in {"zip", "none"}:
        return requested
    if requested not in {"auto", "tar", "tar.gz", "tgz", "tar.bz2", "tar.xz"}:
        requested = "auto"

    lower_name = filename.lower()
    if requested == "zip" or lower_name.endswith(".zip"):
        return "zip"
    if lower_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        return "tar"
    if requested == "none":
        return "none"
    return "tar"


def select_extracted_subdir(extracted_dir: Path, subdir: str | None) -> Path:
    if subdir:
        return extracted_dir / subdir

    entries = [p for p in extracted_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted_dir


def archive_path_for(model_dir: Path) -> Path:
    """Where a published model directory's archive view is named, beside it."""
    return model_dir.with_name(f"{model_dir.name}.tar.gz")


def get_http_destination(spec: TaskSpecStrictBase) -> HTTPDestination | None:
    dest = spec.output and spec.output.destination
    ignore_error = False
    method = "POST"
    url: str | None = None
    headers: dict[str, str] = {}
    timeout: float = 30.0

    if dest and dest.type == "http":
        if dest.method:
            method = dest.method.upper()
        url = dest.url
        if dest.headers:
            headers = dest.headers.copy()
        if dest.timeoutSec is not None:
            timeout = dest.timeoutSec
    elif parse_bool_env("WORKER_UPLOAD_RESULTS", False):
        ignore_error = True
    else:
        return None
    if not url:
        if base_url := os.getenv("FLOWMESH_BASE_URL"):
            url = base_url.rstrip("/") + "/api/v1/results"
        else:
            return None
    if is_flowmesh_origin_url(url):
        add_auth_headers(headers)
        inject_ambient_traceparent(headers)
    return HTTPDestination(
        method=method,
        url=url,
        headers=headers,
        timeout=timeout,
        ignore_error=ignore_error,
    )


def cleanup_artifact_path(
    path: Path | None, logger: logging.Logger | None = None
) -> None:
    """Remove a local file or directory if it exists, ignoring missing paths."""
    if not path:
        return
    try:
        target = Path(path)
    except TypeError:
        return
    try:
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except FileNotFoundError:
        return
    except Exception as exc:
        if logger:
            logger.warning("Failed to remove artifact at %s: %s", target, exc)


def clear_directory(directory: Path) -> None:
    """Remove everything under ``directory``, leaving it empty."""
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def is_cleanup_enabled() -> bool:
    setting = os.getenv("MODEL_CLEANUP_AFTER_UPLOAD")
    if setting is None:
        return False
    normalized = setting.strip().lower()
    return normalized not in {"0", "false", "no", "off"}


def build_artifact_context(spec: TaskSpecStrictBase, out_dir: Path) -> ArtifactContext:
    """Top-level ``_artifacts`` context. ``base_url`` is the destination
    origin (scheme://host[:port]) for HTTP, else ``None``."""
    base_dir = Path(out_dir).resolve().as_posix()
    base_url: str | None = None
    destination = get_http_destination(spec)
    if destination is not None:
        parsed = urlparse(destination.url)
        if parsed.scheme and parsed.netloc:
            base_url = f"{parsed.scheme}://{parsed.netloc.rpartition('@')[2]}"
    return ArtifactContext(base_dir=base_dir, base_url=base_url)


def write_executor_result(
    path: Path, task_id: str, spec: TaskSpecStrictBase, result: BaseExecutorResult
) -> str:
    """Stamp ``_artifacts`` onto ``result``, persist the envelope, and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result.artifacts_ = build_artifact_context(spec, path.parent)
    # A model identifier may be a URL whose credential the dispatch restored.
    if isinstance(model := getattr(result, "model", None), str):
        setattr(result, "model", redact_url(model))
    envelope = ResultEnvelope(task_id=task_id, result=result).model_dump_json(indent=2)
    atomic_write_text(path, envelope)
    return envelope


def maybe_upload_artifacts(
    task: TaskReference,
    out_dir: Path,
    logger: logging.Logger | None = None,
    skip_errors: bool = False,
) -> list[str]:
    """Upload files under `out_dir/artifacts/` when the task has an HTTP
    destination; no-op otherwise. Each file's multipart filename is its
    path relative to `out_dir/artifacts/`. Returns uploaded rel-paths."""
    if not task.task_id:
        raise ExecutionError("Task id missing; cannot upload artifacts")
    out_dir = Path(out_dir).resolve()
    artifacts_dir = out_dir / "artifacts"
    destination = get_http_destination(task.spec)
    if destination is None or not artifacts_dir.is_dir():
        return []

    base_url = destination.url.rstrip("/")
    upload_url = f"{base_url}/{task.task_id}/files"
    uploaded: list[str] = []

    for rel_name, opened in regular_files(artifacts_dir):
        try:
            if isinstance(opened, OSError):
                raise opened
            with opened as fh:
                response = requests.request(
                    destination.method,
                    upload_url,
                    files={"file": (rel_name, fh, "application/octet-stream")},
                    headers=destination.headers,
                    timeout=destination.timeout,
                )
                response.raise_for_status()
                size = os.fstat(fh.fileno()).st_size
        except Exception as exc:
            if not skip_errors:
                raise ExecutionError(
                    redact_urls(
                        f"Artifact upload failed for {rel_name}: {exc}", upload_url
                    ),
                    retryable=True,
                ) from exc
            if logger:
                logger.warning(
                    "Failed to upload artifact %s: %s",
                    rel_name,
                    redact_urls(str(exc), upload_url),
                )
            continue
        if logger:
            logger.info("Uploaded artifact %s (%d bytes)", rel_name, size)
        uploaded.append(rel_name)

    return uploaded


def maybe_upload_traces(
    task: TaskReference,
    out_dir: Path,
    logger: logging.Logger | None = None,
    skip_errors: bool = False,
) -> list[str]:
    """Upload trace JSONL files under `out_dir/logs/` to
    `<api-base>/traces/tasks/{task_id}/{trace_type}` when the task has an HTTP
    destination; no-op otherwise. The destination's `/results` artifact base
    is swapped for `/traces`. Returns uploaded trace types."""
    if not task.task_id:
        raise ExecutionError("Task id missing; cannot upload traces")
    out_dir = Path(out_dir).resolve()
    logs_dir = out_dir / "logs"
    destination = get_http_destination(task.spec)
    if destination is None or not logs_dir.is_dir():
        return []

    upload_base = destination.url.rstrip("/").removesuffix("/results") + "/traces"
    uploaded: list[str] = []

    for trace_type in ("spans", "assets", "lineage"):
        file_path = logs_dir / f"{trace_type}.jsonl"
        if not file_path.is_file():
            continue
        upload_url = f"{upload_base}/tasks/{task.task_id}/{trace_type}"
        try:
            with file_path.open("rb") as fh:
                response = requests.request(
                    destination.method,
                    upload_url,
                    files={"file": (file_path.name, fh, "application/octet-stream")},
                    headers=destination.headers,
                    timeout=destination.timeout,
                )
                response.raise_for_status()
        except Exception as exc:
            if not skip_errors:
                raise ExecutionError(
                    redact_urls(
                        f"Trace upload failed for {file_path}: {exc}", upload_url
                    ),
                    retryable=True,
                ) from exc
            if logger:
                logger.warning(
                    "Failed to upload trace %s: %s",
                    trace_type,
                    redact_urls(str(exc), upload_url),
                )
            continue
        if logger:
            logger.info(
                "Uploaded trace %s (%d bytes)", trace_type, file_path.stat().st_size
            )
        uploaded.append(trace_type)

    return uploaded
