import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests

from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.tasks.specs import TaskSpecStrictBase
from shared.telemetry.propagation import inject_ambient_traceparent
from shared.utils.atomic import atomic_write_text
from shared.utils.http import add_auth_headers
from shared.utils.redact import redact_url

from ...utils.redaction import redact_urls
from ..base_executor import ExecutionError, TaskReference
from .artifacts import is_flowmesh_origin_url


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


def hydrated_path(value: str) -> Path | None:
    """The local path a reference to a published artifact rendered as, once the
    worker hydrated it; None for a URL."""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() and path.exists() else None


def resolve_checkpoint_load(
    load_cfg: dict[str, Any], out_dir: Path, logger: logging.Logger | None = None
) -> Path:
    """The local path a ``checkpoint.load`` names, downloading it when it is a URL.

    A published artifact the worker hydrated resolves whatever ``type`` declares, and
    ``base_url`` with ``task_id`` and ``filename`` fetches that artifact from another
    deployment's download route.
    """
    log = logger or load_cfg.get("_logger")
    url = str(load_cfg.get("url") or "")
    if (local := hydrated_path(url)) is not None:
        if log:
            log.info("Using hydrated checkpoint: %s", local)
        return local
    if not url and (base_url := load_cfg.get("base_url")):
        producer = load_cfg.get("task_id") or load_cfg.get("taskId")
        filename = load_cfg.get("filename")
        if not (producer and filename):
            raise ExecutionError(
                "checkpoint.load.base_url needs the task_id and filename of the "
                "artifact to fetch"
            )
        url = urljoin(
            str(base_url).rstrip("/") + "/",
            f"api/v1/results/{quote(str(producer))}/files/{quote(str(filename))}",
        )
        load_cfg = {**load_cfg, "url": url, "type": "http"}

    if hinted_path := load_cfg.get("path"):
        resolved = Path(str(hinted_path)).expanduser()
        if resolved.exists():
            if log:
                log.info("Resolved checkpoint from local path: %s", resolved)
            return resolved

    source_type = str(load_cfg.get("type", "local")).lower()
    if source_type == "local":
        local_path = load_cfg.get("path") or load_cfg.get("uri")
        if not local_path:
            raise ExecutionError(
                "checkpoint.load.path is required for local checkpoints"
            )
        resolved = Path(str(local_path)).expanduser()
        if not resolved.exists():
            raise ExecutionError(f"Checkpoint path {resolved} not found")
        if log:
            log.info("Using local checkpoint path: %s", resolved)
        return resolved

    if source_type == "http":
        if not url:
            raise ExecutionError(
                "checkpoint.load needs a url, or the task_id of a task that published "
                "the checkpoint"
            )
        if log:
            log.info("Downloading checkpoint from %s", redact_url(url))
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


def write_executor_result(
    path: Path,
    task_id: str,
    spec: TaskSpecStrictBase,
    result: BaseExecutorResult,
    checkpoints: str | None = None,
) -> str:
    """Stamp ``_artifacts`` onto ``result``, persist the envelope, and return it.

    ``checkpoints`` is the revision of the last checkpoint the task published.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    result.artifacts_ = ArtifactContext(task_id=task_id, checkpoints=checkpoints)
    # A model identifier may be a URL whose credential the dispatch restored.
    if isinstance(model := getattr(result, "model", None), str):
        setattr(result, "model", redact_url(model))
    envelope = ResultEnvelope(task_id=task_id, result=result).model_dump_json(indent=2)
    atomic_write_text(path, envelope)
    return envelope


_TRACE_UPLOAD_TIMEOUT_SEC = 30.0


def maybe_upload_traces(
    task: TaskReference,
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Upload the trace JSONL files under `out_dir/logs/` to the server's
    `/api/v1/traces/tasks/{task_id}/{trace_type}` when `FLOWMESH_BASE_URL` names it.

    A trace that fails to upload is logged and left out; the task's own work does not
    depend on it. Returns the uploaded trace types.
    """
    if not task.task_id:
        raise ExecutionError("Task id missing; cannot upload traces")
    logs_dir = Path(out_dir).resolve() / "logs"
    base_url = os.getenv("FLOWMESH_BASE_URL", "").rstrip("/")
    if not base_url or not logs_dir.is_dir():
        return []

    headers: dict[str, str] = {}
    add_auth_headers(headers)
    inject_ambient_traceparent(headers)
    uploaded: list[str] = []
    for trace_type in ("spans", "assets", "lineage"):
        file_path = logs_dir / f"{trace_type}.jsonl"
        if not file_path.is_file():
            continue
        upload_url = f"{base_url}/api/v1/traces/tasks/{task.task_id}/{trace_type}"
        try:
            with file_path.open("rb") as fh:
                response = requests.post(
                    upload_url,
                    files={"file": (file_path.name, fh, "application/octet-stream")},
                    headers=headers,
                    timeout=_TRACE_UPLOAD_TIMEOUT_SEC,
                )
                response.raise_for_status()
        except Exception as exc:
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
