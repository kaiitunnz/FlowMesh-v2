"""Stages an SSH task's ``inputs[]`` in a local directory: the artifacts each upstream
published, which the worker staged from its content plane, with its hydrated result
envelope beside them."""

import shutil
import tempfile
from pathlib import Path

from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.manifest import RESULTS_NAME

from ..base_executor import ExecutionError, RunSignals
from .base import SessionInterrupted
from .config import (
    DEFAULT_INPUTS_ROOT,
    ResolvedSSHInput,
    SSHConfig,
    normalize_mount_path,
)


def resolve_inputs(
    task: WorkerTaskMessage, cfg: SSHConfig, results_root: Path
) -> list[ResolvedSSHInput]:
    if not cfg.inputs:
        return []

    resolved: list[ResolvedSSHInput] = []
    upstream_results = task.upstream_results or {}
    upstream_task_ids = task.upstream_task_ids or {}
    for entry in cfg.inputs:
        stage = entry.stage.strip()
        if not stage:
            raise ExecutionError("SSH input stage names must be non-empty")
        binding = upstream_results.get(stage)
        task_id = binding.task_id if binding else upstream_task_ids.get(stage)
        if not task_id:
            raise ExecutionError(
                f"Missing resolved upstream task ID for SSH input stage '{stage}'"
            )
        resolved.append(
            ResolvedSSHInput(
                stage=stage,
                task_id=task_id,
                results_root=results_root,
                mount_path=normalize_mount_path(
                    entry.mountPath or f"{DEFAULT_INPUTS_ROOT}/{stage}",
                    field_name=f"inputs[{stage}].mountPath",
                ),
                results=task.upstream_envelope(stage) if binding else None,
                staged_path=(
                    Path(staged) if (staged := task.staged_input(task_id)) else None
                ),
            )
        )
    return resolved


def stage_inputs_locally(
    resolved_inputs: list[ResolvedSSHInput],
    session_id: str,
    signals: RunSignals,
    parent: Path | None = None,
) -> Path:
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"flowmesh-ssh-inputs-{session_id[:8]}-", dir=parent)
    )
    try:
        _fill_staging_dir(staging_dir, resolved_inputs, signals)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return staging_dir


def _fill_staging_dir(
    staging_dir: Path, resolved_inputs: list[ResolvedSSHInput], signals: RunSignals
) -> None:
    for resolved in resolved_inputs:
        if signals.interrupted:
            raise SessionInterrupted
        destination = staging_dir / resolved.task_id
        # Links copy as links, so staging never reads what one points to with the
        # worker's access.
        if resolved.staged_path is not None:
            shutil.copytree(
                resolved.staged_path, destination, symlinks=True, dirs_exist_ok=True
            )
        destination.mkdir(parents=True, exist_ok=True)
        if resolved.results is not None:
            (destination / RESULTS_NAME).write_bytes(resolved.results)
