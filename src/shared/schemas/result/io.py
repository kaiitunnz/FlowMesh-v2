"""Where a task's ``results.json`` lives, and how a stored result is typed."""

from pathlib import Path

# How a stored result envelope is typed in the content store.
RESULT_MEDIA_TYPE = "application/json"


def _sanitize_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)


def result_file_path(base_dir: Path, task_id: str) -> Path:
    return base_dir / _sanitize_task_id(task_id) / "results.json"
