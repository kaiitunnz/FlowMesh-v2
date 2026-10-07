"""Physical task records projected for materialized children."""

import time

from ..models import TaskRecord, TaskStatus


def synthesize_child_record(template: TaskRecord, child_task_id: str) -> TaskRecord:
    """Clone a child-template record for one materialized child."""
    return template.model_copy(
        deep=True,
        update={
            "task_id": child_task_id,
            "status": TaskStatus.PENDING,
            "assigned_worker": None,
            "started_ts": None,
            "finished_ts": None,
            "error": None,
            "usages": [],
            "position_in_epoch": None,
            "graph_node_name": None,
            "local_name": None,
            "merge_key": None,
            "merged_children": None,
            "selected_worker": None,
            "submitted_ts": time.time(),
        },
    )
