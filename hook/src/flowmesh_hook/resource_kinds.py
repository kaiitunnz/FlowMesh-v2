"""FlowMesh's resource and action vocabulary.

`lumid-hooks` keeps `kind` / `action` as plain strings; FlowMesh layers a
`StrEnum` on top so call sites get auto-complete and exhaustiveness checks.
The enums are `str`-compatible, so they pass straight into the shared
protocols' `kind: str` / `action: str` parameters with no `.value`.
"""

from enum import StrEnum


class ResourceKind(StrEnum):
    """Resource kinds in FlowMesh's permission contract.

    A `RESULT` check names one task's result by its `task_id`, or names no id
    to gate result values as a whole; workflow-level operations (logs,
    queries) check `WORKFLOW`.
    """

    WORKFLOW = "workflow"
    TASK = "task"
    RESULT = "result"
    NODE = "node"
    WORKER = "worker"
    SYSTEM = "system"


class ResourceAction(StrEnum):
    """Actions in FlowMesh's permission contract."""

    READ = "read"
    WRITE = "write"
    CANCEL = "cancel"
    ADMIN = "admin"
