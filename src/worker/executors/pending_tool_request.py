"""Worker-process-private store for captured, not-yet-executed tool requests.

When a worker originates a mediated tool boundary it keeps the raw request here, keyed
by the stable ``(agent_task_id, call_correlation)`` occurrence, and sends the control
plane only a digest. The off-lane tool-operation executor reads the request back from
this store on the same worker, so the raw request never crosses to the control plane.

The store lives for one worker incarnation. A worker restart is a new incarnation with a
new id and generation, which invalidates any outstanding permit and forces the boundary
to be re-proposed (re-captured) on the freshly assigned worker — so a durable backing is
not needed here. (A future non-deterministic or expensive backend that must relocate an
in-flight operation without re-running the agent step would recover the request through
an authorized worker-private reference; that is not built here.)
"""

import threading

from shared.tools.search.schema import ToolRequest

_lock = threading.Lock()
_store: dict[tuple[str, str], ToolRequest] = {}


def put(agent_task_id: str, call_correlation: str, request: ToolRequest) -> None:
    """Record a captured request for its occurrence, overwriting a stale re-capture."""
    with _lock:
        _store[(agent_task_id, call_correlation)] = request


def take(agent_task_id: str, call_correlation: str) -> ToolRequest | None:
    """Remove and return the request for an occurrence, or None if absent."""
    with _lock:
        return _store.pop((agent_task_id, call_correlation), None)


def peek(agent_task_id: str, call_correlation: str) -> ToolRequest | None:
    """Return the request for an occurrence without removing it (a read-only affordance
    for tests; the executor always consumes via ``take``)."""
    with _lock:
        return _store.get((agent_task_id, call_correlation))
