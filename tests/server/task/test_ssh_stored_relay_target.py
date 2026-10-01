"""An SSH task stored while workers published a relay target for the supervisor to
dial still loads, and the target is ignored."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from server.routers.v1.tasks import _strip_private_fields
from server.ssh import resolve_relay_target
from server.task.models import TaskStatus
from tests.server.task.test_task_credentials import _runtime
from tests.server.task.test_v2_orchestration import FakeRegistry

# Written by a build whose SSH executor published `_relay_target`.
_STORED = Path(__file__).parent / "fixtures" / "ssh_relay_target_records.json"


def test_a_stored_session_with_a_relay_target_loads_and_relays_by_its_endpoint() -> (
    None
):
    stored = json.loads(_STORED.read_text())
    registry = FakeRegistry()
    registry.task_blobs.update(stored["task_blobs"])
    registry.sched.update(stored["sched"])
    registry.workflow_task_ids.update(stored["workflow_task_ids"])
    runtime = _runtime(registry)

    assert asyncio.run(runtime.rehydrate()) == 1
    record = runtime.get_record(stored["task_id"])
    assert record is not None and record.status == TaskStatus.DISPATCHED
    ssh = cast(dict[str, Any], record.latest_update)["ssh"]
    assert "_relay_target" not in _strip_private_fields(ssh)

    workers = MagicMock()
    workers.get_worker_async = AsyncMock(
        return_value=MagicMock(id="wkr-1", node_id="nde-1")
    )
    target = asyncio.run(resolve_relay_target(record, workers))
    assert (target.worker_id, target.endpoint_id) == ("wkr-1", "ssn-stored")
