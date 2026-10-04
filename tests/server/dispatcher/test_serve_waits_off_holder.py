"""A serve task waits rather than start on an agent's private-state holder."""

import asyncio
import logging
from pathlib import Path

from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.dispatcher.test_cordon_drain import _offered, _registry
from tests.server.dispatcher.test_private_state_holder_yield import _SERVE
from tests.server.task.test_v2_orchestration import _register
from tests.server.task.test_worker_originated_boundary import (
    _HOLDER,
    _SEARCH_WF,
    _dispatch_agent,
    _report,
    _runtime,
)


def test_a_serve_task_waits_for_the_agent_on_the_only_idle_worker(
    tmp_path: Path,
) -> None:
    registry = _registry("holder")
    runtime = _runtime()
    _, ids = asyncio.run(_register(runtime, _SEARCH_WF))
    writer = ids["writer"]
    _dispatch_agent(runtime, writer, seal_in=tmp_path)
    _report(runtime, writer, "m0", "sunny")
    _, entries = asyncio.run(runtime.register("owner", "org", _SERVE, format="native"))
    serve = entries[0].task_id
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("serve-holder-test"),
        no_worker_grace_sec=0,
    )

    assert dispatcher.dispatch_once(serve) is False
    assert dispatcher.requeued == [
        (serve, {"reason": "no_idle_worker", "count_retry": False})
    ]
    assert dispatcher.failed == []

    assert _offered(registry, runtime, writer) == [[_HOLDER.worker_id]]
    _dispatch_agent(runtime, writer)
    assert runtime.private_state_holders() == []
    assert _offered(registry, runtime, serve) == [[_HOLDER.worker_id]]
