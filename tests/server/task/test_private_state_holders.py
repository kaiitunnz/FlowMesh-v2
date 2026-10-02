"""A worker holds an activation's private state only until the activation settles."""

import asyncio
from pathlib import Path

from tests.server.task.test_agent_episode_runtime import (
    _AGENT_WF,
    _HOLDER,
    _adapter,
    _step,
)
from tests.server.task.test_private_state_ledger import _manifest
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _runtime


async def _sealed_agent(tmp_path: Path):
    runtime = _runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _AGENT_WF)
    writer = ids["writer"]
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    granted = engine.grant_private_state(writer, _HOLDER.worker_id, _HOLDER.incarnation)
    assert granted is not None
    binding, attachment = granted
    engine.seal_private_state(
        writer,
        _manifest(tmp_path, binding.reference.reference_id, 1),
        attachment.write_epoch,
    )
    return runtime, workflow_id, writer


def test_a_sealed_agent_holds_its_worker_until_it_settles(tmp_path: Path) -> None:
    async def run() -> None:
        runtime, workflow_id, writer = await _sealed_agent(tmp_path)
        assert runtime.private_state_holders() == {_HOLDER}

        adapter = _adapter()
        for _ in range(3):  # spawn, seal the region, complete
            assert runtime.private_state_holders() == {_HOLDER}
            _step(runtime, adapter, writer)

        assert runtime.private_state_holders() == set()

    asyncio.run(run())


def test_a_cancelled_agent_releases_its_worker(tmp_path: Path) -> None:
    async def run() -> None:
        runtime, workflow_id, _writer = await _sealed_agent(tmp_path)

        runtime.cancel_workflow(workflow_id)

        assert runtime.private_state_holders() == set()

    asyncio.run(run())
