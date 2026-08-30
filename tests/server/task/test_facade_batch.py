"""The turn-scoped facade batch: one ordered resume, crash-safe, capped, fail-closed.

A batch of same-interface search calls captured in one turn resumes the episode exactly
once with the full ordered outcome vector; a crash reissues only unresolved members and
never double-delivers a recorded one; an oversized batch settles its overflow as typed
quota outcomes; and mixing search with spawn is denied fail-closed.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from server.config import OrchestrationConfig, WebSearchConfig
from server.orchestration.state import WorkItemStatus
from server.orchestration.tool_dispatch import (
    FacadeBatchMember,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from server.task.runtime import TaskRuntime
from shared.harness import HarnessResult, HarnessResultKind
from tests.server.task.test_v2_agent_harness import (
    _agent,
    _bundle,
    _decl,
    _dispatch_agent,
    _engine,
)
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _register,
    _WorkerRegistryStub,
)

_GRANT = frozenset({"search/v1"})
_TS = "2026-08-31T00:00:00Z"

_SEARCH_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: search-agent}
spec:
  graph:
    nodes:
      - name: searcher
        spec:
          taskType: agent
          v2:
            authority: {invoke: [search/v1], delegate: []}
            tools: [{name: web_search, interface: "search/v1"}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def _search_engine():
    eng = _engine(
        _bundle(
            [_agent("A", invoke=("search/v1",), delegate=())],
            [],
            (_decl("out:A", "A"),),
        ),
        granted=_GRANT,
    )
    _dispatch_agent(eng)
    return eng


def _members(n: int) -> list[FacadeBatchMember]:
    return [
        FacadeBatchMember(
            interface="search/v1",
            call_correlation=f"A:0:{i}",
            ordinal=i,
            original_call_id=f"call{i}",
            tool_name="web_search",
            request_payload=f'{{"query": "q{i}"}}',
        )
        for i in range(n)
    ]


def test_gate1_reverse_completion_is_one_ordered_resume() -> None:
    eng = _search_engine()
    eng.route_boundary_batch("A", "A:0", _members(3))
    assert eng.work_item("A").status is WorkItemStatus.BLOCKED
    # Settle out of order (2, 0, 1): the lane stays suspended until the last member.
    for corr in ("A:0:2", "A:0:0"):
        advance = eng.settle_boundary_outcome("A", corr, value=f"r-{corr}")
        assert not advance.ready
        assert eng.work_item("A").status is WorkItemStatus.BLOCKED
    advance = eng.settle_boundary_outcome("A", "A:0:1", value="r-A:0:1")
    assert advance.ready == ["A"]  # resumed exactly once, only now
    _, outcomes = eng.episode_context("A")
    # The full vector is ordered by source ordinal and maps to each original call id.
    assert [o.value for o in outcomes] == ["r-A:0:0", "r-A:0:1", "r-A:0:2"]
    assert [o.injection_target for o in outcomes] == ["call0", "call1", "call2"]
    assert [o.injection_tool for o in outcomes] == ["web_search"] * 3


def test_gate2_a_crash_reissues_only_unresolved_members() -> None:
    eng = _search_engine()
    eng.route_boundary_batch("A", "A:0", _members(3))
    eng.settle_boundary_outcome("A", "A:0:1", value="done")
    # A restart re-lists only the members with no durable outcome yet.
    pending = eng.pending_tool_dispatches()
    corrs = sorted(e.call_correlation for e in pending)
    assert corrs == ["A:0:0", "A:0:2"]


def test_gate3_all_recorded_injects_the_record_without_requery() -> None:
    eng = _search_engine()
    eng.route_boundary_batch("A", "A:0", _members(3))
    for i in range(3):
        eng.settle_boundary_outcome("A", f"A:0:{i}", value=f"r{i}")
    # Every member is durable: nothing is redispatched, and the recorded ordered vector
    # is what a resume injects — never a re-query.
    assert eng.pending_tool_dispatches() == []
    _, outcomes = eng.episode_context("A")
    assert [o.value for o in outcomes] == ["r0", "r1", "r2"]


def test_gate4_a_late_duplicate_settle_does_not_requeue() -> None:
    eng = _search_engine()
    eng.route_boundary_batch("A", "A:0", _members(2))
    eng.settle_boundary_outcome("A", "A:0:0", value="r0")
    eng.settle_boundary_outcome("A", "A:0:1", value="r1")  # completes, re-readies once
    assert eng.work_item("A").status is WorkItemStatus.READY
    # A duplicate/late settle for an already-resolved member re-readies nothing.
    advance = eng.settle_boundary_outcome("A", "A:0:0", value="r0-again")
    assert not advance.ready
    _, outcomes = eng.episode_context("A")
    assert outcomes[0].value == "r0"  # the first recorded outcome stands


def _runtime(max_parallel: int) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(web_search=WebSearchConfig(max_parallel=max_parallel)),
        Path(tempfile.gettempdir()),
        logging.getLogger("batch-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def test_gate5_an_oversized_batch_settles_overflow_as_quota() -> None:
    async def run() -> None:
        runtime = _runtime(max_parallel=2)
        dispatched: list[ToolInvocationEnvelope] = []
        runtime.set_tool_broker(dispatched.append)
        workflow_id, ids = await _register(runtime, _SEARCH_WF)
        task = ids["searcher"]
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None
        engine.on_dispatched(task, "w1")
        members = [
            FacadeBatchMember(
                interface="search/v1",
                call_correlation=f"{task}:0:{i}",
                ordinal=i,
                original_call_id=f"c{i}",
                tool_name="web_search",
                request_payload=f'{{"query": "q{i}"}}',
            )
            for i in range(4)
        ]
        runtime.originate_episode_batch(task, f"{task}:0", members)
        completion = HarnessResult(
            kind=HarnessResultKind.COMPLETION, value=None, capsule=None
        )
        runtime.mark_succeeded(
            task, "w1", {"agent_episode": completion.model_dump(mode="json")}, _TS
        )
        # Only the first two members (the parallel cap) dispatch to the broker.
        assert len(dispatched) == 2
        assert sorted(e.call_correlation for e in dispatched) == [
            f"{task}:0:0",
            f"{task}:0:1",
        ]
        # Settle the two active members; the two overflow members are already quota, so
        # the batch completes with a full four-outcome vector — no 500, no truncation.
        for env in dispatched:
            runtime.settle_episode_invocation(
                task, env.call_correlation, '{"status": "success", "value": "hit"}'
            )
        _, outcomes = engine.episode_context(task)
        assert len(outcomes) == 4
        statuses = [
            ToolOutcome.model_validate_json(o.value or "{}").status for o in outcomes
        ]
        assert statuses[2:] == [ToolOutcomeStatus.QUOTA, ToolOutcomeStatus.QUOTA]

    asyncio.run(run())


def test_gate6_native_shell_is_permitted_without_network_egress() -> None:
    # The codex sandbox permits native shell but denies it network, so a native curl
    # cannot bypass the mediated search/v1 facade — the only egress.
    from pathlib import Path as _Path

    from worker.executors.harness.codex_transport import CodexTransportConfig

    cfg = CodexTransportConfig(
        base_url="http://gw",
        model="m",
        codex_home=_Path(tempfile.gettempdir()) / "ch",
        initial_input="t",
        task_id="tsk-1",
    )
    overrides = cfg.to_codex_config().config_overrides
    assert 'sandbox_mode="workspace-write"' in overrides  # shell permitted
    assert "sandbox_workspace_write.network_access=false" in overrides  # no egress
    assert "tools.web_search=false" in overrides  # native web search hidden
