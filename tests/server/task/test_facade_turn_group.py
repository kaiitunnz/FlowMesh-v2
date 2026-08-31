"""The turn-scoped facade group: kind-specific completion over one ordered turn.

A model turn's facade calls join one ``FacadeTurnGroup``. Search members hold the resume
gate and resume the episode once with the full ordered outcome vector; spawn members
admit one child each at admission and never hold the gate; a mixed group is held only by
its search; a per-member budget overflow is a typed quota outcome that spares its
siblings; and a crash reissues only the unresolved search members, never
double-delivering a recorded one.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from server.config import OrchestrationConfig, WebSearchConfig
from server.orchestration.state import (
    BoundaryEvent,
    BoundaryEventKind,
    ProgressAxis,
    WorkItemStatus,
)
from server.orchestration.tool_dispatch import (
    FacadeCallMember,
    FacadeCompletionMode,
    FacadeTurnGroup,
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
    _leaf,
    _region,
    _spawning_agent,
)
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _register,
    _WorkerRegistryStub,
)

_GRANT = frozenset({"search/v1"})
_MIXED_GRANT = frozenset({"search/v1", "model"})
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


def _spawn_engine():
    eng = _engine(_spawning_agent(child=_leaf("child")))
    _dispatch_agent(eng)
    return eng


def _search_member(ordinal: int, group_id: str = "A:0") -> FacadeCallMember:
    return FacadeCallMember(
        ordinal=ordinal,
        kind=BoundaryEventKind.INVOCATION,
        completion_mode=FacadeCompletionMode.AWAIT_OUTCOME,
        call_correlation=f"{group_id}:{ordinal}",
        harness_call_id=f"call{ordinal}",
        tool_name="web_search",
        interface_or_region="search/v1",
        request_payload=f'{{"query": "q{ordinal}"}}',
    )


def _spawn_member(
    ordinal: int, *, region: str = "worker", group_id: str = "A:0"
) -> FacadeCallMember:
    return FacadeCallMember(
        ordinal=ordinal,
        kind=BoundaryEventKind.SPAWN,
        completion_mode=FacadeCompletionMode.ADMIT_AND_CLOSE,
        call_correlation=f"{group_id}:{ordinal}",
        harness_call_id=f"spawn{ordinal}",
        tool_name="spawn_agent",
        interface_or_region=region,
        request_payload=f'{{"facet": "f{ordinal}"}}',
    )


def _group(*members: FacadeCallMember, group_id: str = "A:0") -> FacadeTurnGroup:
    return FacadeTurnGroup(
        group_id=group_id, activation_id="A", turn_id="0", members=tuple(members)
    )


def _search_group(n: int, group_id: str = "A:0") -> FacadeTurnGroup:
    return _group(*(_search_member(i, group_id) for i in range(n)), group_id=group_id)


# --------------------------------------------------------------------------- #
# Search members: one ordered resume, crash-safe
# --------------------------------------------------------------------------- #


def test_reverse_order_completion_yields_one_ordered_resume() -> None:
    eng = _search_engine()
    eng.route_facade_turn_group("A", _search_group(3))
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


def test_a_crash_reissues_only_unresolved_members() -> None:
    eng = _search_engine()
    eng.route_facade_turn_group("A", _search_group(3))
    eng.settle_boundary_outcome("A", "A:0:1", value="done")
    # A restart re-lists only the members with no durable outcome yet.
    pending = eng.pending_tool_dispatches()
    corrs = sorted(e.call_correlation for e in pending)
    assert corrs == ["A:0:0", "A:0:2"]


def test_a_fully_recorded_group_injects_without_requery() -> None:
    eng = _search_engine()
    eng.route_facade_turn_group("A", _search_group(3))
    for i in range(3):
        eng.settle_boundary_outcome("A", f"A:0:{i}", value=f"r{i}")
    # Every member is durable: nothing is redispatched, and the recorded ordered vector
    # is what a resume injects — never a re-query.
    assert eng.pending_tool_dispatches() == []
    _, outcomes = eng.episode_context("A")
    assert [o.value for o in outcomes] == ["r0", "r1", "r2"]


def test_a_late_duplicate_settle_does_not_requeue() -> None:
    eng = _search_engine()
    eng.route_facade_turn_group("A", _search_group(2))
    eng.settle_boundary_outcome("A", "A:0:0", value="r0")
    eng.settle_boundary_outcome("A", "A:0:1", value="r1")  # completes, re-readies once
    assert eng.work_item("A").status is WorkItemStatus.READY
    # A duplicate/late settle for an already-resolved member re-readies nothing.
    advance = eng.settle_boundary_outcome("A", "A:0:0", value="r0-again")
    assert not advance.ready
    _, outcomes = eng.episode_context("A")
    assert outcomes[0].value == "r0"  # the first recorded outcome stands


# --------------------------------------------------------------------------- #
# Spawn members: admit-and-close, source order, re-drive idempotent
# --------------------------------------------------------------------------- #


def test_three_same_region_spawns_admit_three_children_then_one_seal() -> None:
    eng = _spawn_engine()
    advance = eng.route_facade_turn_group(
        "A", _group(_spawn_member(0), _spawn_member(1), _spawn_member(2))
    )
    children = [t for t in advance.ready if t.startswith("act-")]
    assert len(children) == 3  # exactly one child per spawn call, in source order
    # A spawn-only group settles at admission: it stages its acceptance vector and holds
    # no resume gate, so the fence is clear for another group next turn.
    assert eng.work_item("A").pending_outcome_group == "A:0"
    assert eng.has_open_facade_group("A") is False
    # A re-drive of the same group creates no additional child.
    redrive = eng.route_facade_turn_group(
        "A", _group(_spawn_member(0), _spawn_member(1), _spawn_member(2))
    )
    assert [t for t in redrive.ready if t.startswith("act-")] == []
    # The region is still open until an explicit seal closes child initiation.
    act = eng.work_item("A").activation_id
    region_scope = eng.region_scope_for(act, "worker")
    assert eng.capability(region_scope, ProgressAxis.CHILD_INIT).status.value == "open"
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="seal0",
            child_region_ref="worker",
        ),
    )
    assert (
        eng.capability(region_scope, ProgressAxis.CHILD_INIT).status.value == "sealed"
    )


def test_a_denied_sibling_does_not_prevent_valid_spawns() -> None:
    eng = _spawn_engine()
    advance = eng.route_facade_turn_group(
        "A",
        _group(
            _spawn_member(0, region="worker"),
            _spawn_member(1, region="undeclared"),  # denied: no such declared region
            _spawn_member(2, region="worker"),
        ),
    )
    children = [t for t in advance.ready if t.startswith("act-")]
    assert len(children) == 2  # the two valid siblings still materialize
    _, outcomes = eng.episode_context("A")
    kinds = {o.call_correlation.rsplit(":", 1)[1]: o.kind.value for o in outcomes}
    assert kinds["1"] == "denied" and kinds["0"] == "result" and kinds["2"] == "result"


def test_a_per_turn_spawn_budget_overflow_is_a_typed_quota_outcome() -> None:
    eng = _engine(_spawning_agent(child=_leaf("child")))
    object.__setattr__(eng._budget, "max_spawns_per_turn", 2)
    _dispatch_agent(eng)
    advance = eng.route_facade_turn_group(
        "A", _group(_spawn_member(0), _spawn_member(1), _spawn_member(2))
    )
    children = [t for t in advance.ready if t.startswith("act-")]
    assert len(children) == 2  # the third overflows the per-turn budget, no child
    _, outcomes = eng.episode_context("A")
    assert ToolOutcomeStatus.QUOTA.value in (outcomes[2].value or "")


# --------------------------------------------------------------------------- #
# Mixed group: only the search holds the resume gate
# --------------------------------------------------------------------------- #


def _mixed_engine():
    ref, region_ops, edge = _region("worker", "child", invoke=("model",), delegate=())
    agent = _agent(
        "A", regions=(ref,), invoke=("search/v1", "model"), delegate=("model",)
    )
    eng = _engine(
        _bundle([agent, _leaf("child"), *region_ops], [edge], (_decl("out:A", "A"),)),
        granted=_MIXED_GRANT,
    )
    _dispatch_agent(eng)
    return eng


def test_only_the_search_member_holds_a_mixed_group_resume_gate() -> None:
    eng = _mixed_engine()
    advance = eng.route_facade_turn_group(
        "A", _group(_spawn_member(0), _search_member(1))
    )
    # The spawn child materializes immediately; the lane still suspends on the search.
    assert [t for t in advance.ready if t.startswith("act-")] == advance.ready
    assert len(advance.ready) == 1
    assert eng.work_item("A").status is WorkItemStatus.BLOCKED
    assert eng.has_open_facade_group("A") is True
    # Settling the one search member releases the gate exactly once.
    resumed = eng.settle_boundary_outcome("A", "A:0:1", value="hit")
    assert resumed.ready == ["A"]
    assert eng.has_open_facade_group("A") is False
    _, outcomes = eng.episode_context("A")
    # The next turn sees the spawn acceptance ack and the search outcome, in order.
    assert len(outcomes) == 2
    assert ToolOutcomeStatus.SUCCESS.value in (outcomes[0].value or "")
    assert outcomes[1].value == "hit"


# --------------------------------------------------------------------------- #
# Runtime: parallel cap overflow settles as quota
# --------------------------------------------------------------------------- #


def _runtime(max_parallel: int) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(web_search=WebSearchConfig(max_parallel=max_parallel)),
        Path(tempfile.gettempdir()),
        logging.getLogger("group-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def test_an_oversized_search_group_settles_overflow_as_quota() -> None:
    async def run() -> None:
        runtime = _runtime(max_parallel=2)
        dispatched: list[ToolInvocationEnvelope] = []
        runtime.set_tool_broker(dispatched.append)
        workflow_id, ids = await _register(runtime, _SEARCH_WF)
        task = ids["searcher"]
        engine = runtime.orchestration_engine(workflow_id)
        assert engine is not None
        engine.on_dispatched(task, "w1")
        members = tuple(
            FacadeCallMember(
                ordinal=i,
                kind=BoundaryEventKind.INVOCATION,
                completion_mode=FacadeCompletionMode.AWAIT_OUTCOME,
                call_correlation=f"{task}:0:{i}",
                harness_call_id=f"c{i}",
                tool_name="web_search",
                interface_or_region="search/v1",
                request_payload=f'{{"query": "q{i}"}}',
            )
            for i in range(4)
        )
        group = FacadeTurnGroup(
            group_id=f"{task}:0", activation_id=task, turn_id="0", members=members
        )
        runtime.originate_facade_turn_group(task, group)
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
        # the group completes with a full four-outcome vector — no 500, no truncation.
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


def test_native_shell_is_permitted_without_network_egress() -> None:
    # The codex sandbox permits native shell but denies it network, so a native curl
    # cannot bypass the mediated search/v1 facade — the only egress. Guarded so the test
    # does not couple to the optional runtime-harness-codex dependency.
    import pytest

    pytest.importorskip("openai_codex")
    from worker.executors.harness.codex_transport import CodexTransportConfig

    cfg = CodexTransportConfig(
        base_url="http://gw",
        model="m",
        codex_home=Path(tempfile.gettempdir()) / "ch",
        initial_input="t",
        task_id="tsk-1",
    )
    overrides = cfg.to_codex_config().config_overrides
    assert 'sandbox_mode="workspace-write"' in overrides  # shell permitted
    assert "sandbox_workspace_write.network_access=false" in overrides  # no egress
    assert "tools.web_search=false" in overrides  # native web search hidden
