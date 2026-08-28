"""Durable orchestration ledger (`DS`) over the acyclic compatibility plan."""

import logging
import threading
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.orchestration import (
    InvocationState,
    OrchestrationEngine,
    PublicationOutcome,
    RecoveryDisposition,
)
from server.orchestration.outcomes import (
    AdmissionError,
    check_admissible,
    classify_recovery,
    is_replayable,
    next_on_uncertain,
)
from server.orchestration.state import LedgerSnapshot
from server.registries.workflow import PersistedTask, WorkflowSched
from server.task.models import TaskStatus
from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from server.task.v2 import FrontendWorkflowSource, compile_bundle
from server.task.v2.representations.operators import (
    EffectClass,
    EffectReplayContract,
)

# --------------------------------------------------------------------------- #
# Durable store double
# --------------------------------------------------------------------------- #


class FakeRegistry:
    """In-memory registry that round-trips durable state through model JSON."""

    def __init__(self) -> None:
        self.task_blobs: dict[str, str] = {}
        self.sched: dict[str, str] = {}
        self.workflow_task_ids: dict[str, list[str]] = {}
        self.v2_blobs: dict[str, str] = {}
        self.ledger_blobs: dict[str, str] = {}

    async def register_workflow_async(
        self, workflow_id: str, tasks: list[Any], v2: Any = None
    ) -> None:
        self.workflow_task_ids[workflow_id] = [t.task_id for t in tasks]
        if v2 is not None:
            self.v2_blobs[workflow_id] = v2.model_dump_json()

    async def get_workflow_ids_async(self) -> set[str]:
        return set(self.workflow_task_ids)

    def get_workflow_record(self, workflow_id: str) -> Any:
        ids = self.workflow_task_ids.get(workflow_id)
        return SimpleNamespace(task_ids=list(ids)) if ids is not None else None

    async def get_workflow_record_async(self, workflow_id: str) -> Any:
        return self.get_workflow_record(workflow_id)

    async def save_task_states_async(self, items: list[PersistedTask]) -> None:
        for item in items:
            self.task_blobs[item.record.task_id] = item.model_dump_json()

    def load_task_states(self, *task_ids: str) -> list[PersistedTask | None]:
        return [
            (
                PersistedTask.model_validate_json(blob)
                if (blob := self.task_blobs.get(t))
                else None
            )
            for t in task_ids
        ]

    async def load_task_states_async(
        self, *task_ids: str
    ) -> list[PersistedTask | None]:
        return self.load_task_states(*task_ids)

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        self.sched[workflow_id] = WorkflowSched(
            in_epoch_order=in_epoch_order, epoch_frontier=frontier
        ).model_dump_json()

    async def load_workflow_sched_async(self, workflow_id: str) -> WorkflowSched | None:
        blob = self.sched.get(workflow_id)
        return WorkflowSched.model_validate_json(blob) if blob else None

    async def get_v2_workflow_async(self, workflow_id: str) -> Any:
        from server.task.v2 import PersistedV2Workflow

        blob = self.v2_blobs.get(workflow_id)
        return PersistedV2Workflow.model_validate_json(blob) if blob else None

    def save_ledger_snapshot(self, workflow_id: str, snapshot: LedgerSnapshot) -> None:
        self.ledger_blobs[workflow_id] = snapshot.model_dump_json()

    async def save_ledger_snapshot_async(
        self, workflow_id: str, snapshot: LedgerSnapshot
    ) -> None:
        self.save_ledger_snapshot(workflow_id, snapshot)

    async def load_ledger_snapshot_async(
        self, workflow_id: str
    ) -> LedgerSnapshot | None:
        blob = self.ledger_blobs.get(workflow_id)
        return LedgerSnapshot.model_validate_json(blob) if blob else None

    def commit_transition(
        self,
        workflow_id: str,
        *,
        records: Sequence[PersistedTask] = (),
        dispatched: Sequence[str] = (),
        pending: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
    ) -> None:
        for item in records:
            self.task_blobs[item.record.task_id] = item.model_dump_json()
        if sched is not None:
            self.sched[workflow_id] = sched.model_dump_json()


class _WorkerRegistryStub:
    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(id=worker_id, node_id="nde-1")

    def publish_interrupt(self, *args: Any) -> int:
        return 0


def _runtime(registry: FakeRegistry) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        logging.getLogger("v2-test"),
    )


def _worker(worker_id: str = "wkr-1") -> Any:
    return SimpleNamespace(id=worker_id, node_id="nde-1")


async def _register(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    workflow_id, results = await runtime.register(
        "owner", "org", payload, format="native"
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}


def _bundle(text: str, workflow_id: str = "wfl-x") -> Any:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    return compile_bundle(workflow_id, parsed, source)


LINEAR = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: linear}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
      - name: c
        dependsOn: [b]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""

DIAMOND = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: diamond}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
      - name: c
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [z]}}
      - name: d
        dependsOn: [b, c]
        spec: {taskType: echo, data: {type: list, items: [w]}}
"""


def _drain(runtime: TaskRuntime, worker_id: str = "wkr-1") -> list[str]:
    """Dispatch and complete every ready task until the queue drains; returns order."""
    stop = threading.Event()
    order: list[str] = []
    while runtime.ready_queue_length() > 0:
        task_id = runtime.next_ready(stop, timeout=0.01)
        if task_id is None:
            break
        order.append(task_id)
        runtime.mark_dispatched(task_id, cast(Any, _worker(worker_id)))
        runtime.mark_started(task_id, worker_id, {}, "2026-06-01T00:00:00Z")
        runtime.mark_succeeded(task_id, worker_id, {}, "2026-06-01T00:00:00Z")
    return order


# --------------------------------------------------------------------------- #
# Outcome model (pure)
# --------------------------------------------------------------------------- #


def test_is_replayable_by_effect_class() -> None:
    assert is_replayable(EffectClass.PURE, None) is True
    assert (
        is_replayable(
            EffectClass.EXTERNAL_EFFECT, EffectReplayContract.REPLAYABLE_DEDUP
        )
        is True
    )
    assert (
        is_replayable(
            EffectClass.EXTERNAL_EFFECT, EffectReplayContract.AMBIGUITY_TERMINAL
        )
        is False
    )
    assert is_replayable(EffectClass.EXTERNAL_EFFECT, None) is False


def test_uncertain_non_replayable_is_ambiguity_terminal() -> None:
    # A replayable invocation may reissue; a non-replayable one is ambiguity-terminal
    # and is never silently retried or reported as success.
    assert (
        next_on_uncertain(InvocationState.ACKNOWLEDGED, replayable=True)
        is InvocationState.UNCERTAIN
    )
    assert (
        next_on_uncertain(InvocationState.ACKNOWLEDGED, replayable=False)
        is InvocationState.AMBIGUITY_TERMINAL
    )
    # A terminal receipt is never regressed by a late uncertainty.
    assert (
        next_on_uncertain(InvocationState.TERMINAL, replayable=True)
        is InvocationState.TERMINAL
    )


def test_admission_rejects_residency_and_private_state() -> None:
    with pytest.raises(AdmissionError):
        check_admissible("op", EffectClass.PRIVATE_STATE, None, False)
    with pytest.raises(AdmissionError):
        check_admissible("op", EffectClass.PURE, None, residency_only=True)
    check_admissible("op", EffectClass.PURE, None, False)  # effect-free is admissible
    # An external effect of any replay contract is admitted; its uncertainty is handled
    # by the FSM without inferring success.
    check_admissible(
        "op",
        EffectClass.EXTERNAL_EFFECT,
        EffectReplayContract.AMBIGUITY_TERMINAL,
        False,
    )
    check_admissible(
        "op", EffectClass.EXTERNAL_EFFECT, EffectReplayContract.COMPENSABLE, False
    )


def test_classify_recovery_pure_deterministic_pinned_recomputes() -> None:
    from server.task.v2.compiler.bindings import leaf_profile
    from shared.tasks import TaskType

    assert (
        classify_recovery(leaf_profile(TaskType.ECHO)) is RecoveryDisposition.RECOMPUTE
    )
    # A sampled op with a recorded outcome must be restored, never recomputed.
    assert (
        classify_recovery(leaf_profile(TaskType.INFERENCE))
        is RecoveryDisposition.RESTORE
    )


# --------------------------------------------------------------------------- #
# Engine construction and admission
# --------------------------------------------------------------------------- #


def test_effectful_plan_builds_and_is_admitted() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: eff}
spec:
  graph:
    nodes:
      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
"""
    bundle = _bundle(text)
    # An external effect (ambiguity-terminal by default) builds; its uncertainty is
    # handled by the FSM rather than rejected at admission.
    led = OrchestrationEngine.build("wfl-x", "owner", "org", bundle)
    caller = bundle.template.operators[0].operator_id
    assert led.work_item(caller) is not None


def test_identity_hierarchy_is_one_per_operator() -> None:
    led = OrchestrationEngine.build("wfl-x", "owner", "org", _bundle(DIAMOND))
    snap = led.to_snapshot()
    # activation -> work item -> continuation, one each per result-owning operator.
    assert len(snap.activations) == 4
    assert len(snap.work_items) == 4
    assert len(snap.continuations) == 4
    assert len(snap.result_slots) == 4
    # No physical attempt exists before a work item is dispatched (incremental).
    assert snap.attempts == []


# --------------------------------------------------------------------------- #
# Runtime integration: a fixed DAG runs via the v2 orchestration path
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_linear_dag_runs_via_v2_path_with_same_dependency_semantics() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    assert runtime.is_v2_workflow(workflow_id)

    order = _drain(runtime)
    # Dependency order preserved: a before b before c.
    assert order == [ids["a"], ids["b"], ids["c"]]

    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    for name in ("a", "b", "c"):
        pub = runtime.resolve_v2_legacy_result(workflow_id, ids[name])
        assert pub is not None
        assert pub.outcome is PublicationOutcome.SUCCESS
        # The publication references the legacy task; the legacy adapter reads the
        # same on-disk value the v1 per-task result endpoint returns.
        assert pub.value_ref is not None
        assert pub.value_ref.legacy_task_id == ids[name]

    # The internal logical-output query resolves by declared output id too.
    by_output = runtime.resolve_v2_output(workflow_id, f"legacy:{ids['a']}")
    assert by_output is not None and by_output.outcome is PublicationOutcome.SUCCESS

    # The contract-relevant trace records the semantic seams for inspection.
    kinds = {kind for kind, _ in engine.contract_trace()}
    assert {
        "work_item_ready",
        "attempt_issued",
        "invocation_acknowledged",
        "effect_receipt",
        "result_published",
        "record_delivered",
    } <= kinds


@pytest.mark.anyio
async def test_conditional_skip_publishes_explicit_empty() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a, b = ids["a"], ids["b"]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    # A conditional skip settles the declared output as explicit-empty, yet still
    # releases the successor (matching the v1 skip-as-success behavior).
    runtime.mark_succeeded(a, None, {}, "2026-06-01T00:00:00Z", empty=True)
    pub = runtime.resolve_v2_legacy_result(workflow_id, a)
    assert pub is not None and pub.outcome is PublicationOutcome.EXPLICIT_EMPTY
    assert pub.value_ref is not None and pub.value_ref.kind == "empty"
    assert runtime.next_ready(stop, timeout=0.01) == b


@pytest.mark.anyio
async def test_diamond_dag_joins_on_both_predecessors() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, DIAMOND)

    stop = threading.Event()
    # a is the only root.
    assert runtime.next_ready(stop, timeout=0.01) == ids["a"]
    runtime.mark_dispatched(ids["a"], cast(Any, _worker()))
    runtime.mark_succeeded(ids["a"], "wkr-1", {}, "2026-06-01T00:00:00Z")

    # a frees b and c, but not d.
    assert runtime.ready_queue_length() == 2
    freed = {
        runtime.next_ready(stop, timeout=0.01),
        runtime.next_ready(stop, timeout=0.01),
    }
    assert freed == {ids["b"], ids["c"]}
    for name in ("b", "c"):
        runtime.mark_dispatched(ids[name], cast(Any, _worker()))
    # d stays blocked until BOTH b and c settle.
    runtime.mark_succeeded(ids["b"], "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert runtime.ready_queue_length() == 0
    runtime.mark_succeeded(ids["c"], "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert runtime.ready_queue_length() == 1
    assert runtime.next_ready(stop, timeout=0.01) == ids["d"]


@pytest.mark.anyio
async def test_scheduler_placement_does_not_change_readiness() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    stop = threading.Event()

    assert runtime.next_ready(stop, timeout=0.01) == ids["a"]
    # Placing a on any worker never readies b; only a's settlement does.
    runtime.mark_dispatched(ids["a"], cast(Any, _worker("wkr-A")))
    assert runtime.ready_queue_length() == 0
    runtime.mark_started(ids["a"], "wkr-A", {}, "2026-06-01T00:00:00Z")
    assert runtime.ready_queue_length() == 0
    runtime.mark_succeeded(ids["a"], "wkr-A", {}, "2026-06-01T00:00:00Z")
    assert runtime.next_ready(stop, timeout=0.01) == ids["b"]


# --------------------------------------------------------------------------- #
# Retries create new attempts, not new work items
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_retry_creates_new_attempt_same_work_item_and_invocation() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a = ids["a"]
    stop = threading.Event()

    assert runtime.next_ready(stop, timeout=0.01) == a
    runtime.mark_dispatched(a, cast(Any, _worker()))
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    wi = engine.work_item(a)
    assert wi is not None
    work_item_id = wi.work_item_id
    invocation_id = engine.invocation_for_task(a).invocation_id  # type: ignore[union-attr]

    # A retryable failure requeues the SAME task as a fresh attempt.
    runtime.mark_pending(a, increment_retry=True)
    runtime.requeue(a, front=True)
    assert runtime.next_ready(stop, timeout=0.01) == a
    runtime.mark_dispatched(a, cast(Any, _worker("wkr-2")))
    runtime.mark_succeeded(a, "wkr-2", {}, "2026-06-01T00:00:00Z")

    snap = engine.to_snapshot()
    same_wi = [w for w in snap.work_items if w.legacy_task_id == a]
    assert len(same_wi) == 1  # no new logical work item
    assert same_wi[0].work_item_id == work_item_id
    attempts = [att for att in snap.attempts if att.work_item_id == work_item_id]
    assert len(attempts) == 2  # retry created a second physical attempt
    # The stable invocation identity is reused across attempts.
    assert engine.invocation_for_task(a).invocation_id == invocation_id  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_declared_output_one_publication_across_retries() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a = ids["a"]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    runtime.mark_pending(a, increment_retry=True)
    runtime.requeue(a, front=True)
    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker("wkr-2")))
    runtime.mark_succeeded(a, "wkr-2", {}, "2026-06-01T00:00:00Z")

    snap = runtime.orchestration_engine(workflow_id).to_snapshot()  # type: ignore[union-attr]
    pubs = [p for p in snap.result_publications if p.output_id == f"legacy:{a}"]
    assert len(pubs) == 1
    assert pubs[0].outcome is PublicationOutcome.SUCCESS


# --------------------------------------------------------------------------- #
# Failure cascade
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_terminal_failure_cascades_to_dependents() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a, b, c = ids["a"], ids["b"], ids["c"]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    impacted, _, _ = runtime.mark_failed(
        a, "wkr-1", {}, "2026-06-01T00:00:00Z", error="boom"
    )

    # a's failure cascades declared-failure obligations to b and c.
    assert {tid for tid, _ in impacted} == {b, c}
    for name in (a, b, c):
        pub = runtime.resolve_v2_legacy_result(workflow_id, name)
        assert pub is not None and pub.outcome is PublicationOutcome.DECLARED_FAILURE
    assert runtime.get_record(b).status == TaskStatus.FAILED  # type: ignore[union-attr]
    assert runtime.ready_queue_length() == 0


# --------------------------------------------------------------------------- #
# Uncertain / lost acknowledgement
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_lost_ack_replayable_retried_through_stable_invocation() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a = ids["a"]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    engine = runtime.orchestration_engine(workflow_id)
    invocation_id = engine.invocation_for_task(a).invocation_id  # type: ignore[union-attr]

    # A lost acknowledgement for a replayable (pure) invocation: reissued through the
    # same invocation identity, not reported as success.
    advance = runtime.mark_v2_uncertain(a)
    assert advance.retry == [a]
    assert runtime.resolve_v2_legacy_result(workflow_id, a) is None  # not published
    assert runtime.next_ready(stop, timeout=0.01) == a
    runtime.mark_dispatched(a, cast(Any, _worker("wkr-2")))
    assert engine.invocation_for_task(a).invocation_id == invocation_id  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_worker_loss_recovery_routes_through_uncertainty_fsm() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a = ids["a"]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker("wkr-dead")))
    engine = runtime.orchestration_engine(workflow_id)
    invocation_id = engine.invocation_for_task(a).invocation_id  # type: ignore[union-attr]

    # The worker departs: recovery resolves the in-flight v2 work item through the
    # uncertainty FSM itself, so it is not returned for a synthetic failure.
    assert runtime.recover_tasks_for_worker("wkr-dead") == []
    assert runtime.get_record(a).status == TaskStatus.PENDING  # type: ignore[union-attr]
    assert runtime.next_ready(stop, timeout=0.01) == a
    runtime.mark_dispatched(a, cast(Any, _worker("wkr-2")))
    assert engine.invocation_for_task(a).invocation_id == invocation_id  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Restart / rehydration
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_rehydration_heals_when_ledger_snapshot_lags_terminal_records() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a, b, c = ids["a"], ids["b"], ids["c"]
    # Snapshot the ledger as it stood at submission, before any settlement.
    stale_ledger = registry.ledger_blobs[workflow_id]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z", error="boom")
    # Simulate a crash after the terminal task records committed but before the ledger
    # snapshot: the ledger lags durable task state (never ahead, per the write order).
    registry.ledger_blobs[workflow_id] = stale_ledger

    restored = _runtime(registry)
    await restored.rehydrate()
    # Rehydration reconciles the lagging ledger from terminal task facts: all three
    # settle declared-failure exactly once, and nothing is left ready.
    snap = restored.orchestration_engine(workflow_id).to_snapshot()  # type: ignore[union-attr]
    for name in (a, b, c):
        assert restored.get_record(name).status == TaskStatus.FAILED  # type: ignore[union-attr]
        pub = restored.resolve_v2_legacy_result(workflow_id, name)
        assert pub is not None and pub.outcome is PublicationOutcome.DECLARED_FAILURE
        matched = [
            p for p in snap.result_publications if p.output_id == f"legacy:{name}"
        ]
        assert len(matched) == 1
    assert restored.ready_queue_length() == 0


@pytest.mark.anyio
async def test_rehydration_readmits_task_orphaned_by_a_mid_retry_crash() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a, b = ids["a"], ids["b"]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    # Snapshot the ledger while a's work item is in flight (DISPATCHED).
    dispatched_ledger = registry.ledger_blobs[workflow_id]
    # A retry persists a's record PENDING and readies the ledger work item, but
    # simulate a crash before that ledger snapshot committed: the durable snapshot
    # still shows the work item DISPATCHED while the task record is PENDING.
    runtime.mark_pending(a, increment_retry=True)
    registry.ledger_blobs[workflow_id] = dispatched_ledger

    restored = _runtime(registry)
    await restored.rehydrate()
    # Rehydration re-derives readiness and re-admits a rather than orphaning it.
    assert restored.get_record(a).status == TaskStatus.PENDING  # type: ignore[union-attr]
    assert restored.ready_queue_length() == 1
    assert restored.next_ready(stop, timeout=0.01) == a
    # The workflow makes progress from there.
    restored.mark_dispatched(a, cast(Any, _worker("wkr-2")))
    restored.mark_succeeded(a, "wkr-2", {}, "2026-06-01T00:00:00Z")
    assert restored.next_ready(stop, timeout=0.01) == b


@pytest.mark.anyio
async def test_rehydration_preserves_publications_without_duplication() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a, b, c = ids["a"], ids["b"], ids["c"]
    stop = threading.Event()

    runtime.next_ready(stop, timeout=0.01)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    restored = _runtime(registry)
    assert await restored.rehydrate() == 1
    engine = restored.orchestration_engine(workflow_id)
    assert engine is not None

    # a's publication is restored exactly once; the settled attempt is not re-run.
    snap = engine.to_snapshot()
    assert (
        len([p for p in snap.result_publications if p.output_id == f"legacy:{a}"]) == 1
    )
    restored_pub = restored.resolve_v2_legacy_result(workflow_id, a)
    assert (
        restored_pub is not None and restored_pub.outcome is PublicationOutcome.SUCCESS
    )
    # b (a's dependent) is re-admitted as the sole ready work item; c stays blocked.
    assert restored.ready_queue_length() == 1
    assert restored.next_ready(stop, timeout=0.01) == b
    assert restored.ready_queue_length() == 0
    assert restored.get_record(c).status == TaskStatus.PENDING  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_rehydration_distinguishes_recompute_from_restore() -> None:
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: mixed}
spec:
  graph:
    nodes:
      - name: pure
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: sampled
        dependsOn: [pure]
        spec:
          taskType: inference
          inference: {model_name: m, prompt: hi}
"""
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, text)
    _drain(runtime)

    restored = _runtime(registry)
    await restored.rehydrate()
    # The pinned deterministic op recomputes; the sampled op restores its outcome.
    assert (
        restored.recovery_disposition(workflow_id, ids["pure"])
        is RecoveryDisposition.RECOMPUTE
    )
    assert (
        restored.recovery_disposition(workflow_id, ids["sampled"])
        is RecoveryDisposition.RESTORE
    )
    # The declared output obligation is preserved in both cases.
    for name in ("pure", "sampled"):
        assert restored.resolve_v2_legacy_result(workflow_id, ids[name]) is not None


# --------------------------------------------------------------------------- #
# Authority
# --------------------------------------------------------------------------- #


def test_authority_denied_settles_declared_failure() -> None:
    # A replayable external-effect op whose interface is outside the pinned grant is
    # denied before invocation and settles a durable declared-failure outcome.
    text = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: eff}
spec:
  graph:
    nodes:
      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
          v2: {recovery: replay_with_dedup}
"""
    bundle = _bundle(text)
    caller = bundle.template.operators[0].operator_id
    # Force the boundary to be replayable so the op is admissible, then deny its grant.
    boundaries = tuple(
        b.model_copy(update={"replay_contract": EffectReplayContract.REPLAYABLE_DEDUP})
        for b in bundle.template.effect_boundaries
    )
    template = bundle.template.model_copy(update={"effect_boundaries": boundaries})
    bundle = bundle.model_copy(update={"template": template})

    led = OrchestrationEngine.build(
        "wfl-x", "owner", "org", bundle, granted_interfaces=frozenset()
    )
    snap = led.to_snapshot()
    denied = [d for d in snap.authority_decisions if d.kind.value == "denied"]
    assert denied and denied[0].work_item_id
    pub = led.resolve_legacy_task(caller)
    assert pub is not None and pub.outcome is PublicationOutcome.DECLARED_FAILURE
