"""Drive the orchestration engine over compiled branch, loop and child workflows."""

from typing import Any

from server.orchestration import OrchestrationEngine, ScopeBudget
from server.orchestration.engine.advance import Advance
from server.orchestration.state import WorkItemStatus
from server.task.parser import parse_workflow
from server.task.v2 import FrontendWorkflowSource, PersistedV2Workflow, compile_bundle
from server.task.v2.compiler.agent_binding import AgentBindingDefaults

ECHO = "{taskType: echo, data: {type: list, items: [x]}}"
_BINDINGS = AgentBindingDefaults(default_backend="codex")


def workflow(nodes: str, templates: str = "") -> str:
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: cf}}
spec:
  graph:
{templates}
    nodes:
{nodes}
"""


def compile_text(text: str) -> PersistedV2Workflow:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    return compile_bundle("wfl-cf", parsed, source, bindings=_BINDINGS)


class Driver:
    """Runs one compiled workflow's engine, tracking the tasks it makes ready."""

    def __init__(
        self,
        text: str,
        *,
        budget: ScopeBudget | None = None,
        granted: frozenset[str] | None = None,
    ) -> None:
        self.bundle = compile_text(text)
        self.engine = OrchestrationEngine.build(
            "wfl-cf",
            "owner",
            "org",
            self.bundle,
            budget=budget,
            granted_interfaces=granted,
        )
        self.names = {
            e.logical_ref: e.source_id for e in self.bundle.template.source_map
        }
        self.ops = {v: k for k, v in self.names.items()}
        # A template member is also reachable by its bare name when no other node
        # shares it.
        tails: dict[str, list[str]] = {}
        for source, op in self.ops.items():
            tails.setdefault(source.rpartition("/")[2], []).append(op)
        for tail, ops in tails.items():
            if len(ops) == 1:
                self.ops.setdefault(tail, ops[0])
        self.ready: list[str] = []
        self.skipped: list[str] = []
        self.failed: list[str] = []
        self.cancelled: list[str] = []
        self.ran: list[str] = []
        self.apply(self.engine.initial_advance())

    def restore(self) -> None:
        """Rebuild the engine from its own snapshot, as a restart does."""
        self.engine = OrchestrationEngine(self.engine.to_snapshot(), self.bundle)

    def apply(self, advance: Advance) -> Advance:
        self.ready.extend(t for t in advance.ready if t not in self.ready)
        self.skipped.extend(advance.skipped)
        self.failed.extend(advance.failed)
        self.cancelled.extend(advance.cancelled)
        return advance

    def operator(self, task_id: str) -> str:
        wi = self.engine.work_item(task_id)
        assert wi is not None
        return wi.operator_id

    def name(self, task_id: str) -> str:
        operator_id = self.operator(task_id)
        return self.names.get(operator_id, operator_id)

    def ready_named(self, name: str) -> list[str]:
        return [t for t in self.ready if self.operator(t) == self.ops[name]]

    def run(self, task_id: str, *, fail: bool = False) -> Advance:
        """Dispatch a ready task and settle it."""
        self.ready.remove(task_id)
        self.ran.append(self.name(task_id))
        self.engine.on_dispatched(task_id, "w1")
        if fail:
            return self.apply(self.engine.on_failed(task_id, "boom", retryable=False))
        return self.apply(self.engine.on_succeeded(task_id))

    def run_one(self, name: str, *, fail: bool = False) -> str:
        (task_id,) = self.ready_named(name)
        self.run(task_id, fail=fail)
        return task_id

    def select(self, value: Any, *, error: str | None = None) -> Advance:
        """Answer the one pending branch read with a selector value."""
        key, _ = self.engine.pending_branch_reads()[0]
        return self.apply(self.engine.accept_branch_selection(key, value, error=error))

    def status(self, name: str) -> WorkItemStatus:
        op = self.ops[name]
        wi = self.engine.work_item(op)
        assert wi is not None
        return wi.status
