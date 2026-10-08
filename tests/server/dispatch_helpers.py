"""Record a task's dispatch on a runtime the way the dispatcher does."""

from types import SimpleNamespace
from typing import cast

from server.orchestration import Advance, OrchestrationEngine
from server.registries.worker import Worker
from server.task.models import PublishGate
from server.task.runtime import TaskRuntime
from server.task.v2.representations.operators import ResolvedEmbodiment


def record_dispatch(
    runtime: TaskRuntime,
    task_id: str,
    worker: str | Worker = "wkr-1",
    dispatch_id: str | None = None,
    input_preparation: bool = False,
    expect: PublishGate = PublishGate.PUBLISH,
) -> bool:
    """Publish and record a dispatch, asserting the publish gate answers ``expect``."""
    if isinstance(worker, str):
        worker = cast(Worker, SimpleNamespace(id=worker, node_id="nde-1"))
    gate = runtime.begin_publish(
        task_id, worker, dispatch_id, input_preparation=input_preparation
    )
    assert gate is expect, gate
    return runtime.mark_dispatched(task_id)


def resolved_embodiment(
    runtime: TaskRuntime, task_id: str
) -> ResolvedEmbodiment | None:
    """The embodiment a menu node's task is bound to, or None while it is unbound."""
    record = runtime.get_record(task_id)
    engine = runtime.orchestration_engine(record.workflow_id) if record else None
    if engine is None:
        return None
    menu = engine.embodiment_menu(task_id)
    if menu is None or (selection := engine.embodiment_selection(task_id)) is None:
        return None
    candidate = menu.candidate(selection.alternative_id)
    if candidate is None:
        return None
    return ResolvedEmbodiment(
        alternative_id=candidate.alternative_id, kind=candidate.kind
    )


def stage_agent_inputs(
    runtime: TaskRuntime, workflow_id: str, engine: OrchestrationEngine
) -> None:
    """Stage an engine's agent inputs as the runtime does after an advance."""
    with runtime._lock:
        runtime._agent_inputs.stage_agent_inputs_locked(workflow_id, engine, Advance())
