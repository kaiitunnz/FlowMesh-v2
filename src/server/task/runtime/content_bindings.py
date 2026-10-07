"""Which content a task is bound to read, and the result it is bound to."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shared.content import ContentReference
from shared.tasks.result_binding import ResultBinding, ResultElementRef, ResultValueRef

from ...orchestration import (
    InputResolution,
    OrchestrationEngine,
    PublicationOutcome,
    ValueRef,
)
from ...utils.time import ts_to_iso
from ..models import TaskRecord, TaskStatus


@dataclass(frozen=True)
class _InputElement:
    """The producer element a leaf fan-out child runs on."""

    producer_task_id: str
    ref: ResultElementRef


def _settled_at(record: TaskRecord) -> str | None:
    return ts_to_iso(record.finished_ts) if record.finished_ts is not None else None


def element_of(value_ref: ValueRef) -> int | None:
    """The collection member a value reference selects, or None for the whole result."""
    return (
        int(value_ref.collection_key) if value_ref.collection_key is not None else None
    )


def consumes_locked(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    original_deps: dict[str, set[str]],
    record: TaskRecord,
    reference: ContentReference,
) -> bool:
    """Whether a task is bound to exactly this object as one of its inputs."""
    if reference.authorization_scope != record.org_id:
        return False
    task_id = record.task_id
    resolution = input_resolution_locked(tasks, engines, task_id)
    if resolution is not None and resolution.reference == reference:
        return True
    if upstream_result_is_locked(tasks, engines, original_deps, record, reference):
        return True
    engine = engines.get(record.workflow_id)
    if engine is None:
        return False
    if frozen_input_is_locked(tasks, engines, engine, task_id, reference):
        return True
    _, outcomes = engine.episode_context(task_id)
    return any(
        outcome.outcome_ref is not None and outcome.outcome_ref.content == reference
        for outcome in outcomes
    )


def upstream_task_ids_locked(
    tasks: dict[str, TaskRecord], original_deps: dict[str, set[str]], task_id: str
) -> set[str]:
    # A record stored with a dependency outside its workflow never reaches it: a
    # stage name resolves only within the workflow that declared it.
    record = tasks.get(task_id)
    workflow_id = record.workflow_id if record else None
    pending = list(original_deps.get(task_id, ()))
    visited: set[str] = set()
    while pending:
        dep_id = pending.pop()
        upstream = tasks.get(dep_id)
        if dep_id in visited or upstream is None:
            continue
        if upstream.workflow_id != workflow_id:
            continue
        visited.add(dep_id)
        pending.extend(original_deps.get(dep_id, ()))
    return visited


def upstream_result_is_locked(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    original_deps: dict[str, set[str]],
    record: TaskRecord,
    reference: ContentReference,
) -> bool:
    """Whether a settled upstream of the task, or of one merged into its dispatch,
    is bound to exactly this result."""
    for member_id in (record.task_id, *(record.merged_children or [])):
        member = tasks.get(member_id)
        if member is None:
            continue
        for dep_id in upstream_task_ids_locked(tasks, original_deps, member_id):
            upstream = tasks.get(dep_id)
            if (
                upstream is None
                or upstream.workflow_id != member.workflow_id
                or upstream.status != TaskStatus.DONE
            ):
                continue
            binding = result_binding_locked(tasks, engines, dep_id)
            if binding is not None and binding.reference == reference:
                return True
    return False


def frozen_input_is_locked(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    engine: OrchestrationEngine,
    task_id: str,
    reference: ContentReference,
) -> bool:
    """Whether an accepted input of the task, or its fan-out element, is frozen
    to exactly this producer result."""
    child_input = engine.child_input(task_id)
    if child_input is not None and child_input.content == reference:
        return True
    return any(
        (source := member_source_locked(tasks, engines, member.value_ref)) is not None
        and source.reference == reference
        for accepted in engine.accepted_inputs_for_task(task_id)
        for member in accepted.members
    )


def input_element_locked(
    tasks: dict[str, TaskRecord], engines: dict[str, OrchestrationEngine], task_id: str
) -> _InputElement | None:
    record = tasks.get(task_id)
    engine = engines.get(record.workflow_id) if record else None
    if engine is None:
        return None
    child_input = engine.child_input(task_id)
    if (
        child_input is None
        or child_input.content is None
        or child_input.collection_key is None
        or child_input.legacy_task_id is None
    ):
        return None
    return _InputElement(
        child_input.legacy_task_id,
        ResultElementRef(
            reference=child_input.content, element=int(child_input.collection_key)
        ),
    )


def input_resolution_locked(
    tasks: dict[str, TaskRecord], engines: dict[str, OrchestrationEngine], task_id: str
) -> InputResolution | None:
    record = tasks.get(task_id)
    engine = engines.get(record.workflow_id) if record else None
    return engine.input_resolution(task_id) if engine else None


def result_binding_locked(
    tasks: dict[str, TaskRecord], engines: dict[str, OrchestrationEngine], task_id: str
) -> ResultBinding | None:
    record = tasks.get(task_id)
    if record is None:
        return None
    if (engine := engines.get(record.workflow_id)) is None:
        if record.result_reference is None and record.result_skip is None:
            return None
        return ResultBinding(
            task_id=task_id,
            reference=record.result_reference,
            skip=record.result_skip,
            settled_at=_settled_at(record),
        )
    settled = engine.legacy_task_value(task_id)
    if settled is None:
        return None
    outcome, value_ref = settled
    if outcome is PublicationOutcome.EXPLICIT_EMPTY:
        if record.result_skip is None:
            return None
        return ResultBinding(
            task_id=task_id, skip=record.result_skip, settled_at=_settled_at(record)
        )
    if (
        outcome is not PublicationOutcome.SUCCESS
        or value_ref is None
        or value_ref.content is None
    ):
        return None
    return ResultBinding(task_id=task_id, reference=value_ref.content)


def settled_unbound_locked(
    tasks: dict[str, TaskRecord], engines: dict[str, OrchestrationEngine], task_id: str
) -> bool:
    """Whether a task settled successfully with no result bound to read."""
    record = tasks.get(task_id)
    if record is None or record.status != TaskStatus.DONE:
        return False
    if record.result_skip is not None:
        return False
    engine = engines.get(record.workflow_id)
    if engine is None:
        return record.result_reference is None
    settled = engine.legacy_task_value(task_id)
    return settled is not None and settled[0] is PublicationOutcome.SUCCESS


def bind_result_locked(
    logger: logging.Logger,
    record: TaskRecord,
    reference: ContentReference | None,
    skip: dict[str, Any] | None,
) -> None:
    """Bind a settling task's result once; a later success never re-points it."""
    if record.result_reference is not None or record.result_skip is not None:
        return
    if skip is not None:
        record.result_skip = skip
        return
    record.result_reference = accepted_reference(logger, record, reference)


def accepted_reference(
    logger: logging.Logger, record: TaskRecord, reference: ContentReference | None
) -> ContentReference | None:
    """A reported result reference, when it lies in the scope control gave the task.

    Control reads a bound result under its own store access, so a reference naming
    another scope's object is refused here rather than served to this task's owner.
    """
    if reference is None or reference.authorization_scope == record.org_id:
        return reference
    logger.error(
        "Task %s reported a result in scope %s outside its own %s; not binding it",
        record.task_id,
        reference.authorization_scope,
        record.org_id,
    )
    return None


def member_source_locked(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    value_ref: ValueRef | None,
) -> ResultValueRef | None:
    """The stored result an input member reads, when a producer supplies it."""
    if value_ref is None or value_ref.kind != "legacy_task_result":
        return None
    reference = value_ref.content
    if reference is None and value_ref.legacy_task_id:
        binding = result_binding_locked(tasks, engines, value_ref.legacy_task_id)
        reference = binding.reference if binding is not None else None
    if reference is None:
        return None
    return ResultValueRef(reference=reference, element=element_of(value_ref))


def consumed_inputs_locked(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    original_deps: dict[str, set[str]],
    record: TaskRecord,
    references: Sequence[ContentReference],
) -> tuple[ContentReference, ...]:
    """The named objects the task consumes, each once."""
    return tuple(
        reference
        for reference in dict.fromkeys(references)
        if consumes_locked(tasks, engines, original_deps, record, reference)
    )
