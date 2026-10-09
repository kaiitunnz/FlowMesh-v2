"""Which content a task is bound to read, and the result it is bound to."""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from shared.content import ContentReference
from shared.tasks.result_binding import (
    BindingKind,
    ResultBinding,
    ResultElementRef,
    ResultMember,
    ResultValueRef,
)

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


@dataclass(frozen=True)
class ScopedInput:
    """One value a task reads through an incoming edge: what it binds to, and the
    task whose whole result it is, when a ``task_id`` read may name one."""

    binding: ResultBinding
    task_id: str | None = None


class UnreadableInput(ValueError):
    """An input carrying a value a task cannot read."""


def _settled_at(record: TaskRecord) -> str | None:
    return ts_to_iso(record.finished_ts) if record.finished_ts is not None else None


def element_of(value_ref: ValueRef) -> int | None:
    """The collection member a value reference selects, or None for the whole result."""
    return (
        int(value_ref.collection_key) if value_ref.collection_key is not None else None
    )


def _references(binding: ResultBinding) -> Iterator[ContentReference]:
    """Every stored object a binding reads, its members' included."""
    if binding.reference is not None:
        yield binding.reference
    for member in binding.members:
        if member.binding is not None:
            yield from _references(member.binding)


class ContentBindings:
    """Answers which content each task is bound to read and which result it is bound
    to, over the runtime's task, engine and dependency tables."""

    def __init__(
        self,
        tasks: dict[str, TaskRecord],
        engines: dict[str, OrchestrationEngine],
        original_deps: dict[str, set[str]],
        logger: logging.Logger,
    ) -> None:
        self._tasks = tasks
        self._engines = engines
        self._original_deps = original_deps
        self._logger = logger

    def consumes_locked(self, record: TaskRecord, reference: ContentReference) -> bool:
        """Whether a task is bound to exactly this object as one of its inputs."""
        if reference.authorization_scope != record.org_id:
            return False
        task_id = record.task_id
        resolution = self.input_resolution_locked(task_id)
        if resolution is not None and resolution.reference == reference:
            return True
        if self._upstream_result_is_locked(record, reference):
            return True
        if self._scoped_input_is_locked(task_id, reference):
            return True
        engine = self._engines.get(record.workflow_id)
        if engine is None:
            return False
        if self._frozen_input_is_locked(engine, task_id, reference):
            return True
        _, outcomes = engine.episode_context(task_id)
        return any(
            outcome.outcome_ref is not None and outcome.outcome_ref.content == reference
            for outcome in outcomes
        )

    def upstream_task_ids_locked(self, task_id: str) -> set[str]:
        # A record stored with a dependency outside its workflow never reaches it: a
        # stage name resolves only within the workflow that declared it.
        record = self._tasks.get(task_id)
        workflow_id = record.workflow_id if record else None
        pending = list(self._original_deps.get(task_id, ()))
        visited: set[str] = set()
        while pending:
            dep_id = pending.pop()
            upstream = self._tasks.get(dep_id)
            if dep_id in visited or upstream is None:
                continue
            if upstream.workflow_id != workflow_id:
                continue
            visited.add(dep_id)
            pending.extend(self._original_deps.get(dep_id, ()))
        return visited

    def _upstream_result_is_locked(
        self, record: TaskRecord, reference: ContentReference
    ) -> bool:
        """Whether a settled upstream of the task, or of one merged into its dispatch,
        is bound to exactly this result."""
        for member_id in (record.task_id, *(record.merged_children or [])):
            member = self._tasks.get(member_id)
            if member is None:
                continue
            for dep_id in self.upstream_task_ids_locked(member_id):
                upstream = self._tasks.get(dep_id)
                if (
                    upstream is None
                    or upstream.workflow_id != member.workflow_id
                    or upstream.status != TaskStatus.DONE
                ):
                    continue
                binding = self.result_binding_locked(dep_id)
                if binding is not None and binding.reference == reference:
                    return True
        return False

    def _scoped_input_is_locked(
        self, task_id: str, reference: ContentReference
    ) -> bool:
        """Whether a value the task reads through an incoming edge reads exactly this
        object."""
        try:
            inputs = self.scoped_inputs_locked(task_id)
        except UnreadableInput:
            return False
        return any(
            reference in _references(entry.binding) for entry in (inputs or {}).values()
        )

    def _frozen_input_is_locked(
        self, engine: OrchestrationEngine, task_id: str, reference: ContentReference
    ) -> bool:
        """Whether an accepted input of the task, or its fan-out element, is frozen
        to exactly this producer result."""
        child_input = engine.child_input(task_id)
        if child_input is not None and child_input.content == reference:
            return True
        return any(
            (source := self.member_source_locked(member.value_ref)) is not None
            and source.reference == reference
            for accepted in engine.accepted_inputs_for_task(task_id)
            for member in accepted.members
        )

    def input_element_locked(self, task_id: str) -> _InputElement | None:
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        if engine is None:
            return None
        child_input = engine.child_input(task_id)
        if (
            child_input is None
            or child_input.content is None
            or child_input.legacy_task_id is None
            or (child_input.collection_key is None and not child_input.projection)
        ):
            return None
        return _InputElement(
            child_input.legacy_task_id,
            ResultElementRef(
                reference=child_input.content,
                element=(
                    int(child_input.collection_key)
                    if child_input.collection_key is not None
                    else None
                ),
                path=child_input.projection,
            ),
        )

    def scoped_inputs_locked(self, task_id: str) -> dict[str, ScopedInput] | None:
        """The values a task reads through its incoming edges, by name; None for a
        root task fed only by tasks, which reads their results by their names.

        Raises ``UnreadableInput`` for a value no binding can carry.
        """
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        inputs = engine.task_inputs(task_id) if engine else None
        if inputs is None:
            return None
        return {
            entry.name: ScopedInput(
                self._value_binding_locked(entry.value), entry.task_id
            )
            for entry in inputs
        }

    def _value_binding_locked(self, value: ValueRef) -> ResultBinding:
        """The binding a worker reads one input value through."""
        match value.kind:
            case "legacy_task_result":
                result = (
                    self.result_binding_locked(value.legacy_task_id)
                    if value.legacy_task_id is not None
                    else None
                )
                reference = value.content or (result.reference if result else None)
                selects = value.collection_key is not None or bool(value.projection)
                if reference is None and (result is None or result.skip is None):
                    if not selects:
                        # Named with nothing bound, so the task's identity still
                        # reaches its reader.
                        return ResultBinding(task_id=value.legacy_task_id)
                    raise UnreadableInput(
                        f"task {value.legacy_task_id} has no bound result to read"
                    )
                return ResultBinding(
                    task_id=value.legacy_task_id,
                    reference=reference,
                    skip=(
                        None if reference is not None or result is None else result.skip
                    ),
                    settled_at=result.settled_at if result is not None else None,
                    element=element_of(value),
                    path=value.projection,
                )
            case "aggregate" | "bundle":
                return ResultBinding(
                    kind=(
                        BindingKind.MEMBERS
                        if value.kind == "aggregate"
                        else BindingKind.BUNDLE
                    ),
                    members=tuple(
                        ResultMember(
                            key=member.key,
                            outcome=member.outcome.value,
                            binding=(
                                self._value_binding_locked(member.value_ref)
                                if member.value_ref is not None
                                and member.outcome is PublicationOutcome.SUCCESS
                                else None
                            ),
                        )
                        for member in value.members
                    ),
                    path=value.projection,
                )
            case "inline":
                return ResultBinding(
                    kind=BindingKind.LITERAL,
                    literal=value.literal,
                    path=value.projection,
                )
            case "empty":
                return ResultBinding(kind=BindingKind.EMPTY)
        raise UnreadableInput(f"a {value.kind} value is not readable as a task input")

    def input_resolution_locked(self, task_id: str) -> InputResolution | None:
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        return engine.input_resolution(task_id) if engine else None

    def result_binding_locked(self, task_id: str) -> ResultBinding | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        if (engine := self._engines.get(record.workflow_id)) is None:
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

    def settled_unbound_locked(self, task_id: str) -> bool:
        """Whether a task settled successfully with no result bound to read."""
        record = self._tasks.get(task_id)
        if record is None or record.status != TaskStatus.DONE:
            return False
        if record.result_skip is not None:
            return False
        engine = self._engines.get(record.workflow_id)
        if engine is None:
            return record.result_reference is None
        settled = engine.legacy_task_value(task_id)
        return settled is not None and settled[0] is PublicationOutcome.SUCCESS

    def bind_result_locked(
        self,
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
        record.result_reference = self.accepted_reference(record, reference)

    def accepted_reference(
        self, record: TaskRecord, reference: ContentReference | None
    ) -> ContentReference | None:
        """A reported result reference, when it lies in the scope control gave the task.

        Control reads a bound result under its own store access, so a reference naming
        another scope's object is refused here rather than served to this task's owner.
        """
        if reference is None or reference.authorization_scope == record.org_id:
            return reference
        self._logger.error(
            "Task %s reported a result in scope %s outside its own %s; not binding it",
            record.task_id,
            reference.authorization_scope,
            record.org_id,
        )
        return None

    def member_source_locked(self, value_ref: ValueRef | None) -> ResultValueRef | None:
        """The stored result an input member reads, when a producer supplies it."""
        if value_ref is None or value_ref.kind != "legacy_task_result":
            return None
        reference = value_ref.content
        if reference is None and value_ref.legacy_task_id:
            binding = self.result_binding_locked(value_ref.legacy_task_id)
            reference = binding.reference if binding is not None else None
        if reference is None:
            return None
        return ResultValueRef(reference=reference, element=element_of(value_ref))

    def consumed_inputs_locked(
        self, record: TaskRecord, references: Sequence[ContentReference]
    ) -> tuple[ContentReference, ...]:
        """The named objects the task consumes, each once."""
        return tuple(
            reference
            for reference in dict.fromkeys(references)
            if self.consumes_locked(record, reference)
        )
