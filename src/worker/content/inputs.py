"""Hydrating the upstream values a task consumes from the references it was dispatched.

Control names what a task reads — each upstream stage's settled result, the collection
element a fan-out child runs on, the producer results an agent's first-turn inputs are
frozen to — and this worker reads them through its content plane: its own cache, then an
authorized peer's copy, then the shared store. Every read is verified against its
reference. The values are installed exactly as the executor would have received them
inline, before anything validates or runs the task.

A store that cannot be reached is a pause, not a loss: the task reports its inputs
unavailable and control runs it again without spending an attempt. Content that is
missing, corrupt, out of the task's scope, or not a result envelope fails the task.
"""

import logging
import time
from typing import Any

from pydantic import ValidationError

from shared.content import ContentReference, ContentStoreError, ContentUnavailable
from shared.harness import AgentEpisodeDispatch, InputBinding
from shared.schemas.event import TaskFailureKind
from shared.schemas.result import ResultEnvelope
from shared.schemas.result.binding import (
    collection_element,
    skip_envelope_bytes,
    value_text,
)
from shared.tasks import MergedChildTaskStrict, TaskEnvelopeStrict
from shared.tasks.result_binding import ResultBinding
from shared.tasks.specs import TaskSpecStrictBase
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.json import normalize_numbers

from ..executors.base_executor import ExecutionError
from .access import ContentAccessDenied
from .plane import WorkerContentPlane

# A read that cannot reach the store retries briefly before the task reports it.
_READ_ATTEMPTS = 3
_READ_BACKOFF_SEC = 0.2


def input_unavailable(message: str) -> ExecutionError:
    return ExecutionError(
        message, retryable=True, failure_kind=TaskFailureKind.INPUT_UNAVAILABLE
    )


def input_unreadable(message: str) -> ExecutionError:
    return ExecutionError(f"input_unreadable: {message}", retryable=False)


class TaskInputHydrator:
    """Installs the values a task's dispatch names by reference."""

    def __init__(
        self,
        plane: WorkerContentPlane | None,
        logger: logging.Logger | None = None,
        *,
        backoff_sec: float = _READ_BACKOFF_SEC,
    ) -> None:
        self._plane = plane
        self._logger = logger or logging.getLogger("task-inputs")
        self._backoff_sec = backoff_sec

    def hydrate(self, msg: WorkerTaskMessage) -> None:
        """Install every value the message names by reference, in place.

        A message naming nothing by reference carries its values inline and is left as
        it is.
        """
        agent = msg.agent_episode
        if not (
            msg.upstream_results
            or msg.input_element is not None
            or any(child.upstream_results for child in msg.merged_children or [])
            or (agent is not None and _sourced_members(agent))
        ):
            return
        reader = _TaskReader(self, msg)
        envelopes: dict[str, bytes] = {}
        upstream: dict[str, ResultEnvelope] = {}
        for stage, binding in (msg.upstream_results or {}).items():
            envelopes[stage] = reader.envelope_bytes(binding)
            upstream[stage] = reader.envelope(binding)
        element: tuple[Any] | None = None
        if (ref := msg.input_element) is not None and ref.element is not None:
            source = reader.envelope(ResultBinding(task_id="", reference=ref.reference))
            try:
                element = (collection_element(source, ref.element),)
            except IndexError as exc:
                raise input_unreadable(str(exc)) from exc
        msg.task = _with_inputs(msg.task, upstream, element)
        if msg.merged_children:
            msg.merged_children = [
                self._hydrate_child(reader, child) for child in msg.merged_children
            ]
        if agent is not None and _sourced_members(agent):
            msg.agent_episode = _with_member_values(agent, reader)
        msg.record_hydration(envelopes, element)

    def _hydrate_child(
        self, reader: "_TaskReader", child: MergedChildTaskStrict
    ) -> MergedChildTaskStrict:
        if not child.upstream_results:
            return child
        upstream = {
            stage: reader.envelope(binding)
            for stage, binding in child.upstream_results.items()
        }
        hydrated = child.model_copy(
            update={"spec": _with_upstream_spec(child.spec, upstream)}
        )
        wire = hydrated.model_dump(mode="json", exclude_none=True, by_alias=True)
        return MergedChildTaskStrict.model_validate(normalize_numbers(wire))

    def read(self, task_id: str, scope: str, reference: ContentReference) -> bytes:
        """One verified object, retrying a store that is briefly away."""
        if self._plane is None:
            raise ExecutionError(
                f"task {task_id} reads its inputs by reference and this worker reaches "
                "no fabric content store",
                retryable=True,
            )
        if reference.authorization_scope != scope:
            raise input_unreadable(
                f"task {task_id} runs in scope {scope} and an input it names is in "
                f"{reference.authorization_scope}"
            )
        for attempt in range(_READ_ATTEMPTS):
            try:
                return self._plane.hydrate(task_id, reference)
            except (ContentUnavailable, ContentAccessDenied) as exc:
                if attempt + 1 == _READ_ATTEMPTS:
                    raise input_unavailable(
                        f"task {task_id} cannot reach input "
                        f"{reference.content_digest}: {exc}"
                    ) from exc
                time.sleep(self._backoff_sec)
            except ContentStoreError as exc:
                raise input_unreadable(
                    f"task {task_id} input {reference.content_digest}: {exc}"
                ) from exc
        raise AssertionError("unreachable")


class _TaskReader:
    """One task's reads, each object fetched and parsed at most once."""

    def __init__(self, hydrator: TaskInputHydrator, msg: WorkerTaskMessage) -> None:
        self._hydrator = hydrator
        self._task_id = msg.task_id
        self._scope = msg.content_scope
        self._bytes: dict[ContentReference, bytes] = {}
        self._envelopes: dict[ContentReference, ResultEnvelope] = {}

    def envelope_bytes(self, binding: ResultBinding) -> bytes:
        if binding.reference is None:
            if binding.skip is None:
                raise input_unreadable(f"task {binding.task_id} has no bound result")
            return skip_envelope_bytes(binding)
        reference = binding.reference
        if (data := self._bytes.get(reference)) is None:
            data = self._hydrator.read(self._task_id, self._scope, reference)
            self._bytes[reference] = data
        return data

    def envelope(self, binding: ResultBinding) -> ResultEnvelope:
        reference = binding.reference
        if (
            reference is not None
            and (cached := self._envelopes.get(reference)) is not None
        ):
            return cached
        data = self.envelope_bytes(binding)
        try:
            envelope = ResultEnvelope.model_validate_json(data)
        except ValidationError as exc:
            raise input_unreadable(
                f"the stored result of task {binding.task_id or '?'} is not a result "
                f"envelope: {exc}"
            ) from exc
        if reference is not None:
            self._envelopes[reference] = envelope
        return envelope


def _sourced_members(agent: AgentEpisodeDispatch) -> bool:
    return any(
        member.source is not None
        for binding in agent.input_bindings
        for member in binding.members
    )


def _with_member_values(
    agent: AgentEpisodeDispatch, reader: _TaskReader
) -> AgentEpisodeDispatch:
    bindings: list[InputBinding] = []
    for binding in agent.input_bindings:
        members = []
        for member in binding.members:
            if (source := member.source) is None:
                members.append(member)
                continue
            envelope = reader.envelope(
                ResultBinding(
                    task_id=member.source_operator_id, reference=source.reference
                )
            )
            members.append(
                member.model_copy(
                    update={
                        "value": value_text(envelope, source.element),
                        "source": None,
                    }
                )
            )
        bindings.append(binding.model_copy(update={"members": tuple(members)}))
    return agent.model_copy(update={"input_bindings": tuple(bindings)})


def _with_inputs(
    task: TaskEnvelopeStrict,
    upstream: dict[str, ResultEnvelope],
    element: tuple[Any] | None,
) -> TaskEnvelopeStrict:
    spec = _with_upstream_spec(task.spec, upstream) if upstream else task.spec
    if element is not None and "data" in type(spec).model_fields:
        spec = spec.model_copy(update={"data": {"type": "list", "items": [element[0]]}})
    if spec is task.spec:
        return task
    return _as_dispatched(task.model_copy(update={"spec": spec}))


def _with_upstream_spec[S: TaskSpecStrictBase](
    spec: S, upstream: dict[str, ResultEnvelope]
) -> S:
    merged = dict(spec.upstreamResults or {})
    merged.update({stage: envelope.result for stage, envelope in upstream.items()})
    return spec.model_copy(update={"upstreamResults": merged})


def _as_dispatched(task: TaskEnvelopeStrict) -> TaskEnvelopeStrict:
    """The envelope as an inline dispatch would have delivered it to this worker."""
    wire = task.model_dump(mode="json", exclude_none=True, by_alias=True)
    return TaskEnvelopeStrict.model_validate(normalize_numbers(wire))


__all__ = ["TaskInputHydrator", "input_unavailable", "input_unreadable"]
