"""Hydrating the upstream values a task consumes from the references it was dispatched.

Control names what a task reads — each upstream stage's settled result, the collection
element a fan-out child runs on, the producer results an agent's first-turn inputs are
frozen to — and this worker reads them through its content plane: its own cache, then an
authorized peer's copy, then the shared store. Every read is verified against its
reference, and the values are installed exactly as an inline dispatch delivers them,
before anything validates or runs the task.

A task whose store cannot be reached reports the inputs it could not read, and control
decides whose fault that is. Content that is missing, corrupt, out of the task's scope,
or not a result envelope fails the task.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel

from shared.content import ContentReference, ContentStoreError, ContentUnavailable
from shared.harness import AgentEpisodeDispatch, InputBinding
from shared.schemas.event import TaskFailureKind
from shared.schemas.result import BaseExecutorResult, ResultEnvelope
from shared.schemas.result.binding import (
    NotAResultEnvelope,
    binding_text,
    element_value,
    result_envelope,
    skip_envelope_bytes,
    upstream_value,
)
from shared.tasks import MergedChildTaskStrict, TaskEnvelopeStrict
from shared.tasks.artifact_binding import (
    ArtifactBinding,
    ArtifactKind,
    UnknownArtifactToken,
    names_artifact_token,
    resolve_artifact_input,
    substitute_artifact_tokens,
)
from shared.tasks.result_binding import BindingKind, ResultBinding
from shared.tasks.specs import SSHSpecStrict, TaskSpecStrictBase
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.json import normalize_numbers
from shared.utils.nofollow import PathRefused
from shared.utils.task_output import ARTIFACTS_DIR

from ..executors.base_executor import ExecutionError
from ..executors.utils.artifacts import data_artifact_refs
from .access import ContentAccessDenied, ContentBackendUnsupported
from .artifact_inputs import (
    ArtifactInputUnavailable,
    ArtifactInputUnreadable,
    ArtifactMaterializer,
)
from .plane import WorkerContentPlane

# A read that cannot reach the store retries briefly before the task reports it.
_READ_ATTEMPTS = 3
_READ_BACKOFF_SEC = 0.2


def input_unavailable(message: str, reference: ContentReference) -> ExecutionError:
    return ExecutionError(
        message,
        retryable=True,
        failure_kind=TaskFailureKind.INPUT_UNAVAILABLE,
        unavailable_inputs=(reference,),
    )


def input_unreadable(message: str) -> ExecutionError:
    return ExecutionError(f"input_unreadable: {message}", retryable=False)


def _check_scope(msg: WorkerTaskMessage, binding: ArtifactBinding) -> None:
    if binding.reference.authorization_scope != msg.content_scope:
        raise input_unreadable(
            f"task {msg.task_id} runs in scope {msg.content_scope} and an "
            f"artifact it loads is in {binding.reference.authorization_scope}"
        )


@contextmanager
def _reading_artifacts(task_id: str) -> Iterator[None]:
    """Classify a failure to put an artifact input on disk."""
    try:
        yield
    except ArtifactInputUnavailable as exc:
        raise input_unavailable(
            f"task {task_id} cannot reach an artifact it loads: {exc}", exc.reference
        ) from exc
    except ContentBackendUnsupported as exc:
        raise ExecutionError(str(exc), retryable=True) from exc
    except (ContentStoreError, ArtifactInputUnreadable) as exc:
        raise input_unreadable(
            f"task {task_id} artifact input unreadable: {exc}"
        ) from exc


def _materialized(
    materializer: ArtifactMaterializer, task_id: str, binding: ArtifactBinding
) -> Path:
    with _reading_artifacts(task_id):
        return materializer.materialize(binding)


def _input_producer(msg: WorkerTaskMessage, stage: str) -> str | None:
    if (binding := (msg.upstream_results or {}).get(stage)) is not None:
        return binding.task_id
    return (msg.upstream_task_ids or {}).get(stage)


def read_input(
    plane: WorkerContentPlane | None,
    task_id: str,
    scope: str,
    reference: ContentReference,
    backoff_sec: float = _READ_BACKOFF_SEC,
) -> bytes:
    """One verified input object, classified by what kept it from being read.

    A store that cannot be reached, or access that has not arrived, retries briefly and
    then reports the input unavailable, naming the reference so control can tell a
    shared outage from this worker's own path to the store. A worker that reaches no
    store, or one it cannot open, fails as its own fault. Content that is missing,
    corrupt, or outside the task's scope fails the task.
    """
    if plane is None:
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
            return plane.hydrate(task_id, reference)
        except ContentBackendUnsupported as exc:
            raise ExecutionError(str(exc), retryable=True) from exc
        except (ContentUnavailable, ContentAccessDenied) as exc:
            if attempt + 1 == _READ_ATTEMPTS:
                raise input_unavailable(
                    f"task {task_id} cannot reach input "
                    f"{reference.content_digest}: {exc}",
                    reference,
                ) from exc
            time.sleep(backoff_sec)
        except ContentStoreError as exc:
            raise input_unreadable(
                f"task {task_id} input {reference.content_digest}: {exc}"
            ) from exc
    raise AssertionError("unreachable")


class TaskInputHydrator:
    """Installs the values a task's dispatch names by reference."""

    def __init__(
        self,
        plane: WorkerContentPlane | None,
        backoff_sec: float = _READ_BACKOFF_SEC,
        cancelled: Callable[[str], bool] = lambda task_id: False,
    ) -> None:
        self._plane = plane
        self._backoff_sec = backoff_sec
        self._cancelled = cancelled

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
        upstream: dict[str, BaseExecutorResult] = {}
        for stage, binding in _bound(msg.upstream_results):
            if binding.whole_result:
                envelopes[stage] = reader.envelope_bytes(binding)
            upstream[stage] = reader.upstream_value(binding)
        element: tuple[Any] | None = None
        if (ref := msg.input_element) is not None:
            source = reader.reference_envelope(ref.reference)
            try:
                element = (element_value(source, ref),)
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

    def install_artifacts(self, msg: WorkerTaskMessage, root: Path) -> None:
        """Put each published artifact the spec names under ``root`` and its path in
        the spec, in place."""
        if not msg.artifact_inputs:
            return
        inputs = msg.artifact_inputs or {}
        for binding in inputs.values():
            _check_scope(msg, binding)
        with self._materializer(msg, root) as materializer:
            paths = {
                key: _materialized(materializer, msg.task_id, binding).as_posix()
                for key, binding in inputs.items()
            }
        scope = msg.dispatch_id or ""
        try:
            msg.task = _as_dispatched(msg.task, paths, scope)
            if msg.merged_children:
                msg.merged_children = [
                    _as_dispatched(child, paths, scope) for child in msg.merged_children
                ]
        except UnknownArtifactToken as exc:
            raise input_unreadable(str(exc)) from exc

    def install_upstream_artifacts(self, msg: WorkerTaskMessage, root: Path) -> None:
        """Put each upstream artifact a spec's data loads under ``root``, and record
        its local path on the spec.

        Only the artifacts the data's refs name are read, each from the producer's
        publications the dispatch carries; a ref to a path no producer published fails
        the task.
        """
        if not msg.upstream_artifacts:
            return
        specs = [msg.task.spec, *(child.spec for child in msg.merged_children or ())]
        wanted = [(spec, refs) for spec in specs if (refs := data_artifact_refs(spec))]
        if not wanted:
            return
        published = {
            producer: {binding.key: binding for binding in bindings}
            for producer, bindings in msg.upstream_artifacts.items()
        }
        with self._materializer(msg, root) as materializer:
            for spec, refs in wanted:
                paths: dict[tuple[str, str], str] = {}
                for producer, path in refs:
                    missing = input_unreadable(
                        f"task {producer} published no artifact named {path!r}"
                    )
                    found = resolve_artifact_input(published.get(producer, {}), path)
                    if found is None:
                        raise missing
                    binding, inner = found
                    _check_scope(msg, binding)
                    local = _materialized(materializer, msg.task_id, binding)
                    if inner and not (local := local / inner).is_file():
                        raise missing
                    paths[(producer, path)] = local.as_posix()
                spec.record_hydrated_artifacts(paths)

    def stage_ssh_inputs(self, msg: WorkerTaskMessage, root: Path) -> None:
        """Stage writable copies of the published artifacts of each upstream task an
        SSH task's inputs name, as its result bundle lays them out
        (``<root>/<task_id>/artifacts/<name>``), and record where."""
        spec = msg.task.spec
        if not isinstance(spec, SSHSpecStrict) or not spec.inputs:
            return
        producers = {
            producer
            for entry in spec.inputs
            if (producer := _input_producer(msg, entry.stage.strip())) is not None
        }
        published = {
            producer: [
                binding
                for binding in (msg.upstream_artifacts or {}).get(producer, ())
                if binding.kind is not ArtifactKind.ARCHIVE
            ]
            for producer in producers
        }
        if not any(published.values()):
            return
        staged: dict[str, str] = {}
        with self._materializer(msg, root) as materializer:
            for producer, bindings in published.items():
                for binding in bindings:
                    _check_scope(msg, binding)
                    rel = PurePosixPath(producer, ARTIFACTS_DIR, binding.name)
                    with _reading_artifacts(msg.task_id):
                        materializer.copy_at(binding, rel)
                staged[producer] = (root / producer).as_posix()
        msg.record_staged_inputs(staged)

    @contextmanager
    def _materializer(
        self, msg: WorkerTaskMessage, root: Path
    ) -> Iterator[ArtifactMaterializer]:
        if self._plane is None:
            raise ExecutionError(
                f"task {msg.task_id} loads published artifacts and this worker reaches "
                "no fabric content store",
                retryable=True,
            )
        task_id = msg.task_id
        materializer = ArtifactMaterializer(
            self._plane, task_id, root, lambda: self._cancelled(task_id)
        )
        try:
            entered = materializer.__enter__()
        except PathRefused as exc:
            raise ExecutionError(
                f"task {task_id} cannot hold its artifact inputs at {root}: {exc}",
                retryable=False,
            ) from exc
        with materializer:
            yield entered

    def _hydrate_child(
        self, reader: "_TaskReader", child: MergedChildTaskStrict
    ) -> MergedChildTaskStrict:
        if not child.upstream_results:
            return child
        upstream = {
            stage: reader.upstream_value(binding)
            for stage, binding in _bound(child.upstream_results)
        }
        return _as_dispatched(
            child.model_copy(update={"spec": _with_upstream_spec(child.spec, upstream)})
        )

    def read(self, task_id: str, scope: str, reference: ContentReference) -> bytes:
        return read_input(self._plane, task_id, scope, reference, self._backoff_sec)


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
        return self.reference_bytes(binding.reference)

    def reference_bytes(self, reference: ContentReference) -> bytes:
        if (data := self._bytes.get(reference)) is None:
            data = self._hydrator.read(self._task_id, self._scope, reference)
            self._bytes[reference] = data
        return data

    def upstream_value(self, binding: ResultBinding) -> BaseExecutorResult:
        """What a task's upstream map carries for one input: its whole result, or
        the value the binding reads as."""
        try:
            return upstream_value(binding, self.envelope)
        except IndexError as exc:
            raise input_unreadable(str(exc)) from exc

    def envelope(self, binding: ResultBinding) -> ResultEnvelope:
        if binding.reference is not None:
            return self.reference_envelope(binding.reference)
        return _parsed(
            self.envelope_bytes(binding), f"the stored result of task {binding.task_id}"
        )

    def reference_envelope(self, reference: ContentReference) -> ResultEnvelope:
        if (cached := self._envelopes.get(reference)) is None:
            cached = _parsed(
                self.reference_bytes(reference),
                f"the stored result {reference.content_digest}",
            )
            self._envelopes[reference] = cached
        return cached


def _parsed(data: bytes, source: str) -> ResultEnvelope:
    try:
        return result_envelope(data, source)
    except NotAResultEnvelope as exc:
        raise input_unreadable(str(exc)) from exc


def _bound(
    upstream: dict[str, ResultBinding] | None,
) -> list[tuple[str, ResultBinding]]:
    """The inputs that carry a value to read: a result or a skip, or a value that is
    not one whole result."""
    return [
        (stage, binding)
        for stage, binding in (upstream or {}).items()
        if binding.reference is not None
        or binding.skip is not None
        or binding.kind is not BindingKind.RESULT
    ]


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
            members.append(
                member.model_copy(
                    update={
                        "value": binding_text(source, reader.envelope),
                        "source": None,
                    }
                )
            )
        bindings.append(binding.model_copy(update={"members": tuple(members)}))
    return agent.model_copy(update={"input_bindings": tuple(bindings)})


def _with_inputs(
    task: TaskEnvelopeStrict,
    upstream: dict[str, BaseExecutorResult],
    element: tuple[Any] | None,
) -> TaskEnvelopeStrict:
    spec = _with_upstream_spec(task.spec, upstream) if upstream else task.spec
    if element is not None and "data" in type(spec).model_fields:
        spec = spec.model_copy(update={"data": {"type": "list", "items": [element[0]]}})
    if spec is task.spec:
        return task
    return _as_dispatched(task.model_copy(update={"spec": spec}))


def _with_upstream_spec[S: TaskSpecStrictBase](
    spec: S, upstream: dict[str, BaseExecutorResult]
) -> S:
    merged = dict(spec.upstreamResults or {})
    merged.update(upstream)
    return spec.model_copy(update={"upstreamResults": merged})


def _as_dispatched[M: BaseModel](
    model: M, artifact_paths: dict[str, str] | None = None, scope: str = ""
) -> M:
    """Return the model as an inline dispatch delivers it to this worker, each artifact
    token of its dispatch's ``scope`` replaced by the artifact's local path."""
    wire = model.model_dump(mode="json", exclude_none=True, by_alias=True)
    if artifact_paths:
        wire = _with_artifact_paths(wire, artifact_paths, scope)
    return type(model).model_validate(normalize_numbers(wire))


def _with_artifact_paths(value: Any, paths: dict[str, str], scope: str) -> Any:
    if isinstance(value, str):
        return (
            substitute_artifact_tokens(value, paths, scope)
            if names_artifact_token(value)
            else value
        )
    if isinstance(value, dict):
        return {
            key: _with_artifact_paths(item, paths, scope) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_with_artifact_paths(item, paths, scope) for item in value]
    return value


__all__ = ["TaskInputHydrator", "input_unavailable", "input_unreadable", "read_input"]
