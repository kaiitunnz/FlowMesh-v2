"""A worker hydrates exactly the values an inline dispatch would have delivered.

Each case builds the message control dispatched before its values moved to the worker —
values inline, delivered over the same wire — and the message it dispatches now, with
references, and checks the hydrated task equals the inline one.
"""

import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct

from server.task.results import ResultReader
from shared.content import (
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    SharedFilesystemObjectStore,
)
from shared.harness import AgentEpisodeDispatch, InputBinding, InputBindingMember
from shared.harness.adapter import HarnessBackendKey
from shared.inference import (
    CanonicalInferenceContract,
    CanonicalInferenceInputSource,
    InferenceSourceKind,
    InputResolutionError,
    UpstreamProvenance,
)
from shared.schemas.event import TaskFailureKind
from shared.schemas.result import RESULT_MEDIA_TYPE, ResultEnvelope
from shared.schemas.result.binding import collection_elements, value_text
from shared.tasks import MergedChildTaskStrict
from shared.tasks.result_binding import ResultBinding, ResultValueRef
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.json import normalize_numbers
from worker.content.inputs import TaskInputHydrator
from worker.executors.base_executor import ExecutionError
from worker.executors.inference.resolution import resolve_task_contract

_SCOPE = "org"
_PRODUCED = {
    "taskType": "echo",
    "items": [{"output": "facet-a"}, "facet-b", {"output": {"n": 2.0}}],
}


class _Plane:
    """A content plane reading one shared store, failing reads while told to."""

    def __init__(self, store: SharedFilesystemObjectStore) -> None:
        self.store = store
        self.error: Exception | None = None
        self.reads: list[ContentReference] = []

    def hydrate(self, task_id: str, reference: ContentReference) -> bytes:
        self.reads.append(reference)
        if self.error is not None:
            raise self.error
        return self.store.hydrate(reference)


@pytest.fixture
def plane() -> _Plane:
    return _Plane(SharedFilesystemObjectStore(Path(tempfile.mkdtemp()) / "content"))


def _store(plane: _Plane, task_id: str, result: dict[str, Any]) -> ResultBinding:
    envelope = ResultEnvelope.model_validate({"task_id": task_id, "result": result})
    reference = plane.store.write(
        _SCOPE,
        envelope.model_dump_json(indent=2).encode("utf-8"),
        media_type=RESULT_MEDIA_TYPE,
    )
    return ResultBinding(task_id=task_id, reference=reference)


def _deliver(message: WorkerTaskMessage) -> WorkerTaskMessage:
    """The message as a worker receives it: control's wire form through the relay."""
    wire = message.model_dump(mode="json", exclude_none=True, by_alias=True)
    relayed = MessageToDict(ParseDict(wire, Struct()), preserving_proto_field_name=True)
    return WorkerTaskMessage.model_validate(normalize_numbers(relayed))


def _message(spec: dict[str, Any], **fields: Any) -> WorkerTaskMessage:
    return WorkerTaskMessage.model_validate(
        {
            "task_id": "tsk-consumer",
            "workflow_id": "wfl-1",
            "owner_id": "owner",
            "content_scope": _SCOPE,
            "assigned_worker": "wkr-1",
            "dispatched_at": "2026-09-27T00:00:00Z",
            "task": {
                "apiVersion": "flowmesh/v1",
                "kind": "Task",
                "metadata": {"name": "wf:consumer"},
                "spec": spec,
            },
            **fields,
        }
    )


def _hydrate(plane: _Plane, message: WorkerTaskMessage) -> WorkerTaskMessage:
    delivered = _deliver(message)
    TaskInputHydrator(cast(Any, plane), backoff_sec=0.0).hydrate(delivered)
    return delivered


def _read(plane: _Plane, binding: ResultBinding) -> ResultEnvelope:
    return ResultReader(plane.store).read(binding)


def test_upstream_results_hydrate_to_the_inline_map(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    other = _store(plane, "tsk-q", {"taskType": "echo", "items": ["q"]})
    authored = {"_upstreamResults": {"p": {"stale": True}, "kept": {"x": 1}}}
    spec = {"taskType": "echo", "data": {"type": "list", "items": ["x"]}, **authored}

    inline = _deliver(
        _message(
            {
                **spec,
                "_upstreamResults": {
                    "p": _read(plane, producer).result.model_dump(mode="json"),
                    "kept": {"x": 1},
                    "q": _read(plane, other).result.model_dump(mode="json"),
                },
            }
        )
    )
    hydrated = _hydrate(
        plane, _message(spec, upstream_results={"p": producer, "q": other})
    )

    assert hydrated.task == inline.task
    assert set(hydrated.spec.upstreamResults or {}) == {"p", "kept", "q"}


def test_a_skipped_upstream_hydrates_to_its_skip_envelope(plane: _Plane) -> None:
    skipped = ResultBinding(
        task_id="tsk-s", skip={"skipped": True}, settled_at="2026-09-27T00:00:00Z"
    )
    spec = {"taskType": "echo", "data": {"type": "list", "items": ["x"]}}

    hydrated = _hydrate(plane, _message(spec, upstream_results={"s": skipped}))

    inline = _deliver(
        _message({**spec, "_upstreamResults": {"s": {}}}),
    )
    assert hydrated.task == inline.task
    envelope = ResultEnvelope.model_validate_json(
        hydrated.upstream_envelope("s") or b""
    )
    assert envelope.metadata == {"skipped": True}
    assert envelope.received_at == "2026-09-27T00:00:00Z"
    assert plane.reads == []


def test_the_envelope_bytes_are_the_stored_bytes(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    assert producer.reference is not None
    hydrated = _hydrate(
        plane,
        _message({"taskType": "echo"}, upstream_results={"p": producer}),
    )
    assert hydrated.upstream_envelope("p") == plane.store.hydrate(producer.reference)


def test_a_merged_child_hydrates_its_own_upstream(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    child_spec = {"taskType": "echo", "data": {"type": "list", "items": ["y"]}}
    child = MergedChildTaskStrict.model_validate(
        {
            "task_id": "tsk-child",
            "owner_id": "owner",
            "workflow_id": "wfl-1",
            "spec": child_spec,
            "upstream_results": {"p": producer},
        }
    )
    inline_child = MergedChildTaskStrict.model_validate(
        {
            "task_id": "tsk-child",
            "owner_id": "owner",
            "workflow_id": "wfl-1",
            "spec": {
                **child_spec,
                "_upstreamResults": {
                    "p": _read(plane, producer).result.model_dump(mode="json")
                },
            },
        }
    )
    spec = {"taskType": "echo", "data": {"type": "list", "items": ["x"]}}

    hydrated = _hydrate(plane, _message(spec, merged_children=[child]))
    inline = _deliver(_message(spec, merged_children=[inline_child]))

    assert hydrated.merged_children is not None and inline.merged_children is not None
    assert hydrated.merged_children[0].spec == inline.merged_children[0].spec


@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_fan_out_element_hydrates_to_the_inline_data(
    plane: _Plane, index: int
) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    assert producer.reference is not None
    element = collection_elements(_read(plane, producer))[index]
    spec = {"taskType": "echo", "data": {"type": "list", "items": ["template"]}}

    hydrated = _hydrate(
        plane,
        _message(
            spec,
            input_element=ResultValueRef(reference=producer.reference, element=index),
        ),
    )
    inline = _deliver(_message({**spec, "data": {"type": "list", "items": [element]}}))

    assert hydrated.task == inline.task
    assert hydrated.hydrated_element() == (element,)


def _agent(members: tuple[InputBindingMember, ...]) -> AgentEpisodeDispatch:
    return AgentEpisodeDispatch(
        backend=HarnessBackendKey(backend="scripted", version="v1"),
        input_bindings=(
            InputBinding(port="in", provenance="producer", members=members),
        ),
    )


def _member(**fields: Any) -> InputBindingMember:
    return InputBindingMember(
        source_operator_id="producer",
        source_activation_id="act-0",
        outcome="success",
        **fields,
    )


def test_agent_inputs_hydrate_to_the_projected_strings(plane: _Plane) -> None:
    collection = _store(plane, "tsk-p", _PRODUCED)
    whole = _store(plane, "tsk-r", {"taskType": "agent", "value": "grounded"})
    assert collection.reference is not None and whole.reference is not None
    agent = _agent(
        (
            _member(source=ResultValueRef(reference=collection.reference, element=0)),
            _member(source=ResultValueRef(reference=collection.reference, element=2)),
            _member(source=ResultValueRef(reference=whole.reference)),
            _member(value="an inline literal"),
        )
    )

    hydrated = _hydrate(plane, _message({"taskType": "agent"}, agent_episode=agent))

    assert hydrated.agent_episode is not None
    (binding,) = hydrated.agent_episode.input_bindings
    assert [m.value for m in binding.members] == [
        value_text(_read(plane, collection), 0),
        value_text(_read(plane, collection), 2),
        "grounded",
        "an inline literal",
    ]
    assert [m.value for m in binding.members][:2] == ["facet-a", '{"n": 2.0}']
    assert all(m.source is None for m in binding.members)


def test_a_message_naming_nothing_is_left_as_delivered(plane: _Plane) -> None:
    inline = _message(
        {"taskType": "echo", "_upstreamResults": {"p": {"items": ["x"]}}},
        upstream_task_ids={"p": "tsk-p"},
    )
    delivered = _deliver(inline)
    TaskInputHydrator(None).hydrate(delivered)
    assert delivered == _deliver(inline)


def test_an_unreachable_store_reports_the_inputs_unavailable(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    plane.error = ContentUnavailable("store down")

    with pytest.raises(ExecutionError) as caught:
        _hydrate(
            plane, _message({"taskType": "echo"}, upstream_results={"p": producer})
        )

    assert caught.value.retryable
    assert caught.value.failure_kind is TaskFailureKind.INPUT_UNAVAILABLE
    assert len(plane.reads) == 3


def test_missing_content_fails_the_task(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    plane.error = ContentHydrationError("no such object")

    with pytest.raises(ExecutionError) as caught:
        _hydrate(
            plane, _message({"taskType": "echo"}, upstream_results={"p": producer})
        )

    assert not caught.value.retryable
    assert caught.value.failure_kind is None
    assert str(caught.value).startswith("input_unreadable:")


def test_content_that_is_not_an_envelope_fails_the_task(plane: _Plane) -> None:
    reference = plane.store.write(_SCOPE, b"not json", media_type=RESULT_MEDIA_TYPE)
    binding = ResultBinding(task_id="tsk-p", reference=reference)

    with pytest.raises(ExecutionError) as caught:
        _hydrate(plane, _message({"taskType": "echo"}, upstream_results={"p": binding}))

    assert not caught.value.retryable
    assert str(caught.value).startswith("input_unreadable:")


def test_a_reference_outside_the_task_scope_is_refused(plane: _Plane) -> None:
    envelope = ResultEnvelope.model_validate({"task_id": "tsk-p", "result": _PRODUCED})
    reference = plane.store.write(
        "another-org", envelope.model_dump_json().encode(), media_type=RESULT_MEDIA_TYPE
    )
    binding = ResultBinding(task_id="tsk-p", reference=reference)

    with pytest.raises(ExecutionError) as caught:
        _hydrate(plane, _message({"taskType": "echo"}, upstream_results={"p": binding}))

    assert not caught.value.retryable
    assert plane.reads == []


def test_an_element_past_the_collection_fails_the_task(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    assert producer.reference is not None
    with pytest.raises(ExecutionError) as caught:
        _hydrate(
            plane,
            _message(
                {"taskType": "echo"},
                input_element=ResultValueRef(reference=producer.reference, element=9),
            ),
        )
    assert not caught.value.retryable


def test_a_fan_out_child_contract_resolves_its_hydrated_element(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    assert producer.reference is not None
    contract = CanonicalInferenceContract(
        model="m",
        source=CanonicalInferenceInputSource(
            kind=InferenceSourceKind.UPSTREAM, node="tsk-p", element=1, max_items=1
        ),
    )
    message = _message(
        {"taskType": "inference", "data": {"type": "list", "items": ["template"]}},
        input_element=ResultValueRef(reference=producer.reference, element=1),
        declared_contract=contract.model_dump(mode="json"),
    )

    resolved = resolve_task_contract(_hydrate(plane, message))
    again = resolve_task_contract(_hydrate(plane, message))

    assert resolved is not None and again is not None
    assert resolved.request.prompts == ("facet-b",)
    assert resolved.binding.cardinality == 1
    assert resolved.binding.upstream == (
        UpstreamProvenance(
            node="tsk-p", content_digest=producer.reference.content_digest
        ),
    )
    assert again.binding.matches(resolved.binding)


def test_an_element_that_is_not_a_prompt_fails_before_any_model(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    assert producer.reference is not None
    contract = CanonicalInferenceContract(
        model="m",
        source=CanonicalInferenceInputSource(
            kind=InferenceSourceKind.UPSTREAM, node="tsk-p", element=2, max_items=1
        ),
    )
    message = _message(
        {"taskType": "inference"},
        input_element=ResultValueRef(reference=producer.reference, element=2),
        declared_contract=contract.model_dump(mode="json"),
    )

    with pytest.raises(InputResolutionError):
        resolve_task_contract(_hydrate(plane, message))


def test_an_upstream_with_nothing_bound_is_left_out(plane: _Plane) -> None:
    producer = _store(plane, "tsk-p", _PRODUCED)
    spec = {"taskType": "echo", "data": {"type": "list", "items": ["x"]}}

    hydrated = _hydrate(
        plane,
        _message(
            spec,
            upstream_results={"p": producer, "e": ResultBinding(task_id="tsk-e")},
        ),
    )
    inline = _deliver(
        _message(
            {
                **spec,
                "_upstreamResults": {
                    "p": _read(plane, producer).result.model_dump(mode="json")
                },
            }
        )
    )

    assert hydrated.task == inline.task
    assert hydrated.upstream_envelope("e") is None
