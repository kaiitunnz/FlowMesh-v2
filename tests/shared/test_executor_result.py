"""Round-trip tests for the shared executor-result schema."""

import json
from typing import Any

import pytest
from pydantic import Field, ValidationError

from shared.schemas.artifact import ArtifactContext, ArtifactRef
from shared.schemas.result import (
    APIResult,
    BaseExecutorResult,
    DataRetrievalItem,
    DataRetrievalResult,
    EchoItem,
    EchoResult,
    InferenceItem,
    InferenceItemStrict,
    InferenceResult,
    OmniText2ImageResult,
    ResultEnvelope,
    RoutedValue,
)
from shared.tasks.specs import EchoSpecStrict
from shared.tasks.task_type import TaskType


class _SampleResult(BaseExecutorResult):
    ok: bool = True
    items: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None


def test_subclass_round_trip_through_base_preserves_extra_fields() -> None:
    """A subclass's executor-specific fields survive a JSON trip through the
    base class via ``extra='allow'``."""
    original = _SampleResult(
        ok=True,
        items=[{"output": "hello"}],
        usage={"latency_sec": 0.5},
        _artifacts=ArtifactContext(base_dir="/tmp/t", base_url="http://h"),
    )
    payload = original.model_dump_json()

    base = BaseExecutorResult.model_validate_json(payload)
    redumped = json.loads(base.model_dump_json())

    assert redumped["items"] == [{"output": "hello"}]
    assert redumped["usage"] == {"latency_sec": 0.5}
    assert redumped["_artifacts"] == {"base_dir": "/tmp/t", "base_url": "http://h"}


def test_recursive_children_round_trip() -> None:
    """Nested ``children`` deserialize as ``BaseExecutorResult`` and re-emit
    their extra fields when serialized."""
    parent = _SampleResult(
        items=[{"output": "p"}],
        children={
            "c1": _SampleResult(items=[{"output": "c"}], usage={"total_tokens": 3}),
        },
    )
    payload = parent.model_dump_json()
    base = BaseExecutorResult.model_validate_json(payload)

    assert "c1" in base.children
    child = base.children["c1"]
    redumped = json.loads(child.model_dump_json())
    assert redumped["items"] == [{"output": "c"}]
    assert redumped["usage"] == {"total_tokens": 3}


def test_artifact_ref_is_a_typed_path_reference() -> None:
    ref = ArtifactRef(path="lora/final")
    assert ref.model_dump() == {"path": "lora/final"}


def test_envelope_round_trip_preserves_subclass_payload() -> None:
    """The production write→read path (``write_result_in_envelope`` →
    ``ResultEnvelope.model_validate``) round-trips subclass fields."""
    inner = _SampleResult(items=[{"output": "hello"}], usage={"total_tokens": 7})
    env = ResultEnvelope(task_id="tsk-x", result=inner)
    raw = env.model_dump_json()

    parsed = ResultEnvelope.model_validate_json(raw)
    dumped = parsed.result.model_dump()
    assert dumped["items"] == [{"output": "hello"}]
    assert dumped["usage"] == {"total_tokens": 7}


def test_none_declared_fields_are_dropped_from_serialization() -> None:
    """Optional declared fields that are ``None`` are omitted, not emitted as
    ``null``, while set fields survive."""
    result = InferenceResult(model=None, usage=None)
    dumped = result.model_dump(by_alias=True)
    assert "model" not in dumped
    assert "usage" not in dumped
    assert dumped["task_type"] == TaskType.INFERENCE
    assert dumped["items"] == []


def test_open_passthrough_nulls_are_preserved() -> None:
    """The drop-``None`` filter does not recurse into open mapping fields, so an
    explicit ``null`` inside a passthrough payload survives."""
    result = APIResult.model_validate(
        {
            "executor": "api",
            "method": "GET",
            "url": "http://h",
            "status_code": 200,
            "json": {"present": None, "value": 1},
            "usage": {"cost": None},
            "text": None,
        }
    )
    dumped = result.model_dump(by_alias=True)
    assert dumped["json"] == {"present": None, "value": 1}
    assert dumped["usage"] == {"cost": None}
    assert "text" not in dumped


def test_extra_passthrough_nulls_survive_but_declared_none_drops() -> None:
    """An ``extra='allow'`` item drops its declared ``None`` fields but keeps
    connector-specific extra keys, even when their value is ``null``."""
    item = DataRetrievalItem.model_validate(
        {"index": 0, "tokens_out": None, "connector_extra": None, "keep": 1}
    )
    dumped = item.model_dump()
    assert "tokens_out" not in dumped
    assert dumped["connector_extra"] is None
    assert dumped["keep"] == 1


def test_required_nullable_fields_survive_serialization() -> None:
    """Required-but-nullable fields (declared without a default, e.g. the Omni
    modality outputs) keep their ``null`` so the payload still re-validates."""
    result = OmniText2ImageResult(model=None, image=None, items=[])
    dumped = result.model_dump(by_alias=True)
    assert dumped["model"] is None
    assert dumped["image"] is None
    reloaded = OmniText2ImageResult.model_validate_json(result.model_dump_json())
    assert reloaded == result


def test_a_produced_inference_item_requires_index_prompt_and_output() -> None:
    """A producer reports ``index``, ``prompt`` and ``output`` for every item, so an
    item omitting one is rejected; ``finish_reason`` and ``metadata`` may be absent."""
    item = InferenceItemStrict(index=0, prompt="hi", output="hello")
    assert item.finish_reason is None and item.metadata is None

    complete = {"index": 0, "prompt": "hi", "output": "hello", "finish_reason": "stop"}
    for missing in ("index", "prompt", "output"):
        payload = dict(complete)
        del payload[missing]
        with pytest.raises(ValidationError):
            InferenceItemStrict.model_validate(payload)


def test_a_produced_null_output_reads_back_as_null() -> None:
    """``output`` must be reported but may be null: a structured output whose model
    returned ``null`` is produced and reads back as null."""
    result = InferenceResult(
        items=[
            InferenceItemStrict.produce({"index": 0, "prompt": "hi", "output": None})
        ]
    )
    envelope = ResultEnvelope(task_id="t", result=result).model_dump_json()
    reread = ResultEnvelope.model_validate_json(envelope).result
    assert isinstance(reread, InferenceResult)
    assert reread.items[0].output is None


@pytest.mark.parametrize("output", ["hello", None, ["a", "b"], {"answer": 1}], ids=repr)
def test_a_produced_result_equals_its_stored_read_back(output: Any) -> None:
    item = InferenceItemStrict.produce({"index": 0, "prompt": "hi", "output": output})
    result = InferenceResult(model="m", items=[item])

    assert type(item) is InferenceItem
    assert InferenceResult.model_validate(result.model_dump()) == result


# Every item shape an inference producer has stored, as stored.
_STORED_ITEMS = {
    "generated": {"index": 0, "prompt": "p", "output": "o", "finish_reason": "stop"},
    "no finish reason": {"index": 0, "prompt": "p", "output": "o"},
    "empty completion": {"index": 0, "prompt": "p", "output": "", "metadata": {"a": 1}},
    "structured null": {"index": 0, "prompt": "p", "finish_reason": "stop"},
    "grouped": {
        "index": 0,
        "prompt": "p",
        "output": ["a", "b"],
        "finish_reason": ["stop", None],
    },
    "table without row fields": {"output": ["a", "b"]},
}


@pytest.mark.parametrize("stored", _STORED_ITEMS.values(), ids=list(_STORED_ITEMS))
def test_every_stored_inference_item_reads_back_unchanged(
    stored: dict[str, Any],
) -> None:
    envelope = {
        "task_id": "t",
        "result": {
            "task_type": "inference",
            "items": [stored],
            "children": {"c": {"task_type": "inference", "items": [stored]}},
        },
    }
    result = ResultEnvelope.model_validate_json(json.dumps(envelope)).result
    assert isinstance(result, InferenceResult)
    assert isinstance(result.items[0], InferenceItem)
    reread = json.loads(result.model_dump_json())
    assert reread["items"] == [stored]
    assert reread["children"]["c"]["items"] == [stored]


def test_a_stored_omni_result_without_a_model_reads_back() -> None:
    envelope = {
        "task_id": "t",
        "result": {
            "task_type": "omni_text2image",
            "model": None,
            "image": None,
            "items": [],
        },
    }
    result = ResultEnvelope.model_validate_json(json.dumps(envelope)).result
    assert isinstance(result, OmniText2ImageResult)
    assert result.model is None


def test_drop_none_round_trip_is_lossless() -> None:
    """Dropping ``None`` on the wire round-trips: absent optionals reload as
    their ``None`` defaults."""
    original = DataRetrievalResult(type="sql", count=None, metadata=None)
    reloaded = DataRetrievalResult.model_validate_json(original.model_dump_json())
    assert reloaded.count is None
    assert reloaded.metadata is None
    assert reloaded == original


def test_upstream_results_preserve_subclass_payload_over_the_wire() -> None:
    """A server-injected upstream result keeps its subclass payload when the
    task spec is serialized to the worker.

    ``upstreamResults`` is typed with the base class, so without
    ``SerializeAsAny`` the subclass fields a downstream graph node reads
    (e.g. ``items[0].output``) would be stripped on the wire and arrive empty.
    """
    upstream = EchoResult(items=[EchoItem(output="literal_from_a")])
    spec = EchoSpecStrict(taskType=TaskType.ECHO).model_copy(
        update={"upstreamResults": {"echo-a": upstream}}
    )

    reloaded = EchoSpecStrict.model_validate_json(spec.model_dump_json(by_alias=True))
    assert reloaded.upstreamResults is not None
    injected = reloaded.upstreamResults["echo-a"]
    assert injected.model_dump()["items"][0]["output"] == "literal_from_a"


def test_only_a_tagged_payload_reads_as_a_routed_value() -> None:
    routed = RoutedValue(routed_value={"draft": "z"})
    stored = {"ok": True, "routed_value": 1, "custom": 2}

    assert isinstance(
        ResultEnvelope.model_validate_json(
            ResultEnvelope(task_id="tsk-x", result=routed).model_dump_json()
        ).result,
        RoutedValue,
    )
    read = ResultEnvelope.model_validate({"task_id": "tsk-x", "result": stored}).result
    assert type(read) is BaseExecutorResult
    assert read.model_dump() == stored
