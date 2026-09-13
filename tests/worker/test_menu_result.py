"""Tests for the declared result a leaf reports when it admits several embodiments."""

from typing import Any

import pytest

from shared.harness import HarnessResult, HarnessResultKind
from shared.schemas.result.catalog import InferenceResult
from shared.schemas.result.payloads import GenerationUsage, InferenceItem
from shared.tasks.specs import InferenceEmbodimentKind
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import ResolvedEmbodiment
from tests.worker.factories import make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.episode_support import EpisodeStepResult
from worker.executors.menu_result import declared_result

PROMPT = "hello"
MODEL = "Qwen/Qwen3-4B"


def _msg(embodiment: InferenceEmbodimentKind | None = None) -> Any:
    msg = make_worker_task_message(
        task_type=TaskType.INFERENCE,
        spec={
            "taskType": "inference",
            "model": {
                "source": {"identifier": MODEL},
                "vllm": {"gpu_memory_utilization": 0.9},
            },
            "resources": {"hardware": {"gpu": {"count": 1}}},
            "data": {"type": "list", "items": [PROMPT]},
            "service": {"mode": "local_eligible", "primary": "resident_served"},
        },
    )
    if embodiment is not None:
        msg.embodiment = ResolvedEmbodiment(alternative_id="alt", kind=embodiment)
    return msg


def _resident_step(value: str = "world") -> EpisodeStepResult:
    return EpisodeStepResult(
        harness_result=HarnessResult(kind=HarnessResultKind.COMPLETION, value=value),
        value=value,
    )


def _local_result(value: str = "world") -> InferenceResult:
    return InferenceResult(
        model=MODEL,
        items=[
            InferenceItem(
                index=0,
                prompt=PROMPT,
                output=value,
                finish_reason="stop",
                metadata={"prompt": PROMPT},
            )
        ],
        usage=GenerationUsage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            num_requests=1,
            latency_sec=0.1,
        ),
    )


def test_both_embodiments_report_one_identical_declared_result() -> None:
    resident = declared_result(
        _msg(InferenceEmbodimentKind.RESIDENT_SERVED), _resident_step()
    )
    local = declared_result(
        _msg(InferenceEmbodimentKind.SELF_CONTAINED), _local_result()
    )
    # Interchangeable downstream: a consumer, or a guard branching on the result,
    # cannot tell which embodiment ran.
    assert resident.model_dump() == local.model_dump()


def test_the_declared_result_carries_the_pinned_model_and_one_item() -> None:
    result = declared_result(
        _msg(InferenceEmbodimentKind.SELF_CONTAINED), _local_result()
    )
    assert isinstance(result, InferenceResult)
    assert result.model == MODEL
    assert [i.output for i in result.items] == ["world"]
    assert result.items[0].prompt == PROMPT


@pytest.mark.parametrize("field", ["finish_reason", "metadata", "usage"])
def test_fields_only_one_embodiment_can_produce_are_dropped(field: str) -> None:
    # A local run reports these and a resident relay cannot, so carrying them through
    # would make the embodiment choice observable downstream.
    result = declared_result(
        _msg(InferenceEmbodimentKind.SELF_CONTAINED), _local_result()
    )
    assert isinstance(result, InferenceResult)
    subject = result if field == "usage" else result.items[0]
    assert getattr(subject, field) is None


def test_a_leaf_bound_to_no_menu_records_what_its_executor_produced() -> None:
    produced = _local_result()
    assert declared_result(_msg(), produced) is produced

    step = _resident_step()
    assert declared_result(_msg(), step) is step


def test_a_non_terminal_episode_step_is_recorded_unchanged() -> None:
    # Only a completion carries the leaf's declared output; a yielded boundary keeps its
    # episode shape so the server can route it.
    step = EpisodeStepResult(
        harness_result=HarnessResult(kind=HarnessResultKind.BOUNDARY)
    )
    assert declared_result(_msg(InferenceEmbodimentKind.RESIDENT_SERVED), step) is step


def test_an_unprojectable_spec_fails_the_task_rather_than_guessing() -> None:
    msg = _msg(InferenceEmbodimentKind.RESIDENT_SERVED)
    msg.task.spec.data = {"type": "dataset", "url": "squad"}
    with pytest.raises(ExecutionError, match="not projectable"):
        declared_result(msg, _resident_step())
