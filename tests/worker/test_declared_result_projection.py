"""A leaf whose contract the fabric resolves stores one result shape on the worker.

The result a worker stores is the one every reader sees: it is written to the worker's
own results directory, and a deployment that does not upload results leaves it there.
So the shape has to be settled before the write, not by a later pass on another node.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from shared.harness import HarnessResult, HarnessResultKind
from shared.inference import PROJECTION_DROPS, CanonicalInferenceRequest
from shared.schemas.result import BaseExecutorResult
from shared.schemas.result.catalog import InferenceResult
from shared.schemas.result.payloads import GenerationUsage, InferenceItem
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_hardware, make_worker_task_message
from worker.executors.base_executor import Executor
from worker.executors.episode_support import EpisodeStepResult
from worker.runner import Runner

_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_PARAMS = {"max_tokens": 512, "temperature": 0.7}

_CONTRACT = CanonicalInferenceRequest(
    model=_MODEL, prompts=("name one planet",), params=_PARAMS
)

_BATCH_PROMPTS = ("name one planet", "name one ocean", "name one river")
_BATCH_CONTRACT = CanonicalInferenceRequest(
    model=_MODEL, prompts=_BATCH_PROMPTS, params=_PARAMS
)
_BATCH_OUTPUTS = ("Mars", "Pacific", "Nile")


class _FixedExecutor(Executor):
    name = "fixed"

    def __init__(self, result: BaseExecutorResult) -> None:  # noqa: D107
        self._result = result

    def run(self, task, out_dir):  # type: ignore[no-untyped-def]
        return self._result

    def cancel(self, task_id: str) -> None:
        return None


def _stored(tmp_path: Path, result: BaseExecutorResult, contract: str | None) -> dict:
    """Run one task through the real runner and read the result it stored."""
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    executor = _FixedExecutor(result)
    msg = make_worker_task_message(
        {"taskType": "inference", "data": {"prompt": "name one planet"}},
        task_type=TaskType.INFERENCE,
    )
    msg.resolved_contract = contract
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path,
        hardware=make_worker_hardware(),
        executors={"inference": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    )
    runner.start()
    path = tmp_path / msg.task_id / "results.json"
    assert path.exists(), "the worker stored no result"
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def _native_local(
    prompts: tuple[str, ...] = ("name one planet",),
    outputs: tuple[str, ...] = ("Mars",),
) -> InferenceResult:
    """What a local generation reports: its own items, plus what only it can report."""
    return InferenceResult(
        model=_MODEL,
        items=[
            InferenceItem(
                index=index,
                prompt=prompt,
                output=output,
                finish_reason="stop",
                metadata={"engine": "vllm"},
            )
            for index, (prompt, output) in enumerate(zip(prompts, outputs))
        ],
        usage=GenerationUsage(
            prompt_tokens=4,
            completion_tokens=1,
            total_tokens=5,
            num_requests=len(prompts),
            latency_sec=0.2,
        ),
    )


def _native_resident(value: str = "Mars") -> EpisodeStepResult:
    """What a relayed invocation reports: the episode's terminal value."""
    return EpisodeStepResult(
        harness_result=HarnessResult(kind=HarnessResultKind.COMPLETION, value=value),
        value=value,
    )


def _native_resident_batch() -> EpisodeStepResult:
    """What a relayed batch invocation reports: one completion per declared prompt."""
    return _native_resident(json.dumps(list(_BATCH_OUTPUTS)))


def test_a_local_generation_stores_the_declared_shape(tmp_path: Path) -> None:
    stored = _stored(tmp_path, _native_local(), _CONTRACT.model_dump_json())

    assert stored["model"] == _CONTRACT.model
    assert [(i["index"], i["prompt"], i["output"]) for i in stored["items"]] == [
        (0, "name one planet", "Mars")
    ]
    assert stored.get("usage") is None
    assert stored["items"][0].get("finish_reason") is None
    assert stored["items"][0].get("metadata") is None


def test_a_relayed_invocation_stores_the_same_shape(tmp_path: Path) -> None:
    # The point of the projection: a reader cannot tell which embodiment ran the leaf.
    local = _stored(tmp_path / "local", _native_local(), _CONTRACT.model_dump_json())
    resident = _stored(
        tmp_path / "resident", _native_resident(), _CONTRACT.model_dump_json()
    )
    # The artifact root is the task's own directory on the worker that ran it, not part
    # of what the leaf declares.
    local.pop("_artifacts", None)
    resident.pop("_artifacts", None)

    assert local == resident


def test_a_batch_leaf_stores_one_item_per_declared_prompt(tmp_path: Path) -> None:
    stored = _stored(
        tmp_path,
        _native_local(_BATCH_PROMPTS, _BATCH_OUTPUTS),
        _BATCH_CONTRACT.model_dump_json(),
    )

    assert [(i["index"], i["prompt"], i["output"]) for i in stored["items"]] == [
        (0, "name one planet", "Mars"),
        (1, "name one ocean", "Pacific"),
        (2, "name one river", "Nile"),
    ]


def test_a_relayed_batch_stores_the_same_shape(tmp_path: Path) -> None:
    # One resident invocation carries the whole batch, so its terminal value holds every
    # completion — and a reader still cannot tell which embodiment ran the leaf.
    local = _stored(
        tmp_path / "local",
        _native_local(_BATCH_PROMPTS, _BATCH_OUTPUTS),
        _BATCH_CONTRACT.model_dump_json(),
    )
    resident = _stored(
        tmp_path / "resident",
        _native_resident_batch(),
        _BATCH_CONTRACT.model_dump_json(),
    )
    local.pop("_artifacts", None)
    resident.pop("_artifacts", None)

    assert local == resident
    assert len(resident["items"]) == len(_BATCH_PROMPTS)


def test_a_leaf_with_no_resolved_contract_stores_its_own_result(
    tmp_path: Path,
) -> None:
    # A leaf that admits one embodiment reports what that embodiment produced.
    stored = _stored(tmp_path, _native_local(), None)

    assert stored["usage"] is not None
    assert stored["items"][0]["finish_reason"] == "stop"


def test_the_dropped_fields_are_unset_under_both_embodiments(tmp_path: Path) -> None:
    for name, native in (("local", _native_local()), ("resident", _native_resident())):
        stored = _stored(tmp_path / name, native, _CONTRACT.model_dump_json())
        for field in PROJECTION_DROPS:
            assert stored.get(field) is None
            for item in stored["items"]:
                assert item.get(field) is None
