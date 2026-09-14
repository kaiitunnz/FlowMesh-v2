"""The service-leaf executor yields one resident model boundary, then completes.

A resident-backed inference leaf builds its model request from the spec, keeps it in
worker-private custody and emits only a digest on the first step; a resume injects the
settled completion and finishes the leaf.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shared.harness import BoundaryEventKind, HarnessResultKind
from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from shared.tools.model.schema import MODEL_INTERFACE
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import EXECUTOR_REGISTRY
from worker.executors.base_executor import ExecutionError
from worker.executors.episode_support import EpisodeStepResult
from worker.executors.service_leaf_executor import (
    ServiceLeafExecutor,
    _call_correlation,
)
from worker.resident import ResidentRequestStore


def _body(store: ResidentRequestStore) -> dict:
    """The engine request the executor kept in worker-private custody."""
    payload = store.peek("tsk-test", _CORR)
    assert payload is not None
    return json.loads(payload)


def _step(result: BaseExecutorResult) -> EpisodeStepResult:
    assert isinstance(result, EpisodeStepResult)
    return result


_CORR = _call_correlation("tsk-test")


def _executor() -> tuple[ServiceLeafExecutor, ResidentRequestStore]:
    store = ResidentRequestStore()
    lifecycle = MagicMock()
    lifecycle.resident_requests = store
    return ServiceLeafExecutor(make_worker_config(), lifecycle=lifecycle), store


def _msg(spec_data: dict, **episode: object):
    return make_worker_task_message(
        {"taskType": "inference", "data": spec_data},
        task_type=TaskType.INFERENCE,
        service_episode={"interface": "chat", **episode},
    )


def _inference_msg(spec_data: dict, inference: dict, **episode: object):
    return make_worker_task_message(
        {"taskType": "inference", "data": spec_data, "inference": inference},
        task_type=TaskType.INFERENCE,
        service_episode={"interface": "chat", **episode},
    )


def test_the_declared_sampling_is_carried_to_the_replica() -> None:
    # A leaf's declared sampling governs its generation wherever it runs. Dropping it
    # here made a resident run of the same leaf generate under the engine's defaults
    # while a local run honoured the spec.
    ex, store = _executor()
    ex.run(
        _inference_msg(
            {"prompt": "hello there"},
            {"max_tokens": 10, "temperature": 0.0, "stop": ["\n"]},
        ),
        Path("/tmp"),  # noqa: S108
    )

    body = _body(store)
    assert body["max_tokens"] == 10
    assert body["temperature"] == 0.0
    assert body["stop"] == ["\n"]
    assert body["messages"] == [{"role": "user", "content": "hello there"}]


def test_a_leaf_declaring_a_literal_prompt_list_is_served(tmp_path: Path) -> None:
    # The shape a local generation reads. A leaf that admits both embodiments declares
    # its prompt once, so the replica has to read the same declaration.
    ex, store = _executor()
    ex.run(_msg({"type": "list", "items": ["hello there"]}), tmp_path)

    assert _body(store) == {"messages": [{"role": "user", "content": "hello there"}]}


def test_inputs_and_executor_settings_are_not_sent_as_engine_params() -> None:
    ex, store = _executor()
    ex.run(
        _inference_msg({"prompt": "hi"}, {"max_tokens": 4, "batch_size": 8}),
        Path("/tmp"),  # noqa: S108
    )

    body = _body(store)
    assert body["max_tokens"] == 4
    assert "batch_size" not in body


def _embedding_msg(spec_data: dict, **episode: object):
    return make_worker_task_message(
        {"taskType": "embedding", "data": spec_data},
        task_type=TaskType.EMBEDDING,
        service_episode={"interface": "embedding", **episode},
    )


def test_service_leaf_key_is_registered() -> None:
    assert "service_leaf" in EXECUTOR_REGISTRY
    cls = EXECUTOR_REGISTRY.get("service_leaf")
    # It advertises no task-type capability: the dispatch signal selects it.
    assert cls is not None and cls.supported_task_types == frozenset()


def test_first_step_captures_the_request_and_yields_a_resident_boundary(
    tmp_path: Path,
) -> None:
    ex, store = _executor()
    out = ex.run(_msg({"prompt": "hello there"}), tmp_path)

    req = out.harness_result.request
    assert out.harness_result.kind is HarnessResultKind.BOUNDARY
    assert req is not None
    assert req.kind is BoundaryEventKind.INVOCATION
    assert req.interface == MODEL_INTERFACE
    # The raw request is stripped to a digest and kept worker-private.
    assert req.request_payload is None
    assert req.request_digest is not None
    assert _body(store) == {"messages": [{"role": "user", "content": "hello there"}]}


def test_explicit_messages_pass_through_as_a_chat_request(tmp_path: Path) -> None:
    ex, store = _executor()
    messages = [{"role": "user", "content": "summarize"}]
    ex.run(_msg({"messages": messages}), tmp_path)
    stashed = store.peek("tsk-test", _CORR)
    assert stashed is not None and json.loads(stashed) == {"messages": messages}


def test_resume_completes_with_the_settled_value(tmp_path: Path) -> None:
    ex, _store = _executor()
    out = _step(
        ex.run(
            _msg(
                {"prompt": "hello"},
                delivered_outcomes=[
                    {
                        "call_correlation": _CORR,
                        "kind": "result",
                        "value": "the answer",
                    }
                ],
            ),
            tmp_path,
        )
    )
    assert out.harness_result.kind is HarnessResultKind.COMPLETION
    assert out.value == "the answer"


def test_resume_on_a_denied_outcome_fails_the_leaf(tmp_path: Path) -> None:
    ex, _store = _executor()
    out = _step(
        ex.run(
            _msg(
                {"prompt": "hello"},
                delivered_outcomes=[
                    {
                        "call_correlation": _CORR,
                        "kind": "denied",
                        "denial": "authority",
                    }
                ],
            ),
            tmp_path,
        )
    )
    assert out.harness_result.kind is HarnessResultKind.FAILURE


def test_missing_prompt_fails_cleanly(tmp_path: Path) -> None:
    ex, _store = _executor()
    with pytest.raises(ExecutionError, match="no prompt or messages"):
        ex.run(_msg({}), tmp_path)


def test_embedding_leaf_captures_the_input_list_and_yields_a_boundary(
    tmp_path: Path,
) -> None:
    ex, store = _executor()
    out = ex.run(_embedding_msg({"input": ["alpha", "beta"]}), tmp_path)

    req = out.harness_result.request
    assert out.harness_result.kind is HarnessResultKind.BOUNDARY
    assert req is not None and req.interface == MODEL_INTERFACE
    assert req.request_payload is None and req.request_digest is not None
    stashed = store.peek("tsk-test", _CORR)
    assert stashed is not None and json.loads(stashed) == {"input": ["alpha", "beta"]}


def test_embedding_resume_completes_with_the_settled_vectors(tmp_path: Path) -> None:
    ex, _store = _executor()
    vectors = json.dumps([{"index": 0, "embedding": [0.1, 0.2]}])
    out = _step(
        ex.run(
            _embedding_msg(
                {"input": ["alpha"]},
                delivered_outcomes=[
                    {
                        "call_correlation": _CORR,
                        "kind": "result",
                        "value": vectors,
                    }
                ],
            ),
            tmp_path,
        )
    )
    assert out.harness_result.kind is HarnessResultKind.COMPLETION
    assert out.value is not None
    assert json.loads(out.value) == [{"index": 0, "embedding": [0.1, 0.2]}]


def test_embedding_missing_input_fails_cleanly(tmp_path: Path) -> None:
    ex, _store = _executor()
    with pytest.raises(ExecutionError, match="no input"):
        ex.run(_embedding_msg({}), tmp_path)
