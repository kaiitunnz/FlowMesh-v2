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
from shared.tasks.task_type import TaskType
from shared.tools.model.schema import MODEL_INTERFACE
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import EXECUTOR_REGISTRY
from worker.executors.base_executor import ExecutionError
from worker.executors.service_leaf_executor import (
    _CALL_CORRELATION,
    ServiceLeafExecutor,
)
from worker.resident import ResidentRequestStore


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
    assert store.peek("tsk-test", _CALL_CORRELATION) == "hello there"


def test_explicit_messages_pass_through_as_a_chat_request(tmp_path: Path) -> None:
    ex, store = _executor()
    messages = [{"role": "user", "content": "summarize"}]
    ex.run(_msg({"messages": messages}), tmp_path)
    stashed = store.peek("tsk-test", _CALL_CORRELATION)
    assert stashed is not None and json.loads(stashed) == {"messages": messages}


def test_resume_completes_with_the_settled_value(tmp_path: Path) -> None:
    ex, _store = _executor()
    out = ex.run(
        _msg(
            {"prompt": "hello"},
            delivered_outcomes=[
                {
                    "call_correlation": _CALL_CORRELATION,
                    "kind": "result",
                    "value": "the answer",
                }
            ],
        ),
        tmp_path,
    )
    assert out.harness_result.kind is HarnessResultKind.COMPLETION
    assert out.value == "the answer"


def test_resume_on_a_denied_outcome_fails_the_leaf(tmp_path: Path) -> None:
    ex, _store = _executor()
    out = ex.run(
        _msg(
            {"prompt": "hello"},
            delivered_outcomes=[
                {
                    "call_correlation": _CALL_CORRELATION,
                    "kind": "denied",
                    "denial": "authority",
                }
            ],
        ),
        tmp_path,
    )
    assert out.harness_result.kind is HarnessResultKind.FAILURE


def test_missing_prompt_fails_cleanly(tmp_path: Path) -> None:
    ex, _store = _executor()
    with pytest.raises(ExecutionError, match="no prompt or messages"):
        ex.run(_msg({}), tmp_path)
