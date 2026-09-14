"""A menu leaf's stored result is rewritten into the shape its contract declares.

Both embodiments of a leaf store one result, so a consumer cannot tell which ran. The
projection runs where the fabric owns the stored result, from both the settlement path
and the result's own arrival, so it must survive being applied twice.
"""

import json
from pathlib import Path
from typing import Any

from server.task.inference_projection import menu_request_payload, project_menu_result
from shared.inference import SAMPLING_DEFAULTS
from shared.schemas.result.catalog import ResultEnvelope
from shared.schemas.result.io import result_file_path, write_result
from shared.tasks.specs import InferenceSpecStrict

_SPEC = InferenceSpecStrict.model_validate(
    {
        "taskType": "inference",
        "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
        "data": {"type": "list", "items": ["hello there"]},
    }
)


def test_the_request_carries_the_sampling_a_local_generation_would_apply() -> None:
    # The leaf declares no sampling, so the two embodiments are one contract only if the
    # relayed request states what a local generation would have applied by default.
    # Built from the spec alone, this is the request BOTH embodiments issue.
    body = json.loads(menu_request_payload(_SPEC) or "{}")

    assert body["messages"] == [{"role": "user", "content": "hello there"}]
    assert body["max_tokens"] == SAMPLING_DEFAULTS["max_tokens"] == 512
    assert body["temperature"] == SAMPLING_DEFAULTS["temperature"] == 0.7
    assert body["top_p"] == SAMPLING_DEFAULTS["top_p"] == 0.95


def test_declared_sampling_overrides_the_default_in_the_request() -> None:
    spec = InferenceSpecStrict.model_validate(
        {
            "taskType": "inference",
            "model": {"source": {"identifier": "Qwen/Qwen3-4B"}},
            "data": {"type": "list", "items": ["hello there"]},
            "inference": {"max_tokens": 10},
        }
    )
    body = json.loads(menu_request_payload(spec) or "{}")

    assert body["max_tokens"] == 10
    assert body["temperature"] == SAMPLING_DEFAULTS["temperature"]


def test_an_unprojectable_spec_carries_no_request() -> None:
    spec = InferenceSpecStrict.model_validate(
        {"taskType": "inference", "model": {"source": {"identifier": "q"}}}
    )
    assert menu_request_payload(spec) is None


def _store(tmp_path: Path, result: dict[str, Any]) -> None:
    write_result(
        tmp_path,
        ResultEnvelope.model_validate({"task_id": "tsk-a", "result": result}),
    )


def _stored(tmp_path: Path) -> dict[str, Any]:
    text = result_file_path(tmp_path, "tsk-a").read_text(encoding="utf-8")
    return json.loads(text)["result"]


def test_a_local_generation_is_reduced_to_the_declared_fields(tmp_path: Path) -> None:
    _store(
        tmp_path,
        {
            "type": "inference",
            "model": "Qwen/Qwen3-4B",
            "items": [
                {
                    "index": 0,
                    "prompt": "hello there",
                    "output": "hi",
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 12},
        },
    )
    assert project_menu_result(tmp_path, "tsk-a", _SPEC) is True

    result = _stored(tmp_path)
    assert result["items"] == [{"index": 0, "prompt": "hello there", "output": "hi"}]
    # A field only a local generation can report never reaches a consumer.
    assert result.get("usage") is None
    assert result["items"][0].get("finish_reason") is None


def test_a_relayed_invocation_projects_to_the_same_result(tmp_path: Path) -> None:
    # The resident embodiment stores the episode's terminal value, not items.
    _store(tmp_path, {"type": "base", "value": "hi"})
    assert project_menu_result(tmp_path, "tsk-a", _SPEC) is True

    result = _stored(tmp_path)
    assert result["model"] == "Qwen/Qwen3-4B"
    assert result["items"] == [{"index": 0, "prompt": "hello there", "output": "hi"}]


def test_projecting_twice_reproduces_the_same_result(tmp_path: Path) -> None:
    # It runs from both the settlement and the result's own arrival, unordered.
    _store(tmp_path, {"type": "base", "value": "hi"})
    assert project_menu_result(tmp_path, "tsk-a", _SPEC) is True
    once = _stored(tmp_path)
    assert project_menu_result(tmp_path, "tsk-a", _SPEC) is True

    assert _stored(tmp_path) == once


def test_it_does_nothing_until_the_result_has_arrived(tmp_path: Path) -> None:
    assert project_menu_result(tmp_path, "tsk-a", _SPEC) is False


def test_an_unprojectable_spec_is_left_alone(tmp_path: Path) -> None:
    spec = InferenceSpecStrict.model_validate(
        {"taskType": "inference", "model": {"source": {"identifier": "q"}}}
    )
    _store(tmp_path, {"type": "base", "value": "hi"})
    assert project_menu_result(tmp_path, "tsk-a", spec) is False
    assert _stored(tmp_path)["value"] == "hi"
