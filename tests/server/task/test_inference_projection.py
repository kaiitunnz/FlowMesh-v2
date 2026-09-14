"""A menu leaf's stored result is rewritten into the shape its contract declares.

Both embodiments of a leaf store one result, so a consumer cannot tell which ran. The
projection runs where the fabric owns the stored result, from both the settlement path
and the result's own arrival, so it must survive being applied twice.
"""

import json
from pathlib import Path
from typing import Any

from server.task.inference_projection import project_menu_result
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
