"""A warm omni executor drops an engine that vllm_omni closed during a run.

vllm_omni closes its engine inside ``generate()`` when a generation fails, and when a
``py_generator`` generation finishes, so the executor must not reuse it.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference)")

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs.omni import OmniText2GeneralSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors.base_executor import ExecutionError
from worker.executors.omni_text2general_executor import OmniText2GeneralExecutor

from .factories import DEFAULT_WORKER_CONFIG, make_worker_task_message


class _Omni:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.closed = False

    def generate(self, prompts: Any, sampling_params: Any, **kwargs: Any) -> list[Any]:
        if self.fail:
            raise RuntimeError("engine core died")
        return [
            SimpleNamespace(
                request_id="0_a",
                final_output_type="audio",
                error=None,
                multimodal_output={"audio": [0.1], "sample_rate": 24000},
                outputs=[SimpleNamespace(text="")],
            )
        ]

    def close(self) -> None:
        self.closed = True


def _executor(monkeypatch: pytest.MonkeyPatch, omni: _Omni) -> OmniText2GeneralExecutor:
    executor = OmniText2GeneralExecutor(DEFAULT_WORKER_CONFIG)

    def ensure(_spec_dict: dict[str, Any]) -> None:
        if executor._omni is None:
            executor._model_name = "org/omni"
            executor._omni = omni  # type: ignore[assignment]

    monkeypatch.setattr(executor, "_ensure_omni", ensure)
    monkeypatch.setattr(
        "worker.executors.omni_text2general_executor.save_audio", lambda *a, **k: None
    )
    return executor


def _run(executor: OmniText2GeneralExecutor, tmp_path: Path, **omni: Any) -> Any:
    spec = OmniText2GeneralSpecStrict(
        taskType=TaskType.OMNI_TEXT2GENERAL,
        model=ModelConfig(source=ModelSource(identifier="org/omni")),
        data={"type": "list", "items": ["prompt"]},
        omni={"output_format": "wav", **omni},
    )
    task = make_worker_task_message(spec, task_type=TaskType.OMNI_TEXT2GENERAL)
    return executor.run(task, tmp_path)


def test_a_failed_generation_drops_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omni = _Omni(fail=True)
    executor = _executor(monkeypatch, omni)
    with pytest.raises(ExecutionError, match="engine core died"):
        _run(executor, tmp_path)
    assert executor._omni is None and omni.closed


def test_a_py_generator_run_drops_the_engine_it_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omni = _Omni(fail=False)
    executor = _executor(monkeypatch, omni)
    result = _run(executor, tmp_path, py_generator=True)
    assert result.model == "org/omni"
    assert executor._omni is None


def test_a_successful_run_keeps_its_engine_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omni = _Omni(fail=False)
    executor = _executor(monkeypatch, omni)
    _run(executor, tmp_path)
    assert executor._omni is omni and not omni.closed


def test_the_engine_starts_with_its_collective_traffic_on_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(monkeypatch, _Omni(fail=False))
    started = executor._ensure_omni
    seen: dict[str, str | None] = {}

    def ensure(spec_dict: dict[str, Any]) -> None:
        seen.update(
            {
                k: os.environ.get(k)
                for k in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "VLLM_HOST_IP")
            }
        )
        started(spec_dict)

    monkeypatch.setattr(executor, "_ensure_omni", ensure)
    with patch.dict(os.environ, {"NCCL_SOCKET_IFNAME": "eth0"}):
        _run(executor, tmp_path)
    assert seen == {
        "NCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "VLLM_HOST_IP": "127.0.0.1",
    }
