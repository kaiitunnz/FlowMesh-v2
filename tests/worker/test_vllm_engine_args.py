"""The shipped all-args vLLM example builds a real vLLM engine configuration.

The engine is mocked by vLLM's own ``EngineArgs``, so every keyword the executor
forwards from ``model.vllm`` must be one vLLM accepts.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

pytest.importorskip("vllm", reason="vllm not installed (needs --extra inference-gpu)")
torch = pytest.importorskip("torch", reason="torch not installed")

from vllm.engine.arg_utils import EngineArgs  # noqa: E402

from tests.worker.factories import DEFAULT_WORKER_CONFIG  # noqa: E402
from worker.executors import vllm_executor  # noqa: E402
from worker.executors.vllm_executor import VLLMExecutor  # noqa: E402

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "templates"
    / "inference_vllm_all_args.yaml"
)


def _engine(**kwargs: Any) -> MagicMock:
    EngineArgs(**kwargs)
    return MagicMock()


def test_every_example_engine_arg_reaches_a_real_engine_config() -> None:
    vllm_cfg = yaml.safe_load(_EXAMPLE.read_text())["spec"]["model"]["vllm"]

    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(vllm_executor, "LLM", side_effect=_engine) as llm,
    ):
        VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)._init_vllm_engine(
            ident="org/model",
            vllm_cfg=dict(vllm_cfg),
            checkpoint_cfg={},
            new_inference_spec={},
            requested_gpu_count=1,
            revision=None,
            extra_llm_kwargs={},
            adjust_tp=lambda size: size,
            task_ids=None,
        )

    forwarded = llm.call_args.kwargs
    accepted = {field.name for field in dataclasses.fields(EngineArgs)}
    assert set(forwarded) <= accepted
    assert {"quantization", "max_model_len", "hf_overrides"} <= set(forwarded)


def test_the_engine_starts_with_its_collective_traffic_on_loopback() -> None:
    names = ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "VLLM_HOST_IP")
    seen: dict[str, str | None] = {}

    def engine(**_: Any) -> MagicMock:
        seen.update({name: os.environ.get(name) for name in names})
        return MagicMock()

    with (
        patch.dict(os.environ, {"NCCL_SOCKET_IFNAME": "eth0"}),
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(vllm_executor, "LLM", side_effect=engine),
    ):
        VLLMExecutor(DEFAULT_WORKER_CONFIG, lifecycle=None)._init_vllm_engine(
            ident="org/model",
            vllm_cfg={"env_vars": {"GLOO_SOCKET_IFNAME": "ib0"}},
            checkpoint_cfg={},
            new_inference_spec={},
            requested_gpu_count=1,
            revision=None,
            extra_llm_kwargs={},
            adjust_tp=lambda size: size,
            task_ids=None,
        )

    assert seen == {
        "NCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "ib0",
        "VLLM_HOST_IP": "127.0.0.1",
    }
