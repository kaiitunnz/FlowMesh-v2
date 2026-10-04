"""Tests for the serve spec dispatch validation."""

import re
from typing import Any

import pytest

from server.task.parser import parse_workflow
from shared.tasks.specs import ServeSpecStrict, ServeSpecTemplate


def _strict(**fields: Any) -> ServeSpecStrict:
    return ServeSpecStrict.model_validate({"taskType": "serve", **fields})


def _template(**fields: Any) -> ServeSpecTemplate:
    return ServeSpecTemplate.model_validate({"taskType": "serve", **fields})


class TestValidateDispatchable:
    def test_without_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="requests no GPU"):
            _strict(
                model={"source": {"identifier": "Qwen/Qwen3-7B"}}
            ).validate_dispatchable()

    def test_zero_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="requests no GPU"):
            _strict(
                model={"source": {"identifier": "Qwen/Qwen3-7B"}},
                resources={"hardware": {"gpu": {"count": 0}}},
            ).validate_dispatchable()

    def test_with_gpu_ok(self) -> None:
        _strict(
            model={"source": {"identifier": "Qwen/Qwen3-7B"}},
            resources={"hardware": {"gpu": {"count": 1}}},
        ).validate_dispatchable()

    def test_template_without_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="requests no GPU"):
            _template(
                model={"source": {"identifier": "Qwen/Qwen3-7B"}}
            ).validate_dispatchable()

    def test_template_with_gpu_ok(self) -> None:
        _template(
            model={"source": {"identifier": "Qwen/Qwen3-7B"}},
            resources={"hardware": {"gpu": {"count": 2}}},
        ).validate_dispatchable()


_GPU = {"hardware": {"gpu": {"count": 1}}}


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "api-key",
        "host",
        "port",
        "model",
        "uds",
        "hos",
        "api-ke",
        "mod",
        "model=evil/model",
        "compilation_config.level",
        "Host",
        "config",
    ],
)
@pytest.mark.parametrize("build", [_strict, _template])
def test_an_engine_option_the_executor_owns_is_not_dispatchable(
    key: str, build: Any
) -> None:
    spec = build(
        model={"source": {"identifier": "Qwen/Qwen3-7B"}, "vllm": {key: "x"}},
        resources=_GPU,
    )
    with pytest.raises(ValueError, match=re.escape(f"model.vllm.{key} is not")):
        spec.validate_dispatchable()


def test_a_serve_spec_setting_its_engine_key_is_refused_at_submission() -> None:
    workflow = """
apiVersion: flowmesh/v1
kind: ServeTask
metadata: {name: s}
spec:
  taskType: serve
  model:
    source: {identifier: Qwen/Qwen2.5-0.5B-Instruct}
    vllm: {api_key: mine, max_model_len: 1024}
  resources: {hardware: {gpu: {count: 1}}}
"""
    with pytest.raises(ValueError, match="model.vllm.api_key is not supported"):
        parse_workflow(workflow, "native")


@pytest.mark.parametrize("build", [_strict, _template])
def test_engine_options_the_executor_leaves_to_the_spec_are_dispatchable(
    build: Any,
) -> None:
    build(
        model={
            "source": {"identifier": "Qwen/Qwen3-7B"},
            "vllm": {
                "max_model_len": 1024,
                "env_vars": {"A": "1"},
                "served_model_name": "alias",
                "revision": "main",
                "model_impl": "vllm",
            },
        },
        resources=_GPU,
    ).validate_dispatchable()


@pytest.mark.parametrize("env_vars", [["A=1"], {"A": 1}, {"A": None}])
@pytest.mark.parametrize("build", [_strict, _template])
def test_engine_variables_that_are_not_strings_are_not_dispatchable(
    env_vars: Any, build: Any
) -> None:
    spec = build(
        model={
            "source": {"identifier": "Qwen/Qwen3-7B"},
            "vllm": {"env_vars": env_vars},
        },
        resources=_GPU,
    )
    with pytest.raises(ValueError, match="env_vars must map variable names to strings"):
        spec.validate_dispatchable()
