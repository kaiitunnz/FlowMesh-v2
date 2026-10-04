"""The engine profile classifies every engine key a local vLLM executor reads."""

import ast
import json
from pathlib import Path

from shared.inference.engine_profile import (
    EMBEDDING_PROFILE_KEYS,
    ENGINE_LOCAL_KEYS,
    ENGINE_PROFILE_KEYS,
    engine_profile,
)

_EXECUTORS = Path(__file__).resolve().parents[2] / "src" / "worker" / "executors"
_CHAT_READERS = ("vllm_executor.py", "vllm_lora_executor.py")
_EMBEDDING_READERS = ("vllm_executor.py", "vllm_embedding_executor.py")


def _key(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keys_read(source: str) -> set[str]:
    """Keys read from a ``vllm_cfg`` mapping or listed as accepted engine args.

    A key popped only to discard it is not read.
    """
    tree = ast.parse(source)
    discarded = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    }
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and id(node) not in discarded
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "vllm_cfg"
            and node.args
            and (key := _key(node.args[0])) is not None
        ):
            keys.add(key)
        elif (
            isinstance(node, ast.Compare)
            and (key := _key(node.left)) is not None
            and any(
                isinstance(c, ast.Name) and c.id == "vllm_cfg" for c in node.comparators
            )
        ):
            keys.add(key)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_ACCEPTED_ENGINE_ARGS"
            and isinstance(node.value, ast.Dict)
        ):
            keys.update(
                name
                for item in node.value.keys
                if item is not None and (name := _key(item)) is not None
            )
    return keys


def _read_by(readers: tuple[str, ...]) -> set[str]:
    return set().union(*(_keys_read((_EXECUTORS / f).read_text()) for f in readers))


def test_every_engine_key_a_local_executor_reads_is_classified() -> None:
    assert _read_by(_CHAT_READERS) == ENGINE_PROFILE_KEYS | ENGINE_LOCAL_KEYS
    assert _read_by(_EMBEDDING_READERS) == EMBEDDING_PROFILE_KEYS | ENGINE_LOCAL_KEYS
    assert not EMBEDDING_PROFILE_KEYS & ENGINE_LOCAL_KEYS


def test_the_pooling_conversion_keys_only_an_embedding_profile() -> None:
    assert engine_profile({"convert": "embed"}, None) is None
    assert engine_profile({"convert": "embed"}, None, embedding=True) == (
        '{"convert":"embed"}'
    )


def test_an_undeclared_or_engine_local_configuration_has_no_profile() -> None:
    assert engine_profile(None, None) is None
    assert engine_profile({}, "main") is None
    assert (
        engine_profile(
            {"gpu_memory_utilization": 0.5, "seed": 7, "trust_remote_code": False},
            None,
        )
        is None
    )


def test_a_profile_carries_outcome_changing_keys_and_no_credential() -> None:
    profile = engine_profile(
        {
            "max_model_len": 1024,
            "enforce_eager": True,
            "env_vars": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "HF_TOKEN": "msk-x"},
        },
        "v2",
    )

    assert profile is not None
    assert json.loads(profile) == {
        "env_vars": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        "max_model_len": 1024,
        "revision": "v2",
    }
    assert "HF_TOKEN" not in profile and "msk-x" not in profile


def test_a_profile_is_canonical() -> None:
    assert engine_profile({"dtype": "bf16", "max_model_len": 8}, None) == (
        engine_profile({"max_model_len": 8, "dtype": "bf16"}, None)
    )
