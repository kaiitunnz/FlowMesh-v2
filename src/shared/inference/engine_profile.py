"""The vLLM engine configuration that decides what an engine returns.

A local vLLM executor reads a closed set of ``model.vllm`` keys. Each is either part of
the engine profile, which changes what the engine returns for a request, or
engine-local, which changes only how fast, where, or with what access it runs. A
resident replica serves one profile, so a leaf shares a replica only with leaves
declaring the same one.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ..tasks.placeholders import contains_placeholder
from ..utils.redact import is_credential_key

ENGINE_PROFILE_KEYS = frozenset(
    {
        "dtype",
        "enable_mm_embeds",
        "env_vars",
        "kv_cache_dtype",
        "limit_mm_per_prompt",
        "max_model_len",
        "quantization",
        "rope_scaling",
        "rope_theta",
        "tokenizer_revision",
        "trust_remote_code",
    }
)

# The pooling conversion shapes only an embedding engine; a chat engine ignores it.
EMBEDDING_PROFILE_KEYS = ENGINE_PROFILE_KEYS | {"convert"}

# ``seed`` seeds an engine whose requests are sampled draws either way, so it changes
# which sample a request gets, not what the request returns.
ENGINE_LOCAL_KEYS = frozenset(
    {
        "cpu_offload_gb",
        "download_dir",
        "enforce_eager",
        "gpu_memory_utilization",
        "hf_token",
        "max_cudagraph_capture_size",
        "max_num_batched_tokens",
        "seed",
        "tensor_parallel_size",
    }
)

_DEFAULT_REVISION = "main"
_DEFAULT_OFF = frozenset({"enable_mm_embeds", "trust_remote_code"})


def engine_profile(
    vllm: Mapping[str, Any] | None, revision: str | None, *, embedding: bool = False
) -> str | None:
    """Return the profile a chat or embedding leaf's engine configuration declares.

    A credential-named engine variable is access rather than outcome, so no profile
    carries one. A value that renders from upstream at dispatch is unknown when a
    replica is chosen, so no profile carries one either.
    """
    keys = EMBEDDING_PROFILE_KEYS if embedding else ENGINE_PROFILE_KEYS
    profile: dict[str, Any] = {}
    for key, value in (vllm or {}).items():
        if (
            key not in keys
            or (key in _DEFAULT_OFF and value is False)
            or contains_placeholder(value)
        ):
            continue
        if key == "env_vars" and isinstance(value, Mapping):
            value = {k: v for k, v in value.items() if not is_credential_key(str(k))}
            if not value:
                continue
        profile[key] = value
    if (
        revision
        and revision != _DEFAULT_REVISION
        and not contains_placeholder(revision)
    ):
        profile["revision"] = revision
    if not profile:
        return None
    return json.dumps(profile, sort_keys=True, separators=(",", ":"), default=str)


def hf_overrides(rope_scaling: Any, rope_theta: Any) -> dict[str, Any]:
    """Return the config overrides a vLLM engine takes a leaf's RoPE settings as."""
    overrides: dict[str, Any] = {}
    if rope_scaling is not None:
        overrides["rope_scaling"] = rope_scaling
    if rope_theta is not None:
        overrides["rope_theta"] = float(rope_theta)
    return overrides


def engine_profile_key(profile: str) -> str:
    """Return the short identity a profile adds to a service family."""
    return hashlib.sha256(profile.encode()).hexdigest()[:16]
