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
        "convert",
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


def engine_profile(vllm: Mapping[str, Any] | None, revision: str | None) -> str | None:
    """The canonical profile a leaf's engine configuration declares, or None.

    A credential-named engine variable is access rather than outcome, so no profile
    carries one. A value that renders from upstream at dispatch is unknown when a
    replica is chosen, so no profile carries one either.
    """
    profile: dict[str, Any] = {}
    for key, value in (vllm or {}).items():
        if (
            key not in ENGINE_PROFILE_KEYS
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


def engine_profile_key(profile: str) -> str:
    """The short identity a profile adds to a service family."""
    return hashlib.sha256(profile.encode()).hexdigest()[:16]
