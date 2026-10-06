"""The task a materialized resident replica submits, for each kind of family."""

import asyncio
import json
import logging
from typing import Any, cast

import pytest
from lumid_hooks import PrincipalContext

from server.config import ResidentCapacityConfig
from server.resident import ReplicaIncarnation, ServiceFamily
from server.resident.materializer import materialize_resident_replica
from server.task.parser import parse_workflow
from server.task.v2.representations.serving_size import ServingSize
from shared.inference import engine_profile

_OWNER = PrincipalContext(
    principal_id="operator",
    org_id="acme",
    external_id="op",
    principal_type="admin",
    scopes=["*"],
)
_FAMILY = ServiceFamily(family="m", engine_batch_key="m", model_ref="org/m")
_SIZED = ServingSize(
    cpu=8,
    memory_bytes=32 * 1024**3,
    gpu_type="h100",
    gpu_count=2,
    gpu_memory_bytes=80 * 1024**3,
    tensor_parallel_size=2,
)
_SOURCE = {"type": "huggingface", "identifier": "org/m", "revision": "main"}
_DEFAULT_HARDWARE = {"cpu": 2, "memory": "4Gi", "gpu": {"type": "any", "count": 1}}


def _envelope(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "flowmesh/v1",
        "kind": "ResidentServe",
        "metadata": {"name": "resident-rpl-1"},
        "spec": spec,
    }


_CASES: dict[str, tuple[ResidentCapacityConfig, ServiceFamily, dict[str, Any]]] = {
    "default": (
        ResidentCapacityConfig(),
        _FAMILY,
        _envelope(
            {
                "taskType": "serve",
                "resources": {"hardware": _DEFAULT_HARDWARE},
                "model": {
                    "source": _SOURCE,
                    "vllm": {
                        "tensor_parallel_size": 1,
                        "enable_lora": True,
                        "max_loras": 4,
                    },
                },
            }
        ),
    ),
    "sized-gpu": (
        ResidentCapacityConfig(serve_ttl_sec=600.0),
        _FAMILY.model_copy(
            update={
                "serving_size": _SIZED,
                "engine_profile": engine_profile(
                    {"max_model_len": 1024, "rope_theta": 1e6}, "v2"
                ),
            }
        ),
        _envelope(
            {
                "taskType": "serve",
                "resources": {
                    "hardware": {
                        "cpu": 8,
                        "memory": "32Gi",
                        "gpu": {"type": "h100", "count": 2, "memory": "80Gi"},
                    }
                },
                "model": {
                    "source": {**_SOURCE, "revision": "v2"},
                    "vllm": {
                        "max_model_len": 1024,
                        "hf_overrides": {"rope_theta": 1000000.0},
                        "tensor_parallel_size": 2,
                        "enable_lora": True,
                        "max_loras": 4,
                    },
                },
                "ttlSeconds": 600.0,
            }
        ),
    ),
    "embedding": (
        ResidentCapacityConfig(),
        _FAMILY.model_copy(update={"interface": "embedding"}),
        _envelope(
            {
                "taskType": "serve",
                "resources": {"hardware": _DEFAULT_HARDWARE},
                "model": {
                    "source": _SOURCE,
                    "vllm": {"tensor_parallel_size": 1, "runner": "pooling"},
                },
            }
        ),
    ),
    "lora-chat": (
        ResidentCapacityConfig(adapter_slots=3),
        _FAMILY.model_copy(
            update={"engine_profile": engine_profile({"max_model_len": 2048}, None)}
        ),
        _envelope(
            {
                "taskType": "serve",
                "resources": {"hardware": _DEFAULT_HARDWARE},
                "model": {
                    "source": _SOURCE,
                    "vllm": {
                        "max_model_len": 2048,
                        "tensor_parallel_size": 1,
                        "enable_lora": True,
                        "max_loras": 3,
                    },
                },
            }
        ),
    ),
    "dev-model": (
        ResidentCapacityConfig(substrate="dev_model", adapter_slots=2),
        _FAMILY.model_copy(update={"serving_size": _SIZED}),
        _envelope(
            {
                "taskType": "dev_model",
                "resources": {
                    "hardware": {
                        "cpu": 8,
                        "memory": "32Gi",
                        "gpu": {"type": "any", "count": 0},
                    }
                },
                "model": {
                    "source": _SOURCE,
                    "vllm": {"tensor_parallel_size": 2, "max_loras": 2},
                },
            }
        ),
    ),
}


class _Entry:
    task_id = "tsk-resident"


class _Runtime:
    payload: str | None = None

    async def register(
        self,
        owner_id: str,
        org_id: str,
        payload: str,
        format: str,
        *,
        resident: bool = False,
    ) -> tuple[str, list[_Entry]]:
        self.payload = payload
        return "wfl-resident", [_Entry()]


def _canonical(value: Any) -> str:
    # Sorted canonical text tells 1 from 1.0 and from true, which == does not.
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("case", list(_CASES))
def test_a_replica_submits_its_family_task(case: str) -> None:
    config, family, expected = _CASES[case]
    runtime = _Runtime()
    replica = ReplicaIncarnation(replica_id="rpl-1", family="m", incarnation=1)

    asyncio.run(
        materialize_resident_replica(
            cast(Any, runtime),
            _OWNER,
            config,
            family,
            replica,
            logging.getLogger("test.materializer_payload"),
        )
    )

    assert runtime.payload is not None
    assert _canonical(json.loads(runtime.payload)) == _canonical(expected)
    parse_workflow(runtime.payload, "native")
