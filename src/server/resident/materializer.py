"""Materialize a resident replica as an owned task.

Resident-capacity control provisions a replica by submitting its family's substrate
through the runtime under the resolved system principal — a serve (or the GPU-free
`dev_model` stand-in) task for a model-serving family, a sandbox-host allocation for a
sandbox family — and records that ownership with the resource registrars. An operator
then reads a resident replica's logs through the same owner-scoped path as any other
task, rather than through a synthetic owner no principal can authenticate as.
"""

import json
import logging
from typing import Any

from flowmesh_hook import ResourceKind
from lumid_hooks import PrincipalContext

from ..auth import register_resource
from ..config import ResidentCapacityConfig
from ..task.runtime import TaskRuntime
from .state import ReplicaIncarnation, ServiceFamily, ServiceFamilyKind


async def materialize_resident_replica(
    runtime: TaskRuntime,
    owner: PrincipalContext,
    config: ResidentCapacityConfig,
    family: ServiceFamily,
    replica: ReplicaIncarnation,
    logger: logging.Logger,
) -> str:
    """Submit the family's substrate as a task owned by `owner`; return its id."""
    spec = (
        _sandbox_host_spec()
        if family.kind is ServiceFamilyKind.SANDBOX_HOST
        else _model_serving_spec(config, family)
    )
    if config.serve_ttl_sec:
        spec["ttlSeconds"] = config.serve_ttl_sec
    payload = {
        "apiVersion": "flowmesh/v1",
        "kind": "ResidentServe",
        "metadata": {"name": f"resident-{replica.replica_id}"},
        "spec": spec,
    }
    workflow_id, entries = await runtime.register(
        owner.principal_id,
        owner.org_id,
        json.dumps(payload),
        format="native",
        resident=True,
    )
    await register_resource(
        owner,
        ResourceKind.WORKFLOW,
        workflow_id,
        {"format": "native", "task_count": len(entries)},
        logger,
    )
    for entry in entries:
        await register_resource(
            owner,
            ResourceKind.TASK,
            entry.task_id,
            {"workflow_id": workflow_id},
            logger,
        )
    return entries[0].task_id


def _sandbox_host_spec() -> dict[str, Any]:
    """A sandbox host holds a worker's reusable runtime state and session capacity."""
    return {
        "taskType": "sandbox_host",
        "resources": {
            "hardware": {"cpu": 2, "memory": "4Gi", "gpu": {"type": "any", "count": 0}}
        },
    }


def _model_serving_spec(
    config: ResidentCapacityConfig, family: ServiceFamily
) -> dict[str, Any]:
    """The serve substrate a model-serving family's replica runs."""
    spec_type = "dev_model" if config.substrate == "dev_model" else "serve"
    spec: dict[str, Any] = {
        "taskType": spec_type,
        "resources": {
            "hardware": {
                "cpu": 2,
                "memory": "4Gi",
                "gpu": {"type": "any", "count": 0 if spec_type == "dev_model" else 1},
            }
        },
        "model": {
            "source": {
                "type": "huggingface",
                "identifier": family.service_ref,
                "revision": "main",
            }
        },
    }
    if spec_type == "serve":
        # A real vLLM embedding replica runs the pooling runner; a chat replica enables
        # LoRA so a resident consumer can load its adapter into a slot on demand.
        if family.interface == "embedding":
            spec["model"]["vllm"] = {"runner": "pooling"}
        else:
            spec["model"]["vllm"] = {
                "enable_lora": True,
                "max_loras": config.adapter_slots,
            }
    elif family.interface != "embedding":
        # The GPU-free dev_model stand-in forwards by path and needs no serving-mode
        # flag, but it models a finite adapter registry of the same size so the slot
        # reclaim is exercised end to end: a lifetime-distinct load beyond the budget
        # fails until an unloaded slot frees.
        spec["model"]["vllm"] = {"max_loras": config.adapter_slots}
    return spec
