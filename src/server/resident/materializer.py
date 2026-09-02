"""Materialize a resident model-serving replica as an owned serve task.

Resident-capacity control provisions a replica by submitting a serve (or the GPU-free
`dev_model` stand-in) task through the runtime under the resolved system principal, and
records that ownership with the resource registrars. An operator then reads a resident
replica's logs through the same owner-scoped path as any other task, rather than through
a synthetic owner no principal can authenticate as.
"""

import json
import logging
from typing import Any

from flowmesh_hook import ResourceKind
from lumid_hooks import PrincipalContext

from ..auth import register_resource
from ..config import ResidentCapacityConfig
from ..task.runtime import TaskRuntime
from .state import ReplicaIncarnation, ServiceFamily


async def materialize_resident_replica(
    runtime: TaskRuntime,
    owner: PrincipalContext,
    config: ResidentCapacityConfig,
    family: ServiceFamily,
    replica: ReplicaIncarnation,
    logger: logging.Logger,
) -> str:
    """Submit the family's serve substrate as a task owned by `owner`; return its id."""
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
                "identifier": family.model_ref,
                "revision": "main",
            }
        },
        "accessMode": config.access_mode,
        "resident": True,
    }
    if config.serve_ttl_sec:
        spec["ttlSeconds"] = config.serve_ttl_sec
    payload = {
        "apiVersion": "flowmesh/v1",
        "kind": "ResidentServe",
        "metadata": {"name": f"resident-{replica.replica_id}"},
        "spec": spec,
    }
    workflow_id, entries = await runtime.register(
        owner.principal_id, owner.org_id, json.dumps(payload), format="native"
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
