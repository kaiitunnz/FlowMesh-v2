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

from shared.inference import hf_overrides
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.components.metadata import TaskMetadata
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.components.resources import ResourcesSpec
from shared.tasks.specs import DevModelSpecStrict, ServeSpecStrict
from shared.tasks.task_type import TaskType

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
    serve = config.substrate != "dev_model"
    size = family.serving_size
    vllm = _rendered_profile(family.engine_profile)
    vllm["tensor_parallel_size"] = size.tensor_parallel_size
    source = ModelSource(
        type="huggingface",
        identifier=family.model_ref,
        revision=vllm.pop("revision", None) or "main",
    )
    if serve:
        # A real vLLM embedding replica runs the pooling runner; a chat replica enables
        # LoRA so a resident consumer can load its adapter into a slot on demand.
        if family.interface == "embedding":
            vllm["runner"] = "pooling"
        else:
            vllm.update(enable_lora=True, max_loras=config.adapter_slots)
    elif family.interface != "embedding":
        # The GPU-free dev_model stand-in forwards by path and needs no serving-mode
        # flag, but it models a finite adapter registry of the same size so the slot
        # reclaim is exercised end to end: a lifetime-distinct load beyond the budget
        # fails until an unloaded slot frees.
        vllm["max_loras"] = config.adapter_slots
    # The GPU-free stand-in carries the size it stands in for, on no GPU.
    resources = ResourcesSpec(hardware=size.hardware(gpu=serve))
    model = ModelConfig(source=source, vllm=vllm)
    ttl = config.serve_ttl_sec or None
    spec: ServeSpecStrict | DevModelSpecStrict = (
        ServeSpecStrict(
            taskType=TaskType.SERVE, resources=resources, model=model, ttlSeconds=ttl
        )
        if serve
        else DevModelSpecStrict(
            taskType=TaskType.DEV_MODEL,
            resources=resources,
            model=model,
            ttlSeconds=ttl,
        )
    )
    envelope = TaskEnvelopeStrict(
        apiVersion="flowmesh/v1",
        kind="ResidentServe",
        metadata=TaskMetadata(name=f"resident-{replica.replica_id}"),
        spec=spec,
    )
    workflow_id, entries = await runtime.register(
        owner.principal_id,
        owner.org_id,
        envelope.model_dump_json(exclude_none=True, by_alias=True),
        format="native",
        resident=True,
    )
    try:
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
    except BaseException:
        # The caller learns no serve task id from a failed cold start, so nothing else
        # would ever reap the one registered here.
        runtime.cancel_workflow(workflow_id, reason="resident cold start failed")
        raise
    return entries[0].task_id


def _rendered_profile(profile: str | None) -> dict[str, Any]:
    """Render a family's engine profile as the serve task's engine configuration.

    The engine takes RoPE settings as config overrides, as the local executor passes
    them.
    """
    rendered: dict[str, Any] = json.loads(profile) if profile else {}
    if overrides := hf_overrides(
        rendered.pop("rope_scaling", None), rendered.pop("rope_theta", None)
    ):
        rendered["hf_overrides"] = overrides
    return rendered
