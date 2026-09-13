"""Compile a local-eligible inference leaf into its embodiment menu.

The menu is emitted only after the two embodiments are proven to run the same declared
contract, so the scheduler chooses between them without changing what the workflow
declared. The proof is narrow on purpose: a leaf whose equivalence rests on anything
this module does not check keeps the single embodiment its binding names.
"""

import json

from shared.inference import (
    CanonicalInferenceRequest,
    CanonicalProjectionError,
    canonical_request,
)
from shared.tasks.specs import (
    InferenceBackend,
    InferenceEmbodimentKind,
    InferenceSpecStrict,
    InferenceSpecTemplate,
)

from ...parser import ParsedTask
from ..representations.operators import LeafProfile, ServiceDependency
from ..representations.plan import (
    EpisodeBoundaryKind,
    EpisodeSpec,
    InferenceEmbodimentCandidate,
    InferenceEmbodimentMenu,
    LocalExecutionEnvelope,
    ResidencyIntent,
    ServiceFamilyRequirement,
)
from ..representations.versioning import content_digest
from .diagnostics import compile_error

# Token accounting differs between a local generation and a relayed engine response and
# is deliberately outside the equivalence proof: it is telemetry on an optional field
# both embodiments leave unset by default, not part of what the leaf declares. Every
# other attribute of the contract below is proven equal before a menu is emitted.
_TELEMETRY_EXEMPT = ("usage",)


def embodiment_menu(
    task: ParsedTask,
    spec: InferenceSpecStrict | InferenceSpecTemplate,
    dependency: ServiceDependency,
    profile: LeafProfile,
    node_id: str,
) -> InferenceEmbodimentMenu:
    """Prove a local-eligible leaf's two embodiments equivalent and emit its menu.

    Raises a compile error naming the unproven attribute when they are not.
    """
    request = _canonical_request(task, spec)
    _reject_unproven(task, spec)

    resource_class = profile.binding.task_type.value
    resident = InferenceEmbodimentCandidate(
        alternative_id=f"{node_id}:{InferenceEmbodimentKind.RESIDENT_SERVED.value}",
        kind=InferenceEmbodimentKind.RESIDENT_SERVED,
        episode=EpisodeSpec(
            boundary=EpisodeBoundaryKind.SERVICE_ISSUE, resource_class=resource_class
        ),
        service_family_requirement=ServiceFamilyRequirement(
            family=dependency.service_family,
            engine_batch_key=dependency.engine_batch_key,
            isolation=dependency.isolation,
        ),
        residency_intent=ResidencyIntent(
            service_family=dependency.service_family, conditional=True
        ),
    )
    local = InferenceEmbodimentCandidate(
        alternative_id=f"{node_id}:{InferenceEmbodimentKind.SELF_CONTAINED.value}",
        kind=InferenceEmbodimentKind.SELF_CONTAINED,
        episode=EpisodeSpec(
            boundary=EpisodeBoundaryKind.TASK, resource_class=resource_class
        ),
        # The proof above pins the vLLM engine and rejects adapters, so a
        # self-contained run of this leaf resolves to exactly one local executor.
        local=LocalExecutionEnvelope(
            executor_key="vllm", gpu_count=_declared_gpu_count(spec)
        ),
    )
    by_kind = {resident.kind: resident, local.kind: local}
    primary = spec.service.primary if spec.service else None
    return InferenceEmbodimentMenu(
        contract_fingerprint=_contract_fingerprint(spec, dependency, profile, request),
        primary=by_kind[primary].alternative_id if primary else local.alternative_id,
        candidates=(resident, local),
    )


def _canonical_request(
    task: ParsedTask, spec: InferenceSpecStrict | InferenceSpecTemplate
) -> CanonicalInferenceRequest:
    try:
        return canonical_request(spec)
    except CanonicalProjectionError as exc:
        raise _unproven(task, f"its request projection is not shared: {exc}") from exc


def _reject_unproven(
    task: ParsedTask, spec: InferenceSpecStrict | InferenceSpecTemplate
) -> None:
    """Reject a leaf whose embodiments this module cannot prove equivalent.

    A resident replica serves a pinned vLLM engine, so a local embodiment is proven only
    for a leaf that pins the same engine. An adapter, a shard, a parallel split, or a
    postprocessing step changes what one embodiment produces relative to the other.
    """
    if spec.enforce_cpu is True or spec.backend() is not InferenceBackend.VLLM:
        raise _unproven(
            task,
            "it does not pin the vLLM engine a resident replica serves; declare "
            "model.vllm and no enforce_cpu",
        )
    if spec.adapters:
        raise _unproven(task, "adapter serving is proven for one embodiment only")
    if spec.parallel is not None or spec.shard is not None:
        raise _unproven(task, "a sharded or parallel split applies to one embodiment")
    if spec.postprocess is not None:
        raise _unproven(task, "postprocessing applies to one embodiment")


def _unproven(task: ParsedTask, reason: str) -> Exception:
    source_kind, source_id = (
        ("graph_node", task.graph_node_name)
        if task.graph_node_name
        else ("stage", task.local_name) if task.local_name else ("legacy", task.task_id)
    )
    return compile_error(
        "embodiment.not-contract-equivalent",
        f"a local_eligible inference leaf admits both embodiments only when they run "
        f"one proven contract; here {reason}",
        source_id or task.task_id,
        source_kind,
    )


def _declared_gpu_count(
    spec: InferenceSpecStrict | InferenceSpecTemplate,
) -> int | None:
    hardware = spec.resources.hardware if spec.resources else None
    gpu = hardware.gpu if hardware else None
    return gpu.count if gpu else None


def _contract_fingerprint(
    spec: InferenceSpecStrict | InferenceSpecTemplate,
    dependency: ServiceDependency,
    profile: LeafProfile,
    request: CanonicalInferenceRequest,
) -> str:
    """Digest the attributes both embodiments are proven to share."""
    return content_digest(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "service_ref": dependency.service_ref,
                "interface": dependency.interface.value,
                "isolation": dependency.isolation,
                "revision": spec.model_revision,
                "engine": spec.model.vllm if spec.model else None,
                "determinism": profile.determinism.value,
                "effect": profile.effect.value,
                "recovery": profile.recovery.value,
                "input_provenance": profile.input_provenance.value,
                "telemetry_exempt": _TELEMETRY_EXEMPT,
            },
            sort_keys=True,
            default=str,
        )
    )
