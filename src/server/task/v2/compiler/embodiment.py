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
    declares_multiple_prompts,
    unforwarded_inference_keys,
)
from shared.tasks.specs import (
    InferenceBackend,
    InferenceEmbodimentKind,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    TaskSpecBase,
)

from ...parser import ParsedTask
from ..representations.operators import (
    InferenceEmbodimentEligibility,
    LeafProfile,
    ServiceDependency,
)
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


def embodiment_menu(
    task: ParsedTask,
    spec: InferenceSpecStrict | InferenceSpecTemplate,
    dependency: ServiceDependency,
    profile: LeafProfile,
    node_id: str,
) -> InferenceEmbodimentMenu:
    """Emit the menu of a leaf ``unproven_reason`` has already cleared.

    The attributes digested into the fingerprint are the ones proven equal. The per-item
    fields a local generation reports and a relayed engine response cannot — see
    ``PROJECTION_DROPS`` — are outside the proof because the shared result projection
    drops them from both embodiments rather than letting one carry them.

    The primary is the resident-served embodiment unless the binding names the other:
    a leaf declares one model contract, and which capacity serves it is the fabric's to
    decide from what a deployment actually runs.
    """
    request = _canonical_request(task, spec)
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
    declared = spec.service.primary if spec.service else None
    primary = by_kind[declared] if declared else resident
    return InferenceEmbodimentMenu(
        contract_fingerprint=_contract_fingerprint(spec, dependency, profile, request),
        primary=primary.alternative_id,
        candidates=(resident, local),
        batch_size=len(request.prompts),
    )


def _canonical_request(
    task: ParsedTask, spec: InferenceSpecStrict | InferenceSpecTemplate
) -> CanonicalInferenceRequest:
    try:
        return canonical_request(spec)
    except CanonicalProjectionError as exc:
        raise _unproven(task, f"its request projection is not shared: {exc}") from exc


def unproven_reason(
    spec: InferenceSpecStrict | InferenceSpecTemplate,
) -> str | None:
    """Why this module cannot prove a leaf's embodiments equivalent, or ``None``.

    A resident replica serves a pinned vLLM engine, so a local embodiment is proven only
    for a leaf that pins the same engine. An adapter, a shard, a parallel split, or a
    postprocessing step changes what one embodiment produces relative to the other, and
    a request the two do not read identically is unprojectable.
    """
    if spec.enforce_cpu is True or spec.backend() is not InferenceBackend.VLLM:
        return (
            "it does not pin the vLLM engine a resident replica serves; declare "
            "model.vllm and no enforce_cpu"
        )
    if spec.adapters:
        return "adapter serving is proven for one embodiment only"
    if spec.parallel is not None or spec.shard is not None:
        return "a sharded or parallel split applies to one embodiment"
    if spec.postprocess is not None:
        return "postprocessing applies to one embodiment"
    if unforwarded := unforwarded_inference_keys(spec):
        return (
            f"spec.inference declares {', '.join(unforwarded)}, which a local "
            "generation applies and a relayed request does not carry"
        )
    try:
        canonical_request(spec)
    except CanonicalProjectionError as exc:
        return f"its request projection is not shared: {exc}"
    return None


def reject_unproven(
    task: ParsedTask, spec: InferenceSpecStrict | InferenceSpecTemplate
) -> None:
    """Fail a leaf that explicitly asked for a menu this module cannot prove."""
    if (reason := unproven_reason(spec)) is not None:
        raise _unproven(task, reason)


def reject_resident_batch(
    task: ParsedTask, spec: TaskSpecBase, eligibility: InferenceEmbodimentEligibility
) -> None:
    """Fail a leaf served from a replica without a menu that declares several prompts.

    A replica serves one conversation per chat request, so several prompts are served by
    the batch a menu compiles and not otherwise. Failing names what the leaf declared,
    where serving its first prompt alone would lose the rest silently. An embedding leaf
    embeds a list of inputs in one request and is unaffected.
    """
    if eligibility is not InferenceEmbodimentEligibility.RESIDENT_REQUIRED:
        return
    if not isinstance(spec, (InferenceSpecStrict, InferenceSpecTemplate)):
        return
    if not declares_multiple_prompts(spec):
        return
    raise _reject(
        task,
        "embodiment.resident-batch-unserved",
        "a resident-served inference leaf runs one prompt; declare one prompt, or "
        "leave the binding mode undeclared so the leaf admits both embodiments",
    )


def _unproven(task: ParsedTask, reason: str) -> Exception:
    return _reject(
        task,
        "embodiment.not-contract-equivalent",
        f"a local_eligible inference leaf admits both embodiments only when they run "
        f"one proven contract; here {reason}",
    )


def _reject(task: ParsedTask, code: str, message: str) -> Exception:
    source_kind, source_id = (
        ("graph_node", task.graph_node_name)
        if task.graph_node_name
        else ("stage", task.local_name) if task.local_name else ("legacy", task.task_id)
    )
    return compile_error(code, message, source_id or task.task_id, source_kind)


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
            },
            sort_keys=True,
            default=str,
        )
    )
