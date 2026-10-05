"""Compile a local-eligible inference leaf into its embodiment menu.

The menu is emitted only after the two embodiments are proven to run the same declared
contract, so the scheduler chooses between them without changing what the workflow
declared. The proof is narrow on purpose: a leaf whose equivalence rests on anything
this module does not check keeps the single embodiment its binding names.
"""

import json
from typing import Any

from shared.inference import (
    ENGINE_PROFILE_KEYS,
    CanonicalInferenceContract,
    CanonicalProjectionError,
    canonical_contract,
    declares_multiple_prompts,
    unforwarded_inference_keys,
)
from shared.tasks.credentials import credential_pointer
from shared.tasks.placeholders import contains_placeholder
from shared.tasks.specs import (
    InferenceBackend,
    InferenceEmbodimentKind,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    TaskSpecBase,
)
from shared.tasks.specs.common import ModelSpecStrict, ModelSpecTemplate
from shared.utils.redact import is_credential_key

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
    contract = _canonical_contract(task, spec)
    resource_class = profile.binding.task_type.value
    resident = InferenceEmbodimentCandidate(
        alternative_id=f"{node_id}:{InferenceEmbodimentKind.RESIDENT_SERVED.value}",
        kind=InferenceEmbodimentKind.RESIDENT_SERVED,
        episode=EpisodeSpec(
            boundary=EpisodeBoundaryKind.SERVICE_ISSUE, resource_class=resource_class
        ),
        service_family_requirement=dependency.family_requirement(),
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
        contract_fingerprint=_contract_fingerprint(spec, dependency, profile, contract),
        primary=primary.alternative_id,
        candidates=(resident, local),
        max_batch_size=dependency.max_batch_size,
    )


def _canonical_contract(
    task: ParsedTask, spec: InferenceSpecStrict | InferenceSpecTemplate
) -> CanonicalInferenceContract:
    try:
        return canonical_contract(spec)
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
    # An enforce_cpu that renders from upstream may yet move the leaf off vLLM.
    if (
        spec.enforce_cpu not in (None, False)
        or spec.backend() is not InferenceBackend.VLLM
    ):
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
        canonical_contract(spec)
    except CanonicalProjectionError as exc:
        return f"its request projection is not shared: {exc}"
    return None


def replica_unfit_reason(
    task: ParsedTask, spec: InferenceSpecStrict | InferenceSpecTemplate
) -> str | None:
    """Return why a resident replica cannot run a leaf as declared, or ``None``.

    A replica runs with the deployment's own model access, serves the base model, and
    is chosen before any upstream value is known, so a leaf whose engine configuration
    depends on any of these runs as declared only self-contained.
    """
    vllm = (spec.model.vllm if spec.model is not None else None) or {}
    profiled = [vllm.get(key) for key in ENGINE_PROFILE_KEYS]
    if contains_placeholder(
        [*profiled, vllm.get("tensor_parallel_size")]
    ) or contains_placeholder(spec.model_revision):
        return "its engine configuration renders from upstream at dispatch"
    if _engine_credential(task, vllm):
        return "its engine configuration carries a credential"
    if _checkpoint_load(spec) is not None:
        return "it loads a checkpoint in place of its model"
    return None


def reject_unproven(
    task: ParsedTask, spec: InferenceSpecStrict | InferenceSpecTemplate
) -> None:
    """Fail a leaf that explicitly asked for a menu this module cannot prove."""
    if (
        reason := unproven_reason(spec) or replica_unfit_reason(task, spec)
    ) is not None:
        raise _unproven(task, reason)


def reject_resident_checkpoint(task: ParsedTask, spec: TaskSpecBase) -> None:
    """Fail a resident-served leaf that loads a checkpoint in place of its model.

    A replica serves the base model it materializes, so it would answer the leaf with
    another model than the one the checkpoint holds.
    """
    if (
        isinstance(spec, (InferenceSpecStrict, InferenceSpecTemplate))
        and _checkpoint_load(spec) is not None
    ):
        raise _reject(
            task,
            "embodiment.resident-checkpoint",
            "a resident replica serves the base model, not a checkpoint the leaf "
            "loads; serve the leaf self-contained",
        )


def reject_resident_batch(
    task: ParsedTask, spec: TaskSpecBase, eligibility: InferenceEmbodimentEligibility
) -> None:
    """Fail a resident-served leaf that declares several prompts it cannot project.

    A batch is served from the contract its leaf projects into: the contract carries
    every conversation on one boundary and names the result they report. A leaf whose
    request does not project has no such contract, so serving its first prompt alone
    would lose the rest silently. A leaf that does project is served, whether its
    embodiment is pinned or chosen from a menu. An embedding leaf embeds a list of
    inputs in one request and is unaffected.

    NOTE: a non-projectable resident batch could be served later by building the
    fan-out from the spec directly and reporting a native batch result, rather than
    the canonical projection. It is refused here because it has no equivalence
    contract to project, not because a replica cannot serve it.
    """
    if eligibility is not InferenceEmbodimentEligibility.RESIDENT_REQUIRED:
        return
    if not isinstance(spec, (InferenceSpecStrict, InferenceSpecTemplate)):
        return
    if not declares_multiple_prompts(spec):
        return
    if (reason := unproven_reason(spec)) is None:
        return
    raise _reject(
        task,
        "embodiment.resident-batch-unserved",
        f"a resident-served inference leaf serves several prompts as the batch its "
        f"contract projects; here {reason}. Declare one prompt, or declare a leaf "
        f"whose request projects",
    )


def credentialed_source(task: ParsedTask, spec: TaskSpecBase) -> str | None:
    """The source a resident replica would load that names a vaulted credential.

    A replica loads the model and its adapter from the sources the plan carries, and a
    vaulted source reaches the plan masked, so such a leaf has no resident embodiment.
    """
    if not isinstance(spec, (ModelSpecStrict, ModelSpecTemplate)):
        return None
    sources = {credential_pointer(("model", "source", "identifier")): "model"}
    if spec.adapters:
        adapter = spec.adapters[0]
        if field := "path" if adapter.path else "url" if adapter.url else None:
            sources[credential_pointer(("model", "adapters", 0, field))] = "adapter"
    return next(
        (
            name
            for pointer, name in sources.items()
            if pointer in task.masked_credentials
        ),
        None,
    )


def reject_credentialed_source(
    task: ParsedTask, spec: TaskSpecBase, eligibility: InferenceEmbodimentEligibility
) -> None:
    """Fail a leaf that admits resident serving of a model or adapter whose source
    carries a credential."""
    if eligibility is InferenceEmbodimentEligibility.SELF_CONTAINED_REQUIRED:
        return
    if (source := credentialed_source(task, spec)) is not None:
        raise _reject(
            task,
            "embodiment.resident-source-credential",
            f"a resident replica loads the leaf's {source} from a source that "
            f"cannot carry a credential; serve the leaf self-contained, or load the "
            f"{source} from a source without one",
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


def _engine_credential(task: ParsedTask, vllm: dict[str, Any]) -> bool:
    engine = credential_pointer(("model", "vllm")) + "/"
    environment = vllm.get("env_vars")
    return (
        bool(vllm.get("hf_token"))
        or any(pointer.startswith(engine) for pointer in task.masked_credentials)
        or (
            isinstance(environment, dict)
            and any(is_credential_key(str(name)) for name in environment)
        )
    )


def _checkpoint_load(spec: InferenceSpecStrict | InferenceSpecTemplate) -> Any:
    return spec.checkpoint.get("load") if isinstance(spec.checkpoint, dict) else None


def _declared_gpu_count(
    spec: InferenceSpecStrict | InferenceSpecTemplate,
) -> int | None:
    gpu = spec.gpu_requirements()
    return gpu.count if gpu else None


def _contract_fingerprint(
    spec: InferenceSpecStrict | InferenceSpecTemplate,
    dependency: ServiceDependency,
    profile: LeafProfile,
    contract: CanonicalInferenceContract,
) -> str:
    """Digest the attributes both embodiments are proven to share.

    What they share is the contract: the source they resolve their prompts from and the
    request they build around it. A leaf sourcing its prompts from upstream has no
    prompt vector at compile time, so the contract carries that shared identity.
    """
    return content_digest(
        json.dumps(
            {
                "contract": contract.model_dump(mode="json"),
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
