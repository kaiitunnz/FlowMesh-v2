from dataclasses import dataclass

from shared.tasks.specs import AgentHarnessSpec, AgentModelBindingSpec, ModelBindingMode

from ..representations.operators import (
    AgentHarnessBinding,
    AgentModelGatewayBinding,
    BindingProvenance,
    HarnessBindingProvenance,
    ModelBindingProvenance,
)

# The safe hardcoded fallback model for an otherwise-complete external binding.
_FALLBACK_OPENAI_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class AgentBindingDefaults:
    """Deployment-resolved defaults injected into compilation from the config edge.

    Never read from the environment here: the compiler receives resolved values so
    the effective binding a template pins reflects submission time, not a later
    environment change.
    """

    default_backend: str | None = None
    default_version: str | None = None
    default_mode: ModelBindingMode | None = None
    default_url: str | None = None
    default_model: str | None = None


def neutral_defaults() -> AgentBindingDefaults:
    """Defaults with no deployment harness/model configuration.

    A bare agent then resolves no harness binding and a ``canned`` model binding,
    which validation rejects for the missing harness.
    """
    return AgentBindingDefaults()


def service_family_for_ref(ref: str) -> str:
    """The canonical service family a resident model reference belongs to.

    The reference is the model identifier (a served name or repository id), and the
    family is derived from it deterministically so identical references group into one
    demand family for downstream residency scheduling.
    """
    return ref.strip()


def _resolve_harness(
    harness: AgentHarnessSpec | None, defaults: AgentBindingDefaults
) -> AgentHarnessBinding | None:
    if harness is not None:
        return AgentHarnessBinding(
            backend=harness.backend,
            version=harness.version,
            params=dict(harness.params),
            provenance=HarnessBindingProvenance(
                backend=BindingProvenance.SOURCE, version=BindingProvenance.SOURCE
            ),
        )
    if defaults.default_backend is None:
        return None
    version_from_default = defaults.default_version is not None
    return AgentHarnessBinding(
        backend=defaults.default_backend,
        version=defaults.default_version or "v1",
        provenance=HarnessBindingProvenance(
            backend=BindingProvenance.DEFAULT,
            version=(
                BindingProvenance.DEFAULT
                if version_from_default
                else BindingProvenance.FALLBACK
            ),
        ),
    )


def _source_model(
    harness: AgentHarnessSpec | None, model_binding: AgentModelBindingSpec | None
) -> AgentModelBindingSpec | None:
    if model_binding is not None:
        return model_binding
    # Compat sugar: a codex-style harness that declares base_url/model in its params
    # normalizes to an external openai binding.
    if harness is not None:
        base_url, model = harness.params.get("base_url"), harness.params.get("model")
        if isinstance(base_url, str) and isinstance(model, str):
            return AgentModelBindingSpec(
                mode=ModelBindingMode.OPENAI, url=base_url, model=model
            )
    return None


def _effective_source_mode(source: AgentModelBindingSpec) -> ModelBindingMode | None:
    if source.mode is not None:
        return source.mode
    if source.service_model_ref:
        return ModelBindingMode.RESIDENT
    if source.url or source.model:
        return ModelBindingMode.OPENAI
    return None


def _resolve_model(
    source: AgentModelBindingSpec | None,
    defaults: AgentBindingDefaults,
    secret_ref: str | None,
) -> AgentModelGatewayBinding:
    source_mode = _effective_source_mode(source) if source is not None else None
    if source_mode is not None:
        mode, mode_prov = source_mode, BindingProvenance.SOURCE
    elif defaults.default_mode is not None:
        mode, mode_prov = defaults.default_mode, BindingProvenance.DEFAULT
    else:
        mode, mode_prov = ModelBindingMode.CANNED, BindingProvenance.FALLBACK

    if mode is ModelBindingMode.RESIDENT:
        return AgentModelGatewayBinding(
            mode=mode,
            service_model_ref=source.service_model_ref if source else None,
            provenance=ModelBindingProvenance(
                mode=mode_prov,
                url=BindingProvenance.SOURCE,
                model=BindingProvenance.SOURCE,
            ),
        )
    if mode in (ModelBindingMode.CANNED, ModelBindingMode.ECHO):
        return AgentModelGatewayBinding(
            mode=mode,
            provenance=ModelBindingProvenance(
                mode=mode_prov,
                url=BindingProvenance.SOURCE,
                model=BindingProvenance.SOURCE,
            ),
        )

    url, url_prov = _resolve_field(source.url if source else None, defaults.default_url)
    model, model_prov = _resolve_field(
        source.model if source else None, defaults.default_model
    )
    if model is None and url is not None:
        model, model_prov = _FALLBACK_OPENAI_MODEL, BindingProvenance.FALLBACK
    return AgentModelGatewayBinding(
        mode=mode,
        url=url,
        model=model,
        secret_ref=secret_ref,
        provenance=ModelBindingProvenance(
            mode=mode_prov, url=url_prov, model=model_prov
        ),
    )


def _resolve_field(
    source_value: str | None, default_value: str | None
) -> tuple[str | None, BindingProvenance]:
    if source_value is not None:
        return source_value, BindingProvenance.SOURCE
    if default_value is not None:
        return default_value, BindingProvenance.DEFAULT
    return None, BindingProvenance.FALLBACK


def resolve_agent_bindings(
    harness: AgentHarnessSpec | None,
    model_binding: AgentModelBindingSpec | None,
    defaults: AgentBindingDefaults,
    secret_ref: str | None = None,
) -> tuple[AgentHarnessBinding | None, AgentModelGatewayBinding]:
    """Resolve the effective, submission-pinned harness and model bindings.

    Resolution is best-effort and never fails: an unresolved harness stays ``None``
    and an incomplete external binding keeps a ``None`` url. Validation turns those
    into diagnostics so a dry-run and a submission agree. ``secret_ref`` is the vault
    ref minted for an inline credential at submission, pinned onto the model binding.
    """
    return (
        _resolve_harness(harness, defaults),
        _resolve_model(_source_model(harness, model_binding), defaults, secret_ref),
    )
