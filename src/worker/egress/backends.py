"""Per-interface egress backends the mediated-egress sidecar dispatches a permit to.

Each backend pairs one interface's request-integrity digest with its provider-execution
surface. The sidecar looks a backend up by the permit's interface, recomputes the digest
for the fence, and runs the egress. A provider credential comes from the permit or,
where the permit grants it, the local worker environment.
"""

import logging

from shared.tools.contract import (
    MediatedOperationPermit,
    ToolOperationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from shared.tools.model.egress import ExternalModelSidecar
from shared.tools.model.schema import (
    MODEL_INTERFACE,
    ModelCompletion,
    ModelRequest,
    model_request_digest,
)
from shared.tools.search.egress import ExternalToolSidecar
from shared.tools.search.providers import LazySearchProvider
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    ToolRequest,
    tool_request_digest,
)

from .fence import ProviderBinding
from .request_store import CapturedRequest


class SearchEgress:
    """The ``search/v1`` egress backend."""

    interface = SEARCH_INTERFACE

    def __init__(
        self, provider: str, api_key: str | None, logger: logging.Logger
    ) -> None:
        self._sidecar = ExternalToolSidecar(
            LazySearchProvider(ProviderBinding(provider, api_key)), logger
        )
        self._log = logger

    def digest(self, request: CapturedRequest) -> str:
        assert isinstance(request, ToolRequest)
        return tool_request_digest(
            request.interface, request.query, request.max_results
        )

    def execute(
        self,
        envelope: ToolOperationEnvelope,
        request: CapturedRequest,
        permit: MediatedOperationPermit,
    ) -> ToolOutcome:
        assert isinstance(request, ToolRequest)
        try:
            return self._sidecar.execute(envelope, request)
        except ValueError as exc:
            # A misprovisioned provider is a deterministic fault: a typed terminal
            # outcome rather than an ambiguous retry loop.
            self._log.warning("tool provider unavailable: %s", exc)
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the external-tool provider is unavailable",
            )


class ModelEgress:
    """The managed external-``model`` egress backend.

    The key is the permit's per-call ``credential`` (a workflow's own pinned model key),
    or this worker's deployment key when the permit grants it; otherwise the call
    carries no credential.
    """

    interface = MODEL_INTERFACE

    def __init__(self, env_api_key: str | None, logger: logging.Logger) -> None:
        self._sidecar = ExternalModelSidecar(logger)
        self._env_api_key = env_api_key

    def digest(self, request: CapturedRequest) -> str:
        assert isinstance(request, ModelRequest)
        return model_request_digest(request.interface, request.url, request.body)

    def execute(
        self,
        envelope: ToolOperationEnvelope,
        request: CapturedRequest,
        permit: MediatedOperationPermit,
    ) -> ToolOutcome:
        assert isinstance(request, ModelRequest)
        return self._sidecar.execute(envelope, request, self._key(permit))

    def complete(
        self,
        envelope: ToolOperationEnvelope,
        request: CapturedRequest,
        permit: MediatedOperationPermit,
    ) -> ModelCompletion:
        """Egress a held model turn and return the whole message with its tool calls."""
        assert isinstance(request, ModelRequest)
        return self._sidecar.complete(envelope, request, self._key(permit))

    def _key(self, permit: MediatedOperationPermit) -> str | None:
        if permit.credential is not None:
            return permit.credential
        return self._env_api_key if permit.deployment_credential else None


__all__ = ["ModelEgress", "SearchEgress"]
