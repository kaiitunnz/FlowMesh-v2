"""Per-interface egress backends the mediated-egress sidecar dispatches a permit to.

Each backend pairs one interface's request-integrity digest with its provider-execution
surface. The sidecar looks a backend up by the permit's interface, recomputes the digest
for the fence, and runs the egress. Each reads its provider credential only from the
local worker environment.
"""

import logging

from shared.tools.contract import ToolOperationEnvelope, ToolOutcome, ToolOutcomeStatus
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

from ..lifecycle import CapturedRequest
from .fence import ProviderBinding


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
        credential: str | None,
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

    The permit's per-call ``credential`` is a workflow's own pinned model key; a call
    without one falls back to this worker's deployment-global environment key.
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
        credential: str | None,
    ) -> ToolOutcome:
        assert isinstance(request, ModelRequest)
        return self._sidecar.execute(envelope, request, credential or self._env_api_key)

    def complete(
        self,
        envelope: ToolOperationEnvelope,
        request: CapturedRequest,
        credential: str | None,
    ) -> ModelCompletion:
        """Egress a held model turn and return the whole message with its tool calls."""
        assert isinstance(request, ModelRequest)
        key = credential or self._env_api_key
        return self._sidecar.complete(envelope, request, key)


__all__ = ["ModelEgress", "SearchEgress"]
