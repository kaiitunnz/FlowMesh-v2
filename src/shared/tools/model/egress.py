"""The provider-egress surface for a managed external-model boundary.

It egresses only within a server-issued ``ToolOperationEnvelope``, refusing an interface
it does not serve, and maps a provider fault to a typed ``ToolOutcome``. It runs in the
worker's mediated-egress sidecar, the process that actually egresses, and reads the
provider credential only from its local worker environment.
"""

import logging

import requests

from ..contract import ToolOperationEnvelope, ToolOutcome, ToolOutcomeStatus
from .schema import MODEL_INTERFACE, ModelRequest

_SERVED_INTERFACES = frozenset({MODEL_INTERFACE})


class ExternalModelSidecar:
    """The surface that performs external-model egress under an envelope."""

    def __init__(
        self, api_key: str | None, logger: logging.Logger | None = None
    ) -> None:
        self._api_key = api_key
        self._log = logger or logging.getLogger("external-model-sidecar")

    def execute(
        self, envelope: ToolOperationEnvelope, request: ModelRequest
    ) -> ToolOutcome:
        if envelope.interface not in _SERVED_INTERFACES:
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value=f"the sidecar serves no interface {envelope.interface!r}",
            )
        if request.interface != envelope.interface:
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the request interface is outside the issued envelope",
            )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        try:
            response = requests.post(
                f"{request.url.rstrip('/')}/chat/completions",
                json=body,
                headers=headers,
                timeout=envelope.timeout_sec,
            )
            response.raise_for_status()
            content = str(response.json()["choices"][0]["message"]["content"])
        except requests.Timeout:
            return ToolOutcome(
                status=ToolOutcomeStatus.TIMEOUT, value="the model request timed out"
            )
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            self._log.warning("external-model egress failed: %s", exc)
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the model provider was unreachable",
            )
        return ToolOutcome(
            status=ToolOutcomeStatus.SUCCESS, value=content[: envelope.result_char_cap]
        )


__all__ = ["ExternalModelSidecar"]
