"""The provider-egress surface for a managed external-model boundary.

It egresses only within a server-issued ``ToolOperationEnvelope``, refusing an interface
it does not serve, and maps a provider fault to a typed ``ToolOutcome``. It runs in the
worker's mediated-egress sidecar, the process that actually egresses, and reads the
provider credential only from its local worker environment.
"""

import logging
from typing import Any

import requests

from ..contract import ToolOperationEnvelope, ToolOutcome, ToolOutcomeStatus
from .schema import MODEL_INTERFACE, ModelCompletion, ModelRequest, ModelToolCall

_SERVED_INTERFACES = frozenset({MODEL_INTERFACE})


class ModelEgressError(RuntimeError):
    """A provider fault on a held model turn: a terminal, non-retryable turn failure."""


class ExternalModelSidecar:
    """The surface that performs external-model egress under an envelope.

    The credential is supplied per call — the workflow's own pinned key carried on the
    permit, or the worker's deployment-global fallback — never held on the surface.
    ``execute`` renders a text outcome for a deferred boundary; ``complete`` returns the
    model's whole message for a held turn that must surface its tool calls.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger("external-model-sidecar")

    def execute(
        self,
        envelope: ToolOperationEnvelope,
        request: ModelRequest,
        api_key: str | None,
    ) -> ToolOutcome:
        if (unserved := self._unserved(envelope, request)) is not None:
            return ToolOutcome(status=ToolOutcomeStatus.UNAVAILABLE, value=unserved)
        try:
            data = self._chat(request, api_key, envelope.timeout_sec)
            content = str(data["choices"][0]["message"]["content"])
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

    def complete(
        self,
        envelope: ToolOperationEnvelope,
        request: ModelRequest,
        api_key: str | None,
    ) -> ModelCompletion:
        """The model's whole message for a held turn, else a ``ModelEgressError``.

        A held turn cannot proceed without its model reply, so a timeout, an unreachable
        provider, or an unparsable response raises rather than return an empty message.
        """
        if (unserved := self._unserved(envelope, request)) is not None:
            raise ModelEgressError(unserved)
        try:
            message = self._chat(request, api_key, envelope.timeout_sec)["choices"][0][
                "message"
            ]
        except requests.Timeout as exc:
            raise ModelEgressError("the model request timed out") from exc
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            self._log.warning("external-model egress failed: %s", exc)
            raise ModelEgressError("the model provider was unreachable") from exc
        content = message.get("content")
        return ModelCompletion(
            content=str(content) if content is not None else "",
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
        )

    @staticmethod
    def _unserved(envelope: ToolOperationEnvelope, request: ModelRequest) -> str | None:
        if envelope.interface not in _SERVED_INTERFACES:
            return f"the sidecar serves no interface {envelope.interface!r}"
        if request.interface != envelope.interface:
            return "the request interface is outside the issued envelope"
        return None

    def _chat(
        self, request: ModelRequest, api_key: str | None, timeout_sec: float
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        response = requests.post(
            f"{request.url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        return dict(response.json())


def _parse_tool_calls(raw: Any) -> tuple[ModelToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[ModelToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if not isinstance(fn, dict):
            continue
        calls.append(
            ModelToolCall(
                call_id=str(item.get("id", "")),
                name=str(fn.get("name", "")),
                arguments=str(fn.get("arguments", "")),
            )
        )
    return tuple(calls)


__all__ = ["ExternalModelSidecar", "ModelEgressError"]
