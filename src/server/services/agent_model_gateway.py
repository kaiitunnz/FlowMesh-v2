"""The FlowMesh agent-model gateway for mediated model invocations.

A managed model request an agent defers becomes a durable invocation the fabric settles
here rather than a raw resident-engine call: the gateway carries orchestration context,
runs the configured upstream, and injects the result back at the originating call. It
implements the OpenAI Responses API so a harness whose provider targets it (Codex, say)
crosses the same seam; a generic endpoint without this conversion is not valid.

Resident-capacity admission is not part of this path: the invocation settles directly.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..config import AgentModelGatewayConfig


class _EpisodeSettler(Protocol):
    def settle_episode_invocation(
        self, task_id: str, call_correlation: str, value: str | None
    ) -> bool: ...


class ResponsesRequest(BaseModel):
    """The minimal OpenAI Responses API request the gateway accepts."""

    model: str | None = None
    input: Any = None


class AgentModelGateway:
    """Settle a mediated model invocation through a configurable upstream.

    The settle path serves a boundary an adapter defers server-side; the Responses
    API router serves a harness whose provider targets the gateway directly. Both
    convert the request the same way and run one upstream, so a canned settle in a test
    and a provider forward in production share one binding.
    """

    def __init__(
        self,
        settler: _EpisodeSettler,
        config: AgentModelGatewayConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settler = settler
        self._cfg = config
        self._logger = logger or logging.getLogger("agent-model-gateway")
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="agent-model-gateway"
        )

    def settle(self, task_id: str, call_correlation: str, payload: str | None) -> None:
        """Settle a suspended model boundary off the caller's lane, never inline."""
        self._executor.submit(self._settle, task_id, call_correlation, payload)

    def _settle(self, task_id: str, call_correlation: str, payload: str | None) -> None:
        try:
            value = self.invoke(payload)
        except Exception as exc:
            self._logger.warning("agent-model gateway upstream failed: %s", exc)
            value = None
        self._settler.settle_episode_invocation(task_id, call_correlation, value)

    def invoke(self, payload: str | None) -> str:
        """Run the configured upstream over a model request, returning its text."""
        prompt = _extract_prompt(payload)
        if self._cfg.mode == "echo":
            return prompt
        if self._cfg.mode == "openai":
            return self._forward_openai(prompt)
        return f"canned-response:{prompt}" if prompt else "canned-response"

    def _forward_openai(self, prompt: str) -> str:
        if not self._cfg.url:
            raise RuntimeError("openai gateway mode needs a configured upstream url")
        parsed = urlparse(self._cfg.url)
        if parsed.scheme not in ("http", "https"):
            raise RuntimeError(f"unsupported gateway url scheme {parsed.scheme!r}")
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        body = {
            "model": self._cfg.model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            f"{self._cfg.url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=self._cfg.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def responses(self, request: ResponsesRequest) -> dict[str, Any]:
        """Convert one OpenAI Responses request and return its settled output."""
        text = self.invoke(_dump_input(request.input))
        return {
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        }


def _extract_prompt(payload: str | None) -> str:
    if not payload:
        return ""
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload
    if isinstance(parsed, dict):
        for key in ("prompt", "input", "content"):
            if isinstance(value := parsed.get(key), str):
                return value
    return payload


def _dump_input(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def build_agent_model_router(gateway: AgentModelGateway) -> APIRouter:
    """The Responses API surface a harness provider targets."""
    router = APIRouter()

    @router.post("/v1/responses")
    async def create_response(
        body: ResponsesRequest, request: Request
    ) -> dict[str, Any]:
        return gateway.responses(body)

    return router
