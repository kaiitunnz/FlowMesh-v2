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
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
import requests
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import AgentModelGatewayConfig, GatewayMode


class _EpisodeSettler(Protocol):
    def settle_episode_invocation(
        self,
        task_id: str,
        call_correlation: str,
        value: str | None,
        *,
        error: str | None = None,
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
            # A failed upstream fails the boundary rather than resuming the agent with a
            # phantom empty success.
            self._logger.warning("agent-model gateway upstream failed: %s", exc)
            self._settler.settle_episode_invocation(
                task_id, call_correlation, None, error=str(exc)
            )
            return
        self._settler.settle_episode_invocation(task_id, call_correlation, value)

    def shutdown(self) -> None:
        """Stop accepting settles and release the off-lane executor."""
        self._executor.shutdown(wait=False)

    def invoke(self, payload: str | None) -> str:
        """Run the configured upstream over a model request, returning its text."""
        prompt = _extract_prompt(payload)
        if self._cfg.mode is GatewayMode.ECHO:
            return prompt
        if self._cfg.mode is GatewayMode.OPENAI:
            return self._forward_openai(prompt)
        if self._cfg.mode is GatewayMode.PROXY:
            return self._forward_responses(prompt)
        return f"canned-response:{prompt}" if prompt else "canned-response"

    def is_proxy(self) -> bool:
        """Whether a harness's own model turns proxy to an upstream Responses API."""
        return self._cfg.mode is GatewayMode.PROXY

    def _forward_openai(self, prompt: str) -> str:
        body = {
            "model": self._cfg.model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            f"{self._upstream_base()}/chat/completions",
            json=body,
            headers=self._headers(),
            timeout=self._cfg.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def _forward_responses(self, prompt: str) -> str:
        body = {"model": self._cfg.model or "model", "input": prompt, "stream": False}
        response = requests.post(
            f"{self._upstream_base()}/responses",
            json=body,
            headers=self._headers(),
            timeout=self._cfg.timeout_sec,
        )
        response.raise_for_status()
        return _response_text(response.json())

    async def proxy_responses(self, body: dict[str, Any]) -> StreamingResponse:
        """Stream a harness's own model turn from the upstream Responses API verbatim.

        Codex issues its reasoning turns at this surface; the request forwards to the
        upstream ``/v1/responses`` and its SSE streams straight back, so a live model
        drives the agent. The mediated facade a harness defers settles off this path.
        """
        url = f"{self._upstream_base()}/responses"
        forward = {**body, "stream": True, "tools": _upstream_tools(body.get("tools"))}
        if self._cfg.model:
            forward["model"] = self._cfg.model
        headers, timeout = self._headers(), self._cfg.timeout_sec

        async def _stream() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, json=forward, headers=headers
                ) as upstream:
                    upstream.raise_for_status()
                    async for chunk in upstream.aiter_raw():
                        yield chunk

        return StreamingResponse(_stream(), media_type="text/event-stream")

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

    def _upstream_base(self) -> str:
        if not self._cfg.url:
            raise RuntimeError("this gateway mode needs a configured upstream url")
        parsed = urlparse(self._cfg.url)
        if parsed.scheme not in ("http", "https"):
            raise RuntimeError(f"unsupported gateway url scheme {parsed.scheme!r}")
        return self._cfg.url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        return headers


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


def _upstream_tools(tools: Any) -> list[Any]:
    """The tools an upstream Responses API accepts, dropping harness-specific types.

    A standard Responses backend accepts the ``function`` tool; a harness may advertise
    its own types (a namespace tool for native sub-agents) that the upstream rejects
    and the fabric mediates through a facade instead.
    """
    if not isinstance(tools, list):
        return []
    return [t for t in tools if isinstance(t, dict) and t.get("type") == "function"]


def _response_text(data: Any) -> str:
    """The assistant text of a Responses object, ignoring reasoning and tool items."""
    if not isinstance(data, dict):
        return ""
    for item in data.get("output", []):
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") == "output_text":
                    return str(part.get("text", ""))
    return ""


def build_agent_model_router(gateway: AgentModelGateway) -> APIRouter:
    """The Responses API surface a harness provider targets."""
    router = APIRouter()

    @router.post("/v1/responses")
    async def create_response(request: Request) -> Any:
        body = await request.json()
        if gateway.is_proxy():
            return await gateway.proxy_responses(body)
        return gateway.responses(ResponsesRequest.model_validate(body))

    return router
