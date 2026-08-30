"""The FlowMesh agent-model gateway for mediated model invocations.

A managed model request an agent defers becomes a durable invocation the fabric settles
here rather than a raw resident-engine call: the gateway carries orchestration context,
runs the configured upstream, and injects the result back at the originating call. It
implements the OpenAI Responses API so a harness whose provider targets it (Codex, say)
crosses the same seam; a generic endpoint without this conversion is not valid.

Resident-capacity admission is not part of this path: the invocation settles directly.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shared.harness import BoundaryEventKind, BoundaryRequest
from shared.tasks.specs import ModelBindingMode

from ..config import AgentModelGatewayConfig, GatewayMode
from ..orchestration.tool_dispatch import (
    SEARCH_INTERFACE,
    FacadeBatchMember,
    ToolInvocationEnvelope,
)
from ..task.v2.representations.operators import (
    AgentModelGatewayBinding,
    FacadeDescriptor,
)
from .model_secret_vault import ModelSecretVault

_INJECT_CALL_PREFIX = "fab-"

_BINDING_MODE_TO_GATEWAY = {
    ModelBindingMode.CANNED: GatewayMode.CANNED,
    ModelBindingMode.ECHO: GatewayMode.ECHO,
    ModelBindingMode.OPENAI: GatewayMode.OPENAI,
}


@dataclass(frozen=True)
class ResolvedGatewayBinding:
    """The effective per-invocation upstream a mediated model request resolves to.

    Resolved server-side from the pinned binding: ``api_key`` is materialized from a
    ``secret_ref`` here and never leaves the server-to-upstream path.
    """

    mode: GatewayMode
    url: str | None = None
    model: str | None = None
    api_key: str | None = None


GatewayBindingResolver = Callable[[str], ResolvedGatewayBinding | None]


class ResidentBindingNotServable(RuntimeError):
    """A resident model binding needs capacity admission the external gateway lacks."""


def to_gateway_binding(
    pinned: AgentModelGatewayBinding,
    vault: ModelSecretVault,
    workflow_id: str,
) -> ResolvedGatewayBinding:
    """Map a pinned model binding to its effective upstream, resolving the credential.

    The credential is the workflow's own inline key, vaulted at submission under its
    workflow and named here by the generated ``secret_ref``; it resolves only within
    that workflow. Without a resolvable ref the upstream is unauthenticated. A resident
    binding is not served by the external gateway.
    """
    mode = _BINDING_MODE_TO_GATEWAY.get(pinned.mode)
    if mode is None:
        raise ResidentBindingNotServable(
            f"model binding mode {pinned.mode.value!r} is not served externally"
        )
    secret = vault.resolve(workflow_id, pinned.secret_ref)
    api_key = secret.get_secret_value() if secret is not None else None
    return ResolvedGatewayBinding(
        mode=mode, url=pinned.url, model=pinned.model, api_key=api_key
    )


BoundaryOriginator = Callable[[str, BoundaryRequest], None]
# Originates a turn-scoped facade batch (batch id + ordered members) captured on a turn.
BatchOriginator = Callable[[str, str, list[FacadeBatchMember]], None]
# True when a facade boundary/batch is already outstanding for the episode (the fence).
FacadeFence = Callable[[str], bool]
# Resolves the facade tools the gateway may inject for one agent, keyed by its task id;
# a call to one is captured server-side and originated as a fabric boundary.
FacadeResolver = Callable[[str], list[FacadeDescriptor]]


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
        self._originator: BoundaryOriginator | None = None
        self._batch_originator: BatchOriginator | None = None
        self._fence: FacadeFence | None = None
        self._binding_resolver: GatewayBindingResolver | None = None
        self._facade_resolver: FacadeResolver | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="agent-model-gateway"
        )

    def set_boundary_originator(self, originator: BoundaryOriginator) -> None:
        """Install the sink for facade boundaries captured on an episode turn."""
        self._originator = originator

    def set_binding_resolver(self, resolver: GatewayBindingResolver) -> None:
        """Install the per-invocation upstream resolver keyed by episode task id."""
        self._binding_resolver = resolver

    def set_facade_resolver(self, resolver: FacadeResolver) -> None:
        """Install the per-agent facade resolver keyed by episode task id."""
        self._facade_resolver = resolver

    def set_batch_originator(self, originator: BatchOriginator) -> None:
        """Install the sink for a turn-scoped facade batch captured on a turn."""
        self._batch_originator = originator

    def set_facade_fence(self, fence: FacadeFence) -> None:
        """Install the fence: whether a facade group is already outstanding."""
        self._fence = fence

    def _facades_for(self, task_id: str) -> list[FacadeDescriptor]:
        if self._facade_resolver is None:
            return []
        return self._facade_resolver(task_id)

    def _effective(self, task_id: str | None) -> ResolvedGatewayBinding:
        """The upstream for this invocation: the pinned binding, else the default.

        The request body never selects the upstream; only the activation's pinned
        binding (or, off an episode, the deployment default) does.
        """
        if task_id is not None and self._binding_resolver is not None:
            if (resolved := self._binding_resolver(task_id)) is not None:
                return resolved
        return ResolvedGatewayBinding(
            mode=self._cfg.mode,
            url=self._cfg.url,
            model=self._cfg.model,
        )

    def settle(self, env: ToolInvocationEnvelope) -> None:
        """Settle a suspended model boundary off the caller's lane, never inline."""
        self._executor.submit(
            self._settle, env.task_id, env.call_correlation, env.request_payload
        )

    def _settle(self, task_id: str, call_correlation: str, payload: str | None) -> None:
        try:
            value = self.invoke(payload, task_id)
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

    def invoke(self, payload: str | None, task_id: str | None = None) -> str:
        """Run this invocation's upstream over a model request, returning its text."""
        binding = self._effective(task_id)
        prompt = _extract_prompt(payload)
        if binding.mode is GatewayMode.ECHO:
            return prompt
        if binding.mode is GatewayMode.OPENAI:
            return self._forward_openai(prompt, binding)
        if binding.mode is GatewayMode.PROXY:
            return self._forward_responses(prompt, binding)
        return f"canned-response:{prompt}" if prompt else "canned-response"

    def is_proxy(self) -> bool:
        """Whether a harness's own model turns proxy to an upstream Responses API."""
        return self._cfg.mode is GatewayMode.PROXY

    def _forward_openai(self, prompt: str, binding: ResolvedGatewayBinding) -> str:
        body = {
            "model": binding.model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            f"{self._upstream_base(binding)}/chat/completions",
            json=body,
            headers=self._headers(binding),
            timeout=self._cfg.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def _forward_responses(self, prompt: str, binding: ResolvedGatewayBinding) -> str:
        body = {"model": binding.model or "model", "input": prompt, "stream": False}
        response = requests.post(
            f"{self._upstream_base(binding)}/responses",
            json=body,
            headers=self._headers(binding),
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
        binding = self._effective(None)
        url = f"{self._upstream_base(binding)}/responses"
        forward = {**body, "stream": True, "tools": _upstream_tools(body.get("tools"))}
        forward.pop("model", None)
        if binding.model:
            forward["model"] = binding.model
        headers, timeout = self._headers(binding), self._cfg.timeout_sec

        async def _stream() -> AsyncIterator[bytes]:
            # The response status is already sent once streaming starts, so an upstream
            # failure ends the stream gracefully rather than raising through the ASGI
            # layer; the harness observes a truncated turn and fails cleanly.
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", url, json=forward, headers=headers
                    ) as upstream:
                        upstream.raise_for_status()
                        async for chunk in upstream.aiter_raw():
                            yield chunk
            except Exception as exc:
                self._logger.warning("agent-model gateway proxy stream failed: %s", exc)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    async def originate_or_forward(
        self, task_id: str, body: dict[str, Any]
    ) -> StreamingResponse:
        """Run one agent-episode model turn, capturing a facade call server-side.

        The gateway injects the facade tool schema into the turn's tools and runs the
        upstream. If the model calls a facade, the gateway captures its faithful
        structured args, originates the boundary keyed to the episode's task, and
        returns the harness a clean turn-completing message so its rollout never records
        the raw call — durability is fabric-side. An ordinary reasoning turn passes
        through unchanged. The correlation is derived from the resolved-facade count in
        the turn history, so a crash-before-inject re-drive re-derives the same
        correlation and the boundary machinery dedups it.
        """
        try:
            # Resolve off the event loop: the binding lookup reaches the credential
            # vault over the network, and this runs on the async request path.
            binding = await asyncio.to_thread(self._effective, task_id)
        except ResidentBindingNotServable as exc:
            # Detection-only in this transition: the external gateway cannot serve it.
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:
            self._logger.warning("agent-model gateway binding unavailable: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if binding.mode in (GatewayMode.CANNED, GatewayMode.ECHO):
            # No live model to call a facade: settle the turn deterministically.
            text = self.invoke(_dump_input(body.get("input")), task_id)
            return _sse_response(_message_output(text))
        descriptors = self._facades_for(task_id)
        by_name = {d.name: d for d in descriptors}
        injected = [json.loads(d.tool_schema) for d in descriptors]
        tools = _upstream_tools(body.get("tools")) + injected
        forward = {**body, "stream": False, "tools": tools}
        forward.pop("model", None)
        if binding.model:
            forward["model"] = binding.model
        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_sec) as client:
                upstream = await client.post(
                    f"{self._upstream_base(binding)}/responses",
                    json=forward,
                    headers=self._headers(binding),
                )
                upstream.raise_for_status()
                data = upstream.json()
        except Exception as exc:
            # A failed or stalled upstream is not the server's fault; surface a clean
            # typed error, never an unhandled 500 (the exception carries only the url).
            self._logger.warning("agent-model gateway upstream failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        output = data.get("output", []) if isinstance(data, dict) else []
        facades = _facade_calls(output, by_name)
        if facades and self._originator is not None:
            output = self._capture_facades(task_id, body, output, facades, by_name)
        elif facades:
            self._logger.warning(
                "agent-model gateway captured a facade call with no originator "
                "installed; passing it through to the harness"
            )
        return _sse_response(output)

    def _capture_facades(
        self,
        task_id: str,
        body: dict[str, Any],
        output: list[Any],
        facades: list[dict[str, Any]],
        by_name: dict[str, FacadeDescriptor],
    ) -> list[Any]:
        """Capture a turn's facade calls server-side; return the harness-visible output.

        Search facades of one turn form an ordered batch; a spawn facade stays single.
        Mixing search with spawn, or a second facade group while one is outstanding, is
        denied fail-closed. Non-facade (native) tool calls are preserved verbatim so the
        harness runs them; only the captured facade calls become the dispatch message.
        """
        searches = [
            c
            for c in facades
            if by_name[str(c.get("name"))].interface == SEARCH_INTERFACE
        ]
        spawns = [
            c
            for c in facades
            if by_name[str(c.get("name"))].kind is BoundaryEventKind.SPAWN
        ]
        if searches and spawns:
            raise RuntimeError(
                "a facade turn mixes search and spawn interfaces; denied fail-closed"
            )
        if self._fence is not None and self._fence(task_id):
            raise RuntimeError(
                "a facade group is already outstanding for this episode; refused"
            )
        base = _forward_index(body.get("input"))
        if spawns:
            if len(spawns) > 1:
                raise RuntimeError(
                    "a spawn facade turn must carry exactly one spawn call"
                )
            call = spawns[0]
            request = _facade_boundary(
                call, by_name[str(call.get("name"))], f"{task_id}:{base}"
            )
            assert self._originator is not None
            self._originator(task_id, request)
            kept = [item for item in output if item is not call]
            return kept + _message_output(
                _dispatched_message(call, request.child_region_ref)
            )
        members: list[FacadeBatchMember] = []
        for ordinal, call in enumerate(searches):
            raw = call.get("arguments")
            payload = raw if isinstance(raw, str) else json.dumps(raw or {})
            members.append(
                FacadeBatchMember(
                    interface=SEARCH_INTERFACE,
                    call_correlation=f"{task_id}:{base}:{ordinal}",
                    ordinal=ordinal,
                    original_call_id=str(call.get("call_id")),
                    tool_name=str(call.get("name")),
                    request_payload=payload,
                )
            )
        if self._batch_originator is not None:
            self._batch_originator(task_id, f"{task_id}:{base}", members)
        kept = [item for item in output if all(item is not s for s in searches)]
        return kept + _message_output(
            f"Dispatched {len(members)} web search(es); awaiting the mediated results."
        )

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

    def _upstream_base(self, binding: ResolvedGatewayBinding) -> str:
        if not binding.url:
            raise RuntimeError("this gateway mode needs a configured upstream url")
        parsed = urlparse(binding.url)
        if parsed.scheme not in ("http", "https"):
            raise RuntimeError(f"unsupported gateway url scheme {parsed.scheme!r}")
        return binding.url.rstrip("/")

    def _headers(self, binding: ResolvedGatewayBinding) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if binding.api_key:
            headers["Authorization"] = f"Bearer {binding.api_key}"
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


def _facade_calls(
    output: Any, by_name: dict[str, FacadeDescriptor]
) -> list[dict[str, Any]]:
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") in by_name
    ]


def _other_tool_calls(
    output: Any, by_name: dict[str, FacadeDescriptor]
) -> list[dict[str, Any]]:
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") not in by_name
    ]


def _forward_index(history: Any) -> int:
    """The settled-outcome count in the turn history, for a re-drive-stable correlation.

    A harness posts its full turn history each turn, so a re-drive before a facade's
    outcome injects derives the same base, while every injected outcome (a batch member
    under its own call id, or a spawn under a ``fab-`` id) adds a function-call output
    that advances the base for the next turn's facades.
    """
    if not isinstance(history, list):
        return 0
    return sum(
        1
        for item in history
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )


def _facade_boundary(
    call: dict[str, Any], descriptor: FacadeDescriptor, call_correlation: str
) -> BoundaryRequest:
    raw = call.get("arguments")
    payload = raw if isinstance(raw, str) else json.dumps(raw or {})
    args: Any = raw
    if isinstance(raw, str):
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {}
    region = args.get("region") if isinstance(args, dict) else None
    return BoundaryRequest(
        kind=descriptor.kind,
        interface=descriptor.interface,
        call_correlation=call_correlation,
        child_region_ref=region if isinstance(region, str) else None,
        request_payload=payload,
    )


def _dispatched_message(call: dict[str, Any], region: str | None) -> str:
    where = f" to the {region} region" if region else ""
    return f"Dispatched {call.get('name')}{where}; awaiting the mediated result."


def _message_output(text: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "id": "msg_fm",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }
    ]


def _responses_sse(output: list[dict[str, Any]]) -> bytes:
    """A minimal Responses SSE stream re-emitting a buffered turn's output items."""
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp_fm"}}
    ]
    for index, item in enumerate(output):
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    events.append(
        {
            "type": "response.completed",
            "response": {"id": "resp_fm", "status": "completed", "output": output},
        }
    )
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )


def _sse_response(output: list[dict[str, Any]]) -> StreamingResponse:
    """Stream one buffered turn's output items as a Responses SSE response."""
    sse = _responses_sse(output)

    async def _stream() -> AsyncIterator[bytes]:
        yield sse

    return StreamingResponse(_stream(), media_type="text/event-stream")


def build_agent_model_router(gateway: AgentModelGateway) -> APIRouter:
    """The Responses API surface a harness provider targets."""
    router = APIRouter()

    @router.post("/v1/responses")
    async def create_response(request: Request) -> Any:
        body = await request.json()
        if gateway.is_proxy():
            return await gateway.proxy_responses(body)
        return gateway.responses(ResponsesRequest.model_validate(body))

    @router.post("/agent/{task_id}/v1/responses")
    async def agent_response(task_id: str, request: Request) -> Any:
        # The per-episode provider surface a harness targets: the task id in the path
        # correlates a captured facade to the awaiting agent activation.
        body = await request.json()
        return await gateway.originate_or_forward(task_id, body)

    return router
