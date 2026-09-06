"""The FlowMesh agent-model gateway for control-plane model settles.

A managed model request an agent defers becomes a durable invocation the fabric settles
here for a ``canned``/``echo`` binding, off the agent's lane, injecting the result back
at the originating call. An external (``openai``) binding egresses on the agent's own
worker, so it never reaches this settle; the pinned binding and its vaulted credential
resolve here only for the control-plane mode.
"""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from shared.tasks.specs import ModelBindingMode

from ..config import AgentModelGatewayConfig, GatewayMode
from ..orchestration.tool_dispatch import ToolInvocationEnvelope
from ..task.v2.representations.operators import AgentModelGatewayBinding

_BINDING_MODE_TO_GATEWAY = {
    ModelBindingMode.CANNED: GatewayMode.CANNED,
    ModelBindingMode.ECHO: GatewayMode.ECHO,
    ModelBindingMode.OPENAI: GatewayMode.OPENAI,
}


@dataclass(frozen=True)
class ResolvedGatewayBinding:
    """The control-plane settle mode a mediated model request resolves to.

    Only the mode is resolved: a canned/echo boundary settles here, and an external
    boundary egresses on the worker with its credential carried on the worker permit, so
    no upstream or credential is materialized server-side.
    """

    mode: GatewayMode


GatewayBindingResolver = Callable[[str], ResolvedGatewayBinding | None]


class ResidentBindingNotServable(RuntimeError):
    """A resident model binding needs capacity admission the external gateway lacks."""


def to_gateway_binding(pinned: AgentModelGatewayBinding) -> ResolvedGatewayBinding:
    """Map a pinned model binding to its control-plane settle mode.

    A canned/echo binding settles on the control plane; an external binding egresses on
    the worker and a resident binding admits through resident-capacity control, so only
    the mode is resolved here — the credential rides the worker permit, never this path.
    """
    mode = _BINDING_MODE_TO_GATEWAY.get(pinned.mode)
    if mode is None:
        raise ResidentBindingNotServable(
            f"model binding mode {pinned.mode.value!r} is not served externally"
        )
    return ResolvedGatewayBinding(mode=mode)


class _EpisodeSettler(Protocol):
    def settle_episode_invocation(
        self,
        task_id: str,
        call_correlation: str,
        value: str | None,
        *,
        error: str | None = None,
    ) -> bool: ...


class AgentModelGateway:
    """Settle a control-plane model invocation for a canned or echo binding.

    A model boundary an adapter defers with a ``canned``/``echo`` binding settles here
    off the agent's lane; the outcome injects at the originating call so the episode
    resumes with the model result. An external binding egresses on the worker and never
    reaches this path.
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
        self._binding_resolver: GatewayBindingResolver | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="agent-model-gateway"
        )

    def set_binding_resolver(self, resolver: GatewayBindingResolver) -> None:
        """Install the per-invocation upstream resolver keyed by episode task id."""
        self._binding_resolver = resolver

    def _effective(self, task_id: str | None) -> ResolvedGatewayBinding:
        """The upstream for this invocation: the pinned binding, else the default.

        The request body never selects the upstream; only the activation's pinned
        binding (or, off an episode, the deployment default) does.
        """
        if task_id is not None and self._binding_resolver is not None:
            if (resolved := self._binding_resolver(task_id)) is not None:
                return resolved
        return ResolvedGatewayBinding(mode=self._cfg.mode)

    def settle(self, env: ToolInvocationEnvelope) -> None:
        """Settle a suspended model boundary off the caller's lane, never inline."""
        self._executor.submit(
            self._settle, env.task_id, env.call_correlation, env.request_payload
        )

    def _settle(self, task_id: str, call_correlation: str, payload: str | None) -> None:
        try:
            value = self.invoke(payload, task_id)
        except Exception as exc:
            # A failed settle fails the boundary rather than resuming the agent with a
            # phantom empty success.
            self._logger.warning("agent-model gateway settle failed: %s", exc)
            self._settler.settle_episode_invocation(
                task_id, call_correlation, None, error=str(exc)
            )
            return
        self._settler.settle_episode_invocation(task_id, call_correlation, value)

    def shutdown(self) -> None:
        """Stop accepting settles and release the off-lane executor."""
        self._executor.shutdown(wait=False)

    def invoke(self, payload: str | None, task_id: str | None = None) -> str:
        """Settle this invocation's model request for a canned or echo binding.

        An external binding egresses on the agent's worker, so it never settles here; a
        request that resolves to one is a fabric misconfiguration and fails the
        boundary.
        """
        binding = self._effective(task_id)
        prompt = _extract_prompt(payload)
        if binding.mode is GatewayMode.ECHO:
            return prompt
        if binding.mode in (GatewayMode.OPENAI, GatewayMode.PROXY):
            raise RuntimeError(
                "an external model binding egresses on the worker, not the server"
            )
        return f"canned-response:{prompt}" if prompt else "canned-response"


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
