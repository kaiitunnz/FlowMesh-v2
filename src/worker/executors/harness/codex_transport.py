"""The live Codex app-server transport that binds the adapter to a real process.

``RealCodexAppServerTransport`` drives an ``openai_codex.CodexClient``, a JSON-RPC/stdio
client that spawns ``codex app-server``, behind the ``CodexAppServerTransport`` protocol
the adapter already speaks. Each step runs against a persisted on-disk rollout under a
stable ``CODEX_HOME``: ``thread/resume`` reattaches by thread id after the process is
gone, ``thread/inject_items`` appends a settled outcome as raw Responses items the next
turn sees, and a turn's notification stream collapses to one ``CodexEvent``. The model
backend is FlowMesh's Responses gateway, which is Codex's native wire.

The 0.147.0 app-server exposes no client-registered tool whose output the fabric
supplies, so a mediated facade call is signalled by the model as a small envelope on its
turn output and resolved by injecting the settled result. The origination is the only
stubbed edge; resolution, rollout persistence, resume, and the exactly-once boundary
stay real. The untyped ``thread/inject_items`` RPC and the provider passthrough are
isolated in ``_CodexExperimentalSurface`` and pinned to this Codex version.
"""

import json
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai_codex.client import CodexClient, CodexConfig
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    ErrorNotification,
    ItemCompletedNotification,
    ThreadInjectItemsResponse,
    TurnCompletedNotification,
    TurnStatus,
)

from .codex import CodexEvent, CodexInjectItem

_FACADE_ENVELOPE_KEY = "facade"
_INJECT_CALL_PREFIX = "fab-"
_INJECT_TOOL = "fabric_mediated"


@dataclass(frozen=True)
class CodexTransportConfig:
    """The Responses-gateway binding and durable rollout home for one app-server."""

    base_url: str
    model: str
    codex_home: Path
    api_key: str | None = None
    api_key_env: str = "FLOWMESH_CODEX_API_KEY"
    provider_id: str = "flowmesh"
    approval_policy: str = "never"
    sandbox_mode: str = "read-only"
    turn_input: str = "continue"
    cwd: Path | None = None
    extra_config_overrides: tuple[str, ...] = ()

    def to_codex_config(self) -> CodexConfig:
        p = self.provider_id
        overrides = [
            f'model_providers.{p}.name="{p}"',
            f'model_providers.{p}.base_url="{self.base_url}"',
            f'model_providers.{p}.wire_api="responses"',
            f"model_providers.{p}.requires_openai_auth=false",
            f'model_providers.{p}.env_key="{self.api_key_env}"',
            f'model_provider="{p}"',
            f'model="{self.model}"',
            f'approval_policy="{self.approval_policy}"',
            f'sandbox_mode="{self.sandbox_mode}"',
            *self.extra_config_overrides,
        ]
        env = {
            "CODEX_HOME": self.codex_home.as_posix(),
            self.api_key_env: self.api_key or "x",
        }
        return CodexConfig(
            config_overrides=tuple(overrides),
            env=env,
            cwd=self.cwd.as_posix() if self.cwd is not None else None,
            experimental_api=True,
        )


class _CodexExperimentalSurface:
    """The untyped Codex RPCs, isolated and pinned to the bound app-server version."""

    @staticmethod
    def inject_response_items(
        client: CodexClient, thread_id: str, items: list[dict[str, Any]]
    ) -> None:
        params: dict[str, Any] = {"threadId": thread_id, "items": items}
        client.request(
            "thread/inject_items", params, response_model=ThreadInjectItemsResponse
        )


def _outcome_to_response_items(item: CodexInjectItem) -> list[dict[str, Any]]:
    call_id = f"{_INJECT_CALL_PREFIX}{item.call_correlation}"
    output = f"denied: {item.value or ''}" if item.denied else (item.value or "")
    return [
        {
            "type": "function_call",
            "name": _INJECT_TOOL,
            "arguments": "{}",
            "call_id": call_id,
        },
        {"type": "function_call_output", "call_id": call_id, "output": output},
    ]


def _parse_facade_envelope(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    facade = obj.get(_FACADE_ENVELOPE_KEY) if isinstance(obj, dict) else None
    return facade if isinstance(facade, dict) else None


def _close_client(client: CodexClient) -> None:
    # Teardown is best-effort; a lost app-server process is already gone.
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass


class RealCodexAppServerTransport:
    """Bind the adapter's transport protocol to a live ``codex app-server`` process."""

    def __init__(self, config: CodexTransportConfig) -> None:
        self._config = config
        self._client: CodexClient | None = None
        self._finalizer: weakref.finalize | None = None

    @property
    def client(self) -> CodexClient:
        if self._client is None:
            raise RuntimeError("the Codex app-server is not connected")
        return self._client

    @property
    def pid(self) -> int:
        proc = self.client._proc
        if proc is None:
            raise RuntimeError("the Codex app-server is not running")
        return proc.pid

    def _connect(self) -> CodexClient:
        if self._client is None:
            self._config.codex_home.mkdir(parents=True, exist_ok=True)
            client = CodexClient(self._config.to_codex_config())
            # Arm teardown before the process spawns, so a failure during start or
            # initialize still reaps the app-server rather than leaking it.
            self._finalizer = weakref.finalize(self, _close_client, client)
            client.start()
            client.initialize()
            self._client = client
        return self._client

    def thread_start(self) -> str:
        return self._connect().thread_start().thread.id

    def thread_resume(self, thread_id: str, rollout_ref: str) -> None:
        self._connect().thread_resume(thread_id)

    def thread_inject_items(
        self, thread_id: str, items: Sequence[CodexInjectItem]
    ) -> None:
        raw: list[dict[str, Any]] = []
        for item in items:
            raw.extend(_outcome_to_response_items(item))
        if raw:
            _CodexExperimentalSurface.inject_response_items(self.client, thread_id, raw)

    def turn_start(self, thread_id: str) -> str:
        started = self.client.turn_start(thread_id, self._config.turn_input)
        return started.turn.id

    def next_event(self, thread_id: str, turn_id: str) -> CodexEvent:
        client = self.client
        agent_texts: list[str] = []
        while True:
            payload = client.next_turn_notification(turn_id).payload
            if isinstance(payload, ItemCompletedNotification):
                if isinstance(root := payload.item.root, AgentMessageThreadItem):
                    agent_texts.append(root.text)
            elif isinstance(payload, ErrorNotification) and not payload.will_retry:
                return CodexEvent(kind="error", value=payload.error.message)
            elif isinstance(payload, TurnCompletedNotification):
                turn = payload.turn
                if turn.status is not TurnStatus.completed or turn.error is not None:
                    message = turn.error.message if turn.error else str(turn.status)
                    return CodexEvent(kind="error", value=message)
                return self._event_from_output(agent_texts, turn_id)

    def _event_from_output(self, agent_texts: list[str], turn_id: str) -> CodexEvent:
        text = agent_texts[-1] if agent_texts else ""
        if (facade := _parse_facade_envelope(text)) is not None:
            return CodexEvent(
                kind="facade_call",
                call_id=turn_id,
                tool=facade.get("tool"),
                interface=facade.get("interface"),
                region=facade.get("region"),
                arguments=facade.get("arguments"),
            )
        return CodexEvent(kind="completed", value=text)

    def cancel(self, thread_id: str) -> None:
        self.close()

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._client = None
