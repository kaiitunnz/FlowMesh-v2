"""The live Codex app-server transport that binds the adapter to a real process.

``RealCodexAppServerTransport`` drives an ``openai_codex.CodexClient``, a JSON-RPC/stdio
client that spawns ``codex app-server``, behind the ``CodexAppServerTransport`` protocol
the adapter already speaks. Each step runs against a persisted on-disk rollout under a
stable ``CODEX_HOME``: a fresh thread's first turn carries the agent's task, ``thread/
resume`` reattaches by thread id after the process is gone, ``thread/inject_items``
appends a settled outcome as raw Responses items the next turn sees, and a turn's
notification stream collapses to one ``CodexEvent``. The model backend is FlowMesh's
Responses gateway, which is Codex's native wire.

The 0.147.0 app-server exposes no client-registered tool whose output the fabric
supplies, so a mediated facade call is detected as a small JSON envelope the model emits
on its turn output — a convention layered on the text, not a native tool-call wire — and
resolved by injecting the settled result. Origination is the only stubbed edge; the
resolution RPC, rollout persistence, and resume are real. The adapter's committed-key
dedup keeps the outcome injected at most once on a resume from the committed capsule.
The untyped ``thread/inject_items`` RPC and provider passthrough are isolated in
``_CodexExperimentalSurface`` and pinned to this Codex version.
"""

import json
import threading
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
# Codex authenticates to the internal FlowMesh Responses gateway with this trusted
# placeholder token (requires_openai_auth=false). The user's model credential is
# supplied at the gateway from the per-workflow secret_ref, resolved server-side, and
# never passes through Codex — so the placeholder is correct here, not a missing one.
_KEY_ENV = "FLOWMESH_CODEX_API_KEY"


class CodexTransportError(RuntimeError):
    """A transport-level failure of the live app-server, distinct from a model outcome.

    A process death, a broken transport, or a turn that stalls past its bound raises
    this rather than a completion, so a physical failure never reads as a result.
    """


@dataclass(frozen=True)
class CodexTransportConfig:
    """The Responses-gateway binding and durable rollout home for one app-server."""

    base_url: str
    model: str
    codex_home: Path
    initial_input: str
    provider_id: str = "flowmesh"
    approval_policy: str = "never"
    sandbox_mode: str = "read-only"
    turn_input: str = "continue"
    turn_timeout_sec: float = 120.0
    cwd: Path | None = None

    def __post_init__(self) -> None:
        # base_url and model reach the codex --config DSL as key="value" overrides, so a
        # quote or newline would break the override; the URL must also stay http(s).
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"the codex base_url must be http(s): {self.base_url!r}")
        for name, value in (("base_url", self.base_url), ("model", self.model)):
            if '"' in value or "\n" in value:
                raise ValueError(f"the codex {name} may not contain a quote or newline")

    def to_codex_config(self) -> CodexConfig:
        p = self.provider_id
        overrides = (
            f'model_providers.{p}.name="{p}"',
            f'model_providers.{p}.base_url="{self.base_url}"',
            f'model_providers.{p}.wire_api="responses"',
            f"model_providers.{p}.requires_openai_auth=false",
            f'model_providers.{p}.env_key="{_KEY_ENV}"',
            f'model_provider="{p}"',
            f'model="{self.model}"',
            f'approval_policy="{self.approval_policy}"',
            f'sandbox_mode="{self.sandbox_mode}"',
        )
        env = {"CODEX_HOME": self.codex_home.as_posix(), _KEY_ENV: "placeholder"}
        return CodexConfig(
            config_overrides=overrides,
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
        self._fresh_thread = False

    @property
    def client(self) -> CodexClient:
        if self._client is None:
            raise RuntimeError("the Codex app-server is not connected")
        return self._client

    @property
    def pid(self) -> int:
        # ``_proc`` is private to the SDK; the exact version pin keeps this stable.
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
        self._fresh_thread = True
        return self._connect().thread_start().thread.id

    def thread_resume(self, thread_id: str, rollout_ref: str) -> None:
        self._fresh_thread = False
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
        # A fresh thread's first turn carries the agent's task; a resume continues the
        # persisted rollout, whose next turn advances past the injected outcome.
        cfg = self._config
        text = cfg.initial_input if self._fresh_thread else cfg.turn_input
        started = self.client.turn_start(thread_id, text)
        return started.turn.id

    def next_event(self, thread_id: str, turn_id: str) -> CodexEvent:
        # Drain on a worker thread bounded by a deadline: a dead process makes the SDK
        # reader fail the queue and re-raise (never a phantom completion), but a process
        # that stalls mid-turn would block forever, so a timeout is a transport failure.
        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["event"] = self._drain_turn(turn_id)
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller below
                box["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(self._config.turn_timeout_sec)
        if worker.is_alive():
            self.close()  # unblock the stalled drain and reap the hung process
            raise CodexTransportError(
                f"the Codex turn {turn_id} produced no terminal event within "
                f"{self._config.turn_timeout_sec}s"
            )
        if "error" in box:
            raise box["error"]
        event: CodexEvent = box["event"]
        return event

    def _drain_turn(self, turn_id: str) -> CodexEvent:
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
