"""A live Codex app-server proves the bounded gateway-captured facade recovery.

A real ``codex app-server`` runs against the real FlowMesh agent-model gateway, which
proxies to a local Responses backend. The backend emits a native ``spawn_agent`` tool
call; the gateway captures it server-side, originates the fabric boundary, and returns
Codex a clean turn-completing message, so the rollout never records the raw call. The
fabric then settles the boundary and the outcome injects back over
``thread/inject_items``.

What is real and load-bearing is the recovery: across a ``kill -9`` the same rollout
resumes by thread id under a stable ``CODEX_HOME``, and the adapter's committed-key
dedup keeps the settled outcome injected exactly once into the live rollout — observed
by counting the injections the backend sees, so a re-injection would fail the assertion.
Exactly-once of the mediated *effect* under a lost capsule is the fabric idempotency-key
property, proven at the engine level; it is not claimed here.

This is a live process, not a fake transport, and not a CPU Docker end-to-end test,
whose environment lacks the Codex binary. The proof is bounded to single-facade
recovery.
"""

import json
import os
import socket
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "openai_codex", reason="needs the openai-codex worker harness dependency"
)

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from server.config import AgentModelGatewayConfig, GatewayMode  # noqa: E402
from server.services.agent_model_gateway import (  # noqa: E402
    AgentModelGateway,
    build_agent_model_router,
)
from shared.harness import (  # noqa: E402
    BoundaryRequest,
    DeliveredOutcome,
    HarnessResultKind,
    OutcomeKind,
)
from worker.executors.harness.codex import CodexAppServerHarnessAdapter  # noqa: E402
from worker.executors.harness.codex_transport import (  # noqa: E402
    CodexTransportConfig,
    CodexTransportError,
    RealCodexAppServerTransport,
)

_TASK_ID = "tsk-codex-int"
_FINAL_TEXT = "final"


def _codex_available() -> bool:
    # _resolve_codex_bin is private to the SDK; the exact version pin keeps it stable.
    from openai_codex.client import CodexConfig, _resolve_codex_bin

    try:
        return _resolve_codex_bin(CodexConfig()).exists()
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.codex_integration,
    pytest.mark.skipif(not _codex_available(), reason="codex binary unavailable"),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _UpstreamStub:
    """A local Responses backend that emits a facade call and counts injected outcomes.

    ``max_injected`` is the most ``function_call_output`` items keyed to a mediated
    correlation seen in a request's history — one if the outcome injected exactly once,
    more if a recovery re-injected it. Before any outcome is in history it returns a
    native ``spawn_agent`` call; after one is, it returns a plain completion.
    ``stall_release``, when set, holds each turn open until the event fires, standing in
    for a hung upstream.
    """

    def __init__(self, stall_release: threading.Event | None = None) -> None:
        self.max_injected = 0
        self._stall = stall_release
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        def _injected(body: dict[str, Any]) -> int:
            return sum(
                1
                for item in body.get("input", [])
                if isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and str(item.get("call_id", "")).startswith("fab-")
            )

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode()) if n else {}
                injected = _injected(body)
                with stub._lock:
                    stub.max_injected = max(stub.max_injected, injected)
                if stub._stall is not None:
                    stub._stall.wait(30)
                if injected:
                    output = [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": _FINAL_TEXT}],
                        }
                    ]
                else:
                    output = [
                        {
                            "type": "function_call",
                            "name": "spawn_agent",
                            "call_id": "call_review",
                            "arguments": json.dumps({"region": "reviewer"}),
                        }
                    ]
                data = json.dumps(
                    {"object": "response", "status": "completed", "output": output}
                ).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the app-server may already be gone (kill or timeout)

        return Handler

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> "_UpstreamStub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class _Fabric:
    """The fabric's role: record the gateway-originated boundary and settle it once."""

    def __init__(self) -> None:
        self.originated: list[tuple[str, BoundaryRequest]] = []
        self._settled: dict[str, DeliveredOutcome] = {}

    def originate(self, task_id: str, request: BoundaryRequest) -> None:
        # A re-drive re-originates the same stable correlation; record it once.
        if all(
            r.call_correlation != request.call_correlation for _, r in self.originated
        ):
            self.originated.append((task_id, request))

    def settle(self, value: str = _FINAL_TEXT) -> DeliveredOutcome:
        assert self.originated, "the gateway never originated a boundary"
        corr = self.originated[0][1].call_correlation
        assert corr is not None
        if corr not in self._settled:
            self._settled[corr] = DeliveredOutcome(
                call_correlation=corr,
                idempotency_key=f"idm-{corr}",
                kind=OutcomeKind.RESULT,
                value=value,
            )
        return self._settled[corr]


class _GatewayServer:
    """The real agent-model gateway, served over HTTP for Codex to target."""

    def __init__(self, upstream: str, fabric: _Fabric) -> None:
        cfg = AgentModelGatewayConfig(
            mode=GatewayMode.PROXY, url=upstream, model="codex-model"
        )
        gateway = AgentModelGateway(None, cfg)  # type: ignore[arg-type]
        gateway.set_boundary_originator(fabric.originate)
        app = FastAPI()
        app.include_router(build_agent_model_router(gateway))
        self._port = _free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self._port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def __enter__(self) -> "_GatewayServer":
        self._thread.start()
        while not self._server.started:
            threading.Event().wait(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


TransportFactory = Callable[..., RealCodexAppServerTransport]


@pytest.fixture
def transports() -> Iterator[TransportFactory]:
    made: list[RealCodexAppServerTransport] = []

    def _make(base_url: str, home: Path, **kwargs: Any) -> RealCodexAppServerTransport:
        transport = RealCodexAppServerTransport(
            CodexTransportConfig(
                base_url=base_url,
                model="codex-model",
                codex_home=home,
                initial_input="review the auth module for security issues",
                task_id=_TASK_ID,
                **kwargs,
            )
        )
        made.append(transport)
        return transport

    yield _make
    for transport in made:
        transport.close()


def test_kill9_before_injection_injects_the_outcome_once(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    fabric = _Fabric()
    with _UpstreamStub() as stub, _GatewayServer(stub.base_url, fabric) as gateway:
        issue = transports(gateway.base_url, home)
        first = CodexAppServerHarnessAdapter(issue, "v1").start(
            _TASK_ID, capsule=None, outcomes=[]
        )
        # The gateway captured the native spawn_agent and clean-completed the turn.
        assert first.kind is HarnessResultKind.COMPLETION
        assert len(fabric.originated) == 1
        assert fabric.originated[0][1].child_region_ref == "reviewer"

        # The app-server dies after the boundary originates, before its outcome injects.
        os.kill(issue.pid, 9)

        recover = transports(gateway.base_url, home)
        done = CodexAppServerHarnessAdapter(recover, "v1").start(
            _TASK_ID, capsule=first.capsule, outcomes=[fabric.settle()]
        )

    assert done.kind is HarnessResultKind.COMPLETION
    assert stub.max_injected == 1


def test_kill9_after_injection_does_not_reinject(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    fabric = _Fabric()
    with _UpstreamStub() as stub, _GatewayServer(stub.base_url, fabric) as gateway:
        run = transports(gateway.base_url, home)
        adapter = CodexAppServerHarnessAdapter(run, "v1")
        first = adapter.start(_TASK_ID, capsule=None, outcomes=[])
        assert first.kind is HarnessResultKind.COMPLETION
        outcome = fabric.settle()

        completed = adapter.start(_TASK_ID, capsule=first.capsule, outcomes=[outcome])
        assert completed.kind is HarnessResultKind.COMPLETION
        os.kill(run.pid, 9)

        # Recover from the durable capsule that already committed the key; the adapter's
        # dedup must keep it from injecting the outcome into the rollout a second time.
        recover = transports(gateway.base_url, home)
        again = CodexAppServerHarnessAdapter(recover, "v1").start(
            _TASK_ID, capsule=completed.capsule, outcomes=[outcome]
        )

    assert again.kind is HarnessResultKind.COMPLETION
    assert stub.max_injected == 1


def test_stalled_turn_raises_a_transport_error(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    release = threading.Event()
    with (
        _UpstreamStub(stall_release=release) as stub,
        _GatewayServer(stub.base_url, _Fabric()) as gateway,
    ):
        transport = transports(gateway.base_url, home, turn_timeout_sec=2.0)
        adapter = CodexAppServerHarnessAdapter(transport, "v1")
        try:
            with pytest.raises(CodexTransportError):
                adapter.start(_TASK_ID, capsule=None, outcomes=[])
        finally:
            release.set()
