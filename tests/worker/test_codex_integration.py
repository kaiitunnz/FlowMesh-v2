"""A live Codex app-server proves the bounded worker-facade recovery.

A real ``codex app-server`` runs against the worker-local Responses facade, whose model
provider is a local Chat Completions backend. The backend emits a native ``spawn_agent``
tool call; the facade captures it into a turn group the completion carries, and returns
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
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "openai_codex", reason="needs the openai-codex worker harness dependency"
)

import logging  # noqa: E402

from shared.harness import (  # noqa: E402
    BoundaryEventKind,
    DeliveredOutcome,
    HarnessResultKind,
    OutcomeKind,
)
from shared.tools.contract import (  # noqa: E402
    AgentModelTurnProposal,
    MediatedOperationPermit,
)
from shared.tools.facade import FacadeDescriptor, FacadeTurnGroup  # noqa: E402
from shared.tools.model.schema import MODEL_INTERFACE  # noqa: E402
from shared.utils.ids import (  # noqa: E402
    new_idempotency_key,
    new_invocation_id,
    new_mediated_permit_id,
)
from worker.egress import MediatedEgressSidecar  # noqa: E402
from worker.egress import ModelEgress  # noqa: E402
from worker.egress import PendingEgressRequestStore  # noqa: E402
from worker.executors.harness.codex import CodexAppServerHarnessAdapter  # noqa: E402
from worker.executors.harness.codex_transport import (  # noqa: E402
    CodexTransportConfig,
    CodexTransportError,
    RealCodexAppServerTransport,
)
from worker.model_turn import HeldModelEgress  # noqa: E402
from worker.model_turn import ModelTurnRendezvous  # noqa: E402
from worker.model_turn import ResponsesFacade  # noqa: E402

_TASK_ID = "tsk-codex-int"
_FINAL_TEXT = "final"
_WORKER_ID = "wkr-int"
_WORKER_GEN = 1
_LOG = logging.getLogger("codex-integration-test")


def _spawn_facade() -> FacadeDescriptor:
    return FacadeDescriptor(
        name="spawn_agent",
        kind=BoundaryEventKind.SPAWN,
        tool_schema=json.dumps(
            {
                "type": "function",
                "name": "spawn_agent",
                "parameters": {
                    "type": "object",
                    "properties": {"region": {"type": "string"}},
                    "required": ["region"],
                },
            }
        ),
    )


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


class _UpstreamStub:
    """A local Chat Completions backend that emits a facade call and counts injections.

    ``max_injected`` is the most tool-result messages keyed to a mediated correlation
    seen in a request's chat history — one if the outcome injected exactly once, more if
    a recovery re-injected it. Before any outcome is in history it returns a native
    ``spawn_agent`` tool call; after one is, it returns a plain completion.
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
                for item in body.get("messages", [])
                if isinstance(item, dict)
                and item.get("role") == "tool"
                and str(item.get("tool_call_id", "")).startswith("fab-")
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
                message: dict[str, Any]
                if injected:
                    message = {"role": "assistant", "content": _FINAL_TEXT}
                else:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_review",
                                "type": "function",
                                "function": {
                                    "name": "spawn_agent",
                                    "arguments": json.dumps({"region": "reviewer"}),
                                },
                            }
                        ],
                    }
                data = json.dumps({"choices": [{"message": message}]}).encode()
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
    """The control plane's role: mint a per-turn permit and settle a captured group.

    ``propose`` stands in for the worker→control propose: it mints the audience-bound,
    digest-fenced permit for the held turn and hands it to the rendezvous, so the facade
    egresses synchronously. ``settle`` builds the single delivered outcome injected back
    on recovery from the group the facade captured.
    """

    def __init__(self, rendezvous: ModelTurnRendezvous) -> None:
        self._rendezvous = rendezvous
        self._settled: dict[str, DeliveredOutcome] = {}

    def propose(self, proposal: AgentModelTurnProposal) -> None:
        permit = MediatedOperationPermit(
            permit_id=new_mediated_permit_id(),
            agent_task_id=proposal.agent_task_id,
            call_correlation=proposal.call_correlation,
            interface=MODEL_INTERFACE,
            subject="test",
            invocation_id=new_invocation_id(),
            idempotency_key=new_idempotency_key(),
            request_digest=proposal.request_digest,
            target_id=_WORKER_ID,
            target_generation=_WORKER_GEN,
            deadline_epoch=time.time() + 120.0,
            max_results=1,
            timeout_sec=60.0,
            result_char_cap=1_000_000,
        )
        self._rendezvous.deliver_permit(permit)

    def settle(
        self, group: FacadeTurnGroup, value: str = _FINAL_TEXT
    ) -> DeliveredOutcome:
        corr = group.members[0].call_correlation
        assert corr is not None
        if corr not in self._settled:
            self._settled[corr] = DeliveredOutcome(
                call_correlation=corr,
                idempotency_key=f"idm-{corr}",
                kind=OutcomeKind.RESULT,
                value=value,
            )
        return self._settled[corr]


class _FacadeServer:
    """The worker-local Responses facade, serving Codex its held model turns."""

    def __init__(self, upstream: str) -> None:
        pending = PendingEgressRequestStore()
        rendezvous = ModelTurnRendezvous()
        self.fabric = _Fabric(rendezvous)
        sidecar = MediatedEgressSidecar(
            pending_requests=pending,
            audience=lambda: (_WORKER_ID, _WORKER_GEN),
            egresses=(ModelEgress(None, _LOG),),
            outcome_sink=lambda outcome: None,
            content_store=None,
            logger=_LOG,
        )
        held_egress = HeldModelEgress(
            rendezvous=rendezvous,
            pending=pending,
            propose=self.fabric.propose,
            sidecar=sidecar,
            timeout_sec=30.0,
            logger=_LOG,
        )
        self._facade = ResponsesFacade(
            held_egress=held_egress, pending=pending, logger=_LOG
        )
        self._upstream = upstream

    def __enter__(self) -> "_FacadeServer":
        self._facade.start()
        self.token = self._facade.register_episode(
            _TASK_ID, self._upstream, "codex-model", [_spawn_facade()]
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self._facade.stop()

    def captured_group(self) -> FacadeTurnGroup | None:
        """The facade group captured on the last turn — what the completion carries."""
        return self._facade.take_captured_group(_TASK_ID)

    @property
    def base_url(self) -> str:
        return self._facade.base_url()


TransportFactory = Callable[..., RealCodexAppServerTransport]


@pytest.fixture
def transports() -> Iterator[TransportFactory]:
    made: list[RealCodexAppServerTransport] = []

    def _make(
        base_url: str, token: str, home: Path, **kwargs: Any
    ) -> RealCodexAppServerTransport:
        transport = RealCodexAppServerTransport(
            CodexTransportConfig(
                base_url=base_url,
                model="codex-model",
                codex_home=home,
                initial_input="review the auth module for security issues",
                task_id=_TASK_ID,
                env_key_value=token,
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
    with _UpstreamStub() as stub, _FacadeServer(stub.base_url) as facade:
        issue = transports(facade.base_url, facade.token, home)
        first = CodexAppServerHarnessAdapter(issue, "v1").start(
            _TASK_ID, capsule=None, outcomes=[]
        )
        # The facade captured the native spawn_agent and clean-completed the turn; the
        # completion carries the captured group.
        assert first.kind is HarnessResultKind.COMPLETION
        group = facade.captured_group()
        assert group is not None
        assert group.members[0].interface_or_region == "reviewer"

        # The app-server dies after the boundary originates, before its outcome injects.
        os.kill(issue.pid, 9)

        recover = transports(facade.base_url, facade.token, home)
        done = CodexAppServerHarnessAdapter(recover, "v1").start(
            _TASK_ID, capsule=first.capsule, outcomes=[facade.fabric.settle(group)]
        )

    assert done.kind is HarnessResultKind.COMPLETION
    assert stub.max_injected == 1


def test_kill9_after_injection_does_not_reinject(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    with _UpstreamStub() as stub, _FacadeServer(stub.base_url) as facade:
        run = transports(facade.base_url, facade.token, home)
        adapter = CodexAppServerHarnessAdapter(run, "v1")
        first = adapter.start(_TASK_ID, capsule=None, outcomes=[])
        assert first.kind is HarnessResultKind.COMPLETION
        group = facade.captured_group()
        assert group is not None
        outcome = facade.fabric.settle(group)

        completed = adapter.start(_TASK_ID, capsule=first.capsule, outcomes=[outcome])
        assert completed.kind is HarnessResultKind.COMPLETION
        os.kill(run.pid, 9)

        # Recover from the durable capsule that already committed the key; the adapter's
        # dedup must keep it from injecting the outcome into the rollout a second time.
        recover = transports(facade.base_url, facade.token, home)
        again = CodexAppServerHarnessAdapter(recover, "v1").start(
            _TASK_ID, capsule=completed.capsule, outcomes=[outcome]
        )

    assert again.kind is HarnessResultKind.COMPLETION
    assert stub.max_injected == 1


def test_app_server_child_env_excludes_worker_secrets(
    tmp_path: Path, transports: TransportFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOWMESH_SENTINEL_SECRET", "sentinel-leak-value")
    home = tmp_path / "codex_home"
    with _UpstreamStub() as stub, _FacadeServer(stub.base_url) as facade:
        issue = transports(facade.base_url, facade.token, home)
        result = CodexAppServerHarnessAdapter(issue, "v1").start(
            _TASK_ID, capsule=None, outcomes=[]
        )
        assert result.kind is HarnessResultKind.COMPLETION
        # The live child's environment, captured at exec, holds no worker secret.
        environ = Path(f"/proc/{issue.pid}/environ").read_bytes()
    assert b"FLOWMESH_SENTINEL_SECRET" not in environ
    assert b"sentinel-leak-value" not in environ


def test_stalled_turn_raises_a_transport_error(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    release = threading.Event()
    with (
        _UpstreamStub(stall_release=release) as stub,
        _FacadeServer(stub.base_url) as facade,
    ):
        transport = transports(
            facade.base_url, facade.token, home, turn_timeout_sec=2.0
        )
        adapter = CodexAppServerHarnessAdapter(transport, "v1")
        try:
            with pytest.raises(CodexTransportError):
                adapter.start(_TASK_ID, capsule=None, outcomes=[])
        finally:
            release.set()
