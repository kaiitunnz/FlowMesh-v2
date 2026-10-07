"""The worker-local Responses facade a held Codex episode runs its model turns through.

Codex speaks the OpenAI Responses API; its provider base url is rebound to this
worker-local server, which binds to loopback only and authenticates each request against
a per-episode token, so one episode can never drive another's egress. For each turn the
facade translates the Responses request into a chat request, injects the agent's pinned
fabric tools, and runs the held model egress (propose the digest, await the one-use
permit, egress synchronously) — the server never egresses, the worker sidecar does. It
then maps the reply back to Responses items; if the model called a fabric tool, the
calls are captured into a turn group the turn completion carries to control and the turn
is cleaned, so the episode suspends on the group rather than running the raw call. The
provider credential and the permit are handled by the egress path and never logged here.
"""

import json
import logging
import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from shared.harness import BoundaryEventKind
from shared.sandbox import (
    LocalSandboxExecutor,
    SandboxCommand,
    SandboxDenied,
    SandboxReapUnproved,
    SandboxUnavailable,
)
from shared.tools.facade import FacadeDescriptor, FacadeTurnGroup
from shared.tools.model.schema import (
    MODEL_INTERFACE,
    ModelCompletion,
    ModelRequest,
    ModelToolCall,
)

from ..egress import HeldEgressReject, PendingEgressRequestStore
from .capture import (
    build_facade_capture,
    partition_facade_calls,
    partition_local_calls,
    turn_base,
)
from .held_egress import HeldModelEgress
from .translation import (
    chat_tools,
    completion_to_responses_output,
    function_call_item,
    message_output_item,
    responses_input_to_messages,
    responses_sse,
)

_MAX_BODY_BYTES = 20 * 1024 * 1024
# Bounds how many local commands one held turn may run before it must report back, so a
# turn neither loops on the model forever nor runs an unbounded batch in one round.
_MAX_TURN_COMMANDS = 32


class FacadeTurnError(RuntimeError):
    """A held turn that cannot complete: a denied or failed egress, or bad auth."""


@dataclass(frozen=True)
class EpisodeContext:
    """One held episode's model binding, injectable facades, auth token, and sandbox."""

    url: str
    model: str
    descriptors: tuple[FacadeDescriptor, ...]
    token: str
    # The dispatch running the episode's step, which its captures belong to.
    dispatch_id: str | None
    sandbox: LocalSandboxExecutor | None = None


class ResponsesFacade:
    """Serve one worker's held Codex episodes their model turns over loopback."""

    def __init__(
        self,
        *,
        held_egress: HeldModelEgress,
        pending: PendingEgressRequestStore,
        logger: logging.Logger | None = None,
    ) -> None:
        self._held_egress = held_egress
        self._pending = pending
        self._log = logger or logging.getLogger("responses-facade")
        self._episodes: dict[str, EpisodeContext] = {}
        # A facade group captured on the episode's current turn, handed to the executor
        # so the completion carries it ordered-with the turn rather than on a separate
        # lossy channel that could race or drop it.
        self._captured: dict[str, FacadeTurnGroup] = {}
        # Episodes being given up, each with the event its release sets: a turn that
        # reaches one waits for the release, so its harness never ends the turn on the
        # refusal while it is still up.
        self._refused: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._server: _FacadeHTTPServer | None = None
        self._serve_thread: threading.Thread | None = None
        self._port: int | None = None

    def register_episode(
        self,
        task_id: str,
        url: str,
        model: str,
        descriptors: list[FacadeDescriptor],
        dispatch_id: str | None,
        sandbox: LocalSandboxExecutor | None = None,
    ) -> str:
        """Register one episode's binding and facades; return its per-episode token."""
        token = secrets.token_urlsafe(24)
        with self._lock:
            if (released := self._refused.pop(task_id, None)) is not None:
                released.set()
            self._episodes[task_id] = EpisodeContext(
                url=url,
                model=model,
                descriptors=tuple(descriptors),
                token=token,
                dispatch_id=dispatch_id,
                sandbox=sandbox,
            )
        self._held_egress.reopen(task_id, token)
        return token

    def unregister_episode(self, task_id: str) -> None:
        """Forget the episode, so a turn still running captures nothing, answer the
        turns waiting on its release, and drop a captured group no step will report
        with the requests it stashed."""
        with self._lock:
            ctx = self._episodes.pop(task_id, None)
            group = self._captured.pop(task_id, None)
            released = self._refused.pop(task_id, None)
        if ctx is not None:
            self._held_egress.close(task_id, ctx.token)
        if released is not None:
            released.set()
        if group is not None:
            for member in group.members:
                self._pending.delete(task_id, member.call_correlation)

    def refuse_episode(self, task_id: str) -> None:
        """Refuse the episode's further turns and model calls.

        A refused turn or call waits for the episode's release. A group a returned turn
        already captured stays for its step to report.
        """
        with self._lock:
            if task_id in self._episodes:
                self._refused.setdefault(task_id, threading.Event())
        self._held_egress.refuse(task_id)

    def release_episode(self, task_id: str) -> None:
        """End the episode's turns and model calls waiting on a permit or its
        release."""
        with self._lock:
            ctx = self._episodes.pop(task_id, None)
            released = self._refused.pop(task_id, None)
        if released is not None:
            released.set()
        self._held_egress.release(task_id)
        if ctx is not None:
            self._held_egress.close(task_id, ctx.token)

    def take_captured_group(self, task_id: str) -> FacadeTurnGroup | None:
        """Return and clear the facade group captured on this episode's last turn."""
        with self._lock:
            return self._captured.pop(task_id, None)

    def base_url(self) -> str:
        """The loopback base url a codex provider binds to, once the server is up."""
        if self._port is None:
            raise RuntimeError("the responses facade is not started")
        return f"http://127.0.0.1:{self._port}"

    def start(self) -> int:
        server = _FacadeHTTPServer(("127.0.0.1", 0), _FacadeHandler, self)
        self._server = server
        self._port = server.server_address[1]
        self._serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._serve_thread.start()
        self._log.info("responses facade serving on %s", self.base_url())
        return self._port

    def stop(self, timeout: float = 5.0) -> None:
        if (server := self._server) is not None:
            server.shutdown()
            server.server_close()
        if (thread := self._serve_thread) is not None:
            thread.join(timeout=timeout)
        self._server = None
        self._serve_thread = None
        self._port = None

    def handle_turn(
        self, task_id: str, token: str | None, body: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Run one held model turn and return its Responses output items.

        A denied or failed egress raises ``FacadeTurnError`` so the handler returns an
        error status and the codex turn fails cleanly, never resuming on a phantom.
        """
        ctx = self._episode_for(task_id, token)
        base = turn_base(body.get("input"))
        messages = responses_input_to_messages(body.get("input"))
        tools = chat_tools(body.get("tools"), [d.tool_schema for d in ctx.descriptors])
        completion = self._run_turn(task_id, ctx, messages, tools, base)
        facade_calls, other = partition_facade_calls(
            completion.tool_calls, list(ctx.descriptors)
        )
        if not facade_calls:
            return completion_to_responses_output(completion)
        return self._capture_and_clean(
            task_id, ctx, completion, facade_calls, other, base
        )

    def _episode_for(self, task_id: str, token: str | None) -> EpisodeContext:
        with self._lock:
            ctx = self._episodes.get(task_id)
            refused = self._refused.get(task_id)
        if ctx is None:
            raise FacadeTurnError(f"no registered episode for {task_id}")
        if token != ctx.token:
            raise FacadeTurnError("episode token mismatch")
        if refused is not None:
            self._await_release(task_id, refused)
        return ctx

    def _await_release(self, task_id: str, released: threading.Event) -> None:
        released.wait(self._held_egress.timeout_sec)
        raise FacadeTurnError(f"the episode {task_id} was given up")

    def _run_turn(
        self,
        task_id: str,
        ctx: EpisodeContext,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        base: int,
    ) -> ModelCompletion:
        """Drive one held turn, answering the model's local commands as it makes them.

        A local command is run here and its result appended to this turn's own message
        list, so the model continues from it without the episode yielding its lane.
        Nothing about the command reaches control: no invocation, claim, route, or
        permit. Only the model calls between them are mediated.
        """
        descriptors = list(ctx.descriptors)
        ran = 0
        for round_index in range(_MAX_TURN_COMMANDS + 1):
            if ctx.sandbox is not None and ctx.sandbox.reap_unproved:
                # A command may still be writing the workspace, so nothing it produced
                # reaches another model call, however the harness retries the turn.
                raise FacadeTurnError("a command of this turn was not proved reaped")
            completion = self._egress_turn(
                task_id, ctx, messages, tools, base, round_index
            )
            local, _ = partition_local_calls(completion.tool_calls, descriptors)
            if not local or ran >= _MAX_TURN_COMMANDS:
                return completion
            messages.append(_assistant_tool_calls(completion))
            # Every call the model emitted needs a result, or the next request carries
            # a dangling tool call the backend rejects: a call this turn will not run is
            # answered as not-run rather than left unanswered. A turn given up runs no
            # further command.
            for call in completion.tool_calls:
                if (
                    call in local
                    and ran < _MAX_TURN_COMMANDS
                    and self._holds_turn(task_id, ctx)
                ):
                    ran += 1
                    messages.append(_tool_result(call, self._run_command(ctx, call)))
                else:
                    messages.append(_tool_result(call, _DEFERRED))
        # Unreachable: a round that does not return runs at least one command, so the
        # command bound trips before the round bound does, unless its episode was given
        # up, whose next round's egress is refused.
        raise FacadeTurnError("held turn exhausted its command rounds")

    def _holds_turn(self, task_id: str, ctx: EpisodeContext) -> bool:
        """Whether the turn's episode is still registered with its context and not
        being given up."""
        with self._lock:
            return self._episodes.get(task_id) is ctx and task_id not in self._refused

    def _run_command(self, ctx: EpisodeContext, call: ModelToolCall) -> str:
        """Run one local command and render its result for the model."""
        if ctx.sandbox is None:
            return "denied: this agent declares no sandbox to run commands in"
        try:
            arguments = json.loads(call.arguments or "{}")
            argv = arguments.get("command")
            if isinstance(argv, str):
                argv = [argv]
            if not isinstance(argv, list) or not argv:
                return "denied: 'command' must be a non-empty list of strings"
            timeout = arguments.get("timeout_sec")
            result = ctx.sandbox.execute(
                SandboxCommand(
                    argv=tuple(str(item) for item in argv),
                    timeout_sec=float(timeout) if timeout is not None else None,
                )
            )
        except (ValueError, TypeError) as exc:
            return f"denied: {exc}"
        except (SandboxDenied, SandboxUnavailable) as exc:
            # A command the runtime will not run is a declared terminal outcome of the
            # action: the model is told, and the turn carries on.
            return f"denied: {exc}"
        except SandboxReapUnproved as exc:
            raise FacadeTurnError(str(exc)) from exc
        return json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            }
        )

    def _egress_turn(
        self,
        task_id: str,
        ctx: EpisodeContext,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        base: int,
        round_index: int = 0,
    ) -> ModelCompletion:
        chat_body: dict[str, Any] = {"model": ctx.model, "messages": messages}
        if tools:
            chat_body["tools"] = tools
        request = ModelRequest(interface=MODEL_INTERFACE, url=ctx.url, body=chat_body)
        correlation = (
            f"model:{base}" if not round_index else f"model:{base}:{round_index}"
        )
        result = self._held_egress.run(
            task_id, correlation, request, ctx.token, ctx.dispatch_id
        )
        if isinstance(result, HeldEgressReject):
            raise FacadeTurnError(f"held model egress rejected: {result.reason}")
        return result

    def _capture_and_clean(
        self,
        task_id: str,
        ctx: EpisodeContext,
        completion: ModelCompletion,
        facade_calls: list[ModelToolCall],
        other: list[ModelToolCall],
        base: int,
    ) -> list[dict[str, Any]]:
        capture = build_facade_capture(
            task_id, facade_calls, list(ctx.descriptors), base
        )
        with self._lock:
            # A turn that returns after its step ended, or while it is being given up,
            # has no step to report the group, so it stashes nothing that would outlive
            # it.
            ended = self._episodes.get(task_id) is not ctx
            refused = self._refused.get(task_id)
            if not ended and refused is None:
                for correlation, request in capture.stashes:
                    self._pending.put(task_id, correlation, request, ctx.dispatch_id)
                self._captured[task_id] = capture.group
        if refused is not None:
            self._await_release(task_id, refused)
        if ended:
            raise FacadeTurnError(f"the episode {task_id} ended before its turn")
        output: list[dict[str, Any]] = []
        if completion.content:
            output.append(message_output_item(completion.content))
        # A native tool call co-emitted with the facade calls stays in the turn so the
        # harness runs it; only the fabric-facade calls become the dispatch summary.
        output.extend(function_call_item(call) for call in other)
        output.append(message_output_item(_dispatch_summary(capture.group)))
        return output


_DEFERRED = "not run: ask for this again once your commands are done"


def _assistant_tool_calls(completion: ModelCompletion) -> dict[str, Any]:
    """The model's own message, replayed so its tool results attach to their calls."""
    return {
        "role": "assistant",
        "content": completion.content or None,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in completion.tool_calls
        ],
    }


def _tool_result(call: ModelToolCall, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.call_id, "content": content}


def _dispatch_summary(group: FacadeTurnGroup) -> str:
    spawns = sum(1 for m in group.members if m.kind is BoundaryEventKind.SPAWN)
    searches = len(group.members) - spawns
    parts = [
        label
        for count, label in (
            (spawns, f"{spawns} spawn(s)"),
            (searches, f"{searches} web search(es)"),
        )
        if count
    ]
    return (
        f"Dispatched {' and '.join(parts)}; the mediated results arrive before your "
        "next turn."
    )


class _FacadeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        facade: ResponsesFacade,
    ) -> None:
        super().__init__(address, handler)
        self.facade = facade


class _FacadeHandler(BaseHTTPRequestHandler):
    server: _FacadeHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        # Never log request lines: a query or a bearer token could ride the path.
        return None

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        facade = self.server.facade
        task_id = _task_id_from_path(self.path)
        if task_id is None:
            self._write_json(404, {"error": "unknown route"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._write_json(400, {"error": "invalid Content-Length"})
            return
        if length < 0 or length > _MAX_BODY_BYTES:
            self._write_json(413, {"error": "request body too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._write_json(400, {"error": "invalid JSON body"})
            return
        token = _bearer(self.headers.get("Authorization"))
        try:
            output = facade.handle_turn(task_id, token, body)
        except FacadeTurnError as exc:
            # Every reason is a bounded diagnostic built here or by the egress lane; the
            # request, the credential, and the permit never reach one.
            facade._log.info("facade turn failed for %s: %s", task_id, exc)
            self._write_json(502, {"error": str(exc)})
            return
        sse = responses_sse(output)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


def _task_id_from_path(path: str) -> str | None:
    parts = [p for p in path.split("?", 1)[0].split("/") if p]
    if len(parts) == 4 and parts[0] == "agent" and parts[2:] == ["v1", "responses"]:
        return parts[1]
    return None


def _bearer(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer ") :]
