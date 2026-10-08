"""The run-to-yield agent-episode executor.

Every agent dispatches here through its resolved harness backend key. One ``run`` is
one adapter step: it resumes the backend from the
durable capsule and delivered outcomes the fabric shipped, takes a single run-to-yield
step, and returns the step's :class:`HarnessResult`. The lane releases after the step;
the server routes any boundary and re-dispatches with the next capsule and outcomes.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

from shared.harness import (
    REQUIRED_MEDIATED_FACADES,
    AgentEpisodeDispatch,
    BoundaryEventKind,
    EgressHandoffMode,
    EpisodeModelBinding,
    HarnessAdapter,
    HarnessCapsule,
    HarnessQuiescenceError,
    HarnessResult,
    HarnessResultKind,
    MediatedFacade,
)
from shared.private_state import (
    PrivateStateAttachment,
    PrivateStateUnavailable,
    PrivateStateUnavailableReason,
)
from shared.tasks.specs.misc import ModelBindingMode
from shared.tasks.task_type import TaskType
from shared.tools.model.schema import (
    MODEL_INTERFACE,
    model_request_digest,
    parse_model_request,
)
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    parse_search_request,
    tool_request_digest,
)

from ..egress import CapturedRequest, PendingEgressRequestStore
from ..model_turn import ResponsesFacade
from ..private_state import MaterializedState, PrivateStateHolder, QuiescenceFence
from ..resident import capture_resident_request
from ..sandbox import AgentSandboxRuntime, SandboxRuntime, build_sandbox_runtime
from .base_executor import (
    ExecutionError,
    Executor,
    ExecutorTask,
    RunSignals,
    TaskCancelledError,
)
from .episode_support import EpisodeStepResult, hydrate_delivered_outcomes
from .harness import build_adapter

_LOG = logging.getLogger("agent-episode-executor")
# Cleanups that retry a teardown its step could not prove before the worker gives it up.
_UNENDED_ATTEMPTS = 3


@dataclass(frozen=True)
class _Unended:
    """A writer whose teardown was not proved, kept for a later cleanup to finish."""

    task_id: str
    writer: str
    end: Callable[[], bool | None]
    abandon: Callable[[], None]
    attempts: int = 0


class AgentEpisodeExecutor(Executor):
    """Drive one run-to-yield step of an agent's harness backend."""

    name = "agent_episode"
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset({TaskType.AGENT})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._adapter: HarnessAdapter | None = None
        self._episode_task_id: str | None = None
        self._sandbox_runtime: SandboxRuntime | None = None
        self._signals = RunSignals()
        self._step_sandbox: AgentSandboxRuntime | None = None
        self._unended: list[_Unended] = []

    def run(self, task: ExecutorTask, out_dir: Path) -> EpisodeStepResult:
        with self._signals.running(task.task_id):
            try:
                return self._step(task)
            except BaseException:
                # The step has given up its turn and ended its harness. A cancelled
                # step's give-up releases the episode's waiting turns; any other raised
                # step releases them here. Control never learns of a group a raised step
                # captured, so the worker drops the requests it stashed for it.
                if (facade := self._facade()) is not None:
                    if not self._signals.cancelled:
                        facade.release_episode(task.task_id)
                    facade.unregister_episode(task.task_id)
                raise

    def _step(self, task: ExecutorTask) -> EpisodeStepResult:
        dispatch = task.agent_episode
        if dispatch is None:
            raise ExecutionError(
                f"{task.task_id} routed to the agent-episode executor without an "
                "agent-episode dispatch context"
            )
        state, holder = self._open_private_state(dispatch)
        facade = self._facade()
        prior = self._episode_task_id
        if facade is not None and prior is not None and prior != task.task_id:
            # A different task means the prior episode finished; drop its facade context
            # so a worker running episodes back-to-back does not accumulate them.
            facade.unregister_episode(prior)
        sandbox = self._sandbox(dispatch, state)
        adapter = build_adapter(
            dispatch.backend, task, self._config, facade, state, sandbox
        )
        self._episode_task_id = task.task_id
        self._adapter = adapter
        self._step_sandbox = sandbox
        try:
            try:
                result = self._run_adapter(task, dispatch, adapter, sandbox)
            except BaseException as exc:
                # The turn may still be running on the harness: give it up, refusing
                # its further turns, before ending the harness.
                if facade is not None:
                    facade.refuse_episode(task.task_id)
                if not self._signals.cancelled:
                    try:
                        adapter.cancel(task.task_id)
                    except Exception:
                        _LOG.exception("Failed to give up the turn of %s", task.task_id)
                fence = self._end_writers(
                    task.task_id, adapter, sandbox, dispatch, holder, state
                )
                if fence is None and state is not None:
                    raise self._unproved(state) from exc
                raise
            fence = self._end_writers(
                task.task_id, adapter, sandbox, dispatch, holder, state
            )
        finally:
            self._adapter = None
            self._step_sandbox = None
        capturable = self._is_capturable_boundary(result, dispatch.model_binding)
        if (
            capturable
            and adapter.egress_handoff_mode()
            is not EgressHandoffMode.DURABLE_PRE_EGRESS_YIELD
        ):
            raise ExecutionError(
                f"backend {dispatch.backend.backend!r} deferred a mediated egress "
                "boundary but is not durable_pre_egress_yield"
            )
        value = result.value if result.kind is HarnessResultKind.COMPLETION else None
        sealed = None
        if state is not None and holder is not None:
            if fence is None:
                raise self._unproved(state)
            try:
                sealed = holder.seal(state, _attachment(dispatch), fence)
            except PrivateStateUnavailable as exc:
                raise ExecutionError(f"PrivateStateUnavailable: {exc}") from exc
        # Captured last, so nothing after the capture can raise past a request the step
        # holds for control.
        if capturable:
            result = self._capture_local_request(
                self._pending_egress_requests(),
                task.task_id,
                result,
                dispatch.model_binding,
                task.dispatch_id,
            )
        elif self._is_resident_boundary(result, dispatch.model_binding):
            result = capture_resident_request(
                self._resident_requests(), task.task_id, result, task.dispatch_id
            )
        if result.kind is HarnessResultKind.BOUNDARY and result.request is not None:
            _LOG.info(
                "[fabric] episode yielded a %s boundary (interface=%s)",
                result.request.kind.value,
                result.request.interface or "-",
            )
        group = facade.take_captured_group(task.task_id) if facade is not None else None
        return EpisodeStepResult(
            harness_result=result,
            value=value,
            facade_group=group,
            private_state=sealed,
        )

    def _run_adapter(
        self,
        task: ExecutorTask,
        dispatch: AgentEpisodeDispatch,
        adapter: HarnessAdapter,
        sandbox: AgentSandboxRuntime | None,
    ) -> HarnessResult:
        """Validate the backend and run its one step from the shipped capsule."""
        missing = REQUIRED_MEDIATED_FACADES - adapter.mediated_facades()
        if missing:
            raise ExecutionError(
                f"harness backend {dispatch.backend.backend!r} does not mediate "
                + ", ".join(sorted(missing))
            )
        if (
            sandbox is not None
            and MediatedFacade.SANDBOX not in adapter.mediated_facades()
        ):
            # The agent may run code but this backend would run it natively, outside the
            # fence: refuse rather than execute unconfined.
            raise ExecutionError(
                f"harness backend {dispatch.backend.backend!r} does not mediate the "
                "sandbox its agent declares"
            )
        capsule = (
            HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
            if dispatch.capsule_blob is not None
            else None
        )
        outcomes = hydrate_delivered_outcomes(
            self._lifecycle, task.task_id, dispatch.delivered_outcomes
        )
        for outcome in outcomes:
            _LOG.info(
                "[fabric] injecting %s outcome at call %s",
                outcome.kind.value,
                outcome.call_correlation,
            )
        try:
            self._signals.raise_if_cancelled()
            return adapter.start(task.task_id, capsule=capsule, outcomes=outcomes)
        except Exception as exc:
            # However a cancelled turn unwinds, the step ends as cancelled.
            if self._signals.cancelled and not isinstance(exc, TaskCancelledError):
                raise TaskCancelledError(f"Task {task.task_id} cancelled") from exc
            raise

    def _end_writers(
        self,
        task_id: str,
        adapter: HarnessAdapter,
        sandbox: AgentSandboxRuntime | None,
        dispatch: AgentEpisodeDispatch,
        holder: PrivateStateHolder | None,
        state: MaterializedState | None,
    ) -> QuiescenceFence | None:
        """Stop every writer the step bound to its attachment; return the fence once
        all were proved stopped. Each writer is ended even when another is not proved
        stopped."""
        if sandbox is not None:
            sandbox.close()
        proved = self._ended(
            _Unended(
                task_id,
                "harness",
                lambda: adapter.quiesce(task_id),
                lambda: adapter.abandon(task_id),
            )
        )
        if sandbox is not None:
            drained = self._ended(
                _Unended(
                    task_id, "sandbox", sandbox.finish_reaps, sandbox.abandon_reaps
                ),
                sandbox.drain,
            )
            proved = drained and proved
        if not proved:
            if holder is not None and state is not None:
                holder.mark_unsealable(state)
            return None
        if state is None:
            return None
        return QuiescenceFence.of(state, _attachment(dispatch))

    def _ended(
        self, writer: _Unended, end: Callable[[], bool | None] | None = None
    ) -> bool:
        """Run a writer's teardown (``end``, or else its own), which reports an
        unproved stop by returning False or raising; keep the writer for a later cleanup
        when unproved, and abandon it once its attempts run out."""
        task_id = writer.task_id
        try:
            if (end or writer.end)() is not False:
                return True
            reason = "its reap was not proved"
        except HarnessQuiescenceError as exc:
            reason = str(exc)
        except Exception as exc:
            _LOG.exception("Ending the %s of task %s failed", writer.writer, task_id)
            reason = repr(exc)
        _LOG.error(
            "The %s of task %s was not proved stopped: %s",
            writer.writer,
            task_id,
            reason,
        )
        if writer.attempts < _UNENDED_ATTEMPTS:
            self._unended.append(writer)
            return False
        _LOG.error(
            "Giving up the %s of task %s, left without an owner", writer.writer, task_id
        )
        try:
            writer.abandon()
        except Exception:
            _LOG.exception(
                "Abandoning the %s of task %s failed", writer.writer, task_id
            )
        return False

    def _unproved(self, state: MaterializedState) -> Exception:
        """Build the non-retryable failure of a step with no proved capture; a
        requested cancellation ends it as cancelled."""
        unavailable = PrivateStateUnavailable(
            PrivateStateUnavailableReason.QUIESCENCE_UNPROVED,
            "the step's writers were not proved stopped before its seal",
            reference_id=state.reference_id,
        )
        if self._signals.cancelled:
            return TaskCancelledError(f"cancelled; {unavailable}")
        return ExecutionError(f"PrivateStateUnavailable: {unavailable}")

    def _sandbox(
        self, dispatch: AgentEpisodeDispatch, state: MaterializedState | None
    ) -> AgentSandboxRuntime | None:
        """The fenced runtime this dispatch's commands run in, or None for an agent
        that declares no sandbox."""
        capability = dispatch.sandbox
        if capability is None or state is None:
            return None
        if self._sandbox_runtime is None:
            self._sandbox_runtime = build_sandbox_runtime()
        return AgentSandboxRuntime(
            capability, _attachment(dispatch), state, self._sandbox_runtime
        )

    def _open_private_state(
        self, dispatch: AgentEpisodeDispatch
    ) -> tuple[MaterializedState | None, PrivateStateHolder | None]:
        """Materialize the activation's bound generation for this dispatch."""
        binding = dispatch.private_state
        if binding is None:
            return None, None
        holder = PrivateStateHolder(self._config.private_state_dir)
        try:
            state = holder.open(binding, _attachment(dispatch))
        except PrivateStateUnavailable as exc:
            raise ExecutionError(f"PrivateStateUnavailable: {exc}") from exc
        return state, holder

    @staticmethod
    def _is_capturable_boundary(
        result: HarnessResult, model_binding: EpisodeModelBinding | None
    ) -> bool:
        """Whether a step yielded a worker-originatable egress boundary.

        A ``search/v1`` invocation is always worker-originated; a ``model`` invocation
        is worker-originated only for an external (``openai``) binding — a
        ``canned``/``echo``/``resident`` model boundary settles on the control plane.
        """
        req = result.request
        if (
            result.kind is not HarnessResultKind.BOUNDARY
            or req is None
            or req.kind is not BoundaryEventKind.INVOCATION
            or req.request_payload is None
            or req.call_correlation is None
        ):
            return False
        if req.interface == SEARCH_INTERFACE:
            return True
        return (
            req.interface == MODEL_INTERFACE
            and model_binding is not None
            and model_binding.mode is ModelBindingMode.OPENAI
        )

    @staticmethod
    def _capture_local_request(
        store: PendingEgressRequestStore,
        task_id: str,
        result: HarnessResult,
        model_binding: EpisodeModelBinding | None,
        dispatch_id: str | None,
    ) -> HarnessResult:
        """Keep a worker-originated egress request local and emit only its digest.

        A worker-originated invocation boundary has its raw request recorded in
        worker-private state keyed by ``(task_id, call_correlation)`` and stripped from
        the returned boundary, which instead carries only the request digest. Any other
        boundary passes through unchanged.
        """
        req = result.request
        if not AgentEpisodeExecutor._is_capturable_boundary(result, model_binding):
            return result
        assert req is not None and req.request_payload is not None
        assert req.call_correlation is not None
        if req.interface == MODEL_INTERFACE:
            assert model_binding is not None
            model = parse_model_request(
                req.request_payload,
                url=model_binding.url or "",
                model=model_binding.model or "",
            )
            captured: CapturedRequest = model
            digest = model_request_digest(model.interface, model.url, model.body)
        else:
            parsed = parse_search_request(req.request_payload)
            captured = parsed
            digest = tool_request_digest(
                parsed.interface, parsed.query, parsed.max_results
            )
        stripped = req.model_copy(
            update={"request_payload": None, "request_digest": digest}
        )
        stripped_result = result.model_copy(update={"request": stripped})
        store.put(task_id, req.call_correlation, captured, dispatch_id)
        return stripped_result

    @staticmethod
    def _is_resident_boundary(
        result: HarnessResult, model_binding: EpisodeModelBinding | None
    ) -> bool:
        """Whether a step yielded a resident model boundary the origin worker drives.

        A resident model invocation is worker-originated like an external one, but its
        raw request is held in a separate resident custody and it settles through the
        claim-gated resident path rather than the egress sidecar.
        """
        req = result.request
        if (
            result.kind is not HarnessResultKind.BOUNDARY
            or req is None
            or req.kind is not BoundaryEventKind.INVOCATION
            or req.request_payload is None
            or req.call_correlation is None
        ):
            return False
        return (
            req.interface == MODEL_INTERFACE
            and model_binding is not None
            and model_binding.mode is ModelBindingMode.RESIDENT
        )

    def cancel(self, task_id: str) -> None:
        if not self._signals.cancel(task_id):
            return
        if (sandbox := self._step_sandbox) is not None:
            sandbox.close()
        adapter = self._adapter
        facade = self._facade()
        # Ending the harness waits for it to exit, and the caller may be the thread that
        # relays the worker's permits and reaps.
        threading.Thread(
            target=_give_up,
            args=(task_id, adapter, facade),
            name="agent-episode-give-up",
            daemon=True,
        ).start()

    def cleanup_after_run(self) -> None:
        facade = self._facade()
        if facade is not None and self._episode_task_id is not None:
            facade.unregister_episode(self._episode_task_id)
        self._episode_task_id = None
        unended, self._unended = self._unended, []
        for writer in unended:
            self._ended(replace(writer, attempts=writer.attempts + 1))

    def _facade(self) -> ResponsesFacade | None:
        return self._lifecycle.responses_facade if self._lifecycle else None


def _give_up(
    task_id: str, adapter: HarnessAdapter | None, facade: ResponsesFacade | None
) -> None:
    if facade is not None:
        facade.refuse_episode(task_id)
    try:
        if adapter is not None:
            adapter.cancel(task_id)
    finally:
        # After the harness has exited, so it cannot end its turn on the released call
        # or retry it into a fresh held turn.
        if facade is not None:
            facade.release_episode(task_id)


def _attachment(dispatch: AgentEpisodeDispatch) -> PrivateStateAttachment:
    """The write authority a holder needs before it may materialize private state."""
    if dispatch.private_state_attachment is None:
        raise ExecutionError("an agent private-state binding ships with its attachment")
    return dispatch.private_state_attachment
