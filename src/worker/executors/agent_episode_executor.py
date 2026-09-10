"""The run-to-yield agent-episode executor.

Every agent dispatches here through its resolved harness backend key. One ``run`` is
one adapter step: it resumes the backend from the
durable capsule and delivered outcomes the fabric shipped, takes a single run-to-yield
step, and returns the step's :class:`HarnessResult`. The lane releases after the step;
the server routes any boundary and re-dispatches with the next capsule and outcomes.
"""

import logging
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
    HarnessResult,
    HarnessResultKind,
)
from shared.private_state import PrivateStateAttachment, PrivateStateUnavailable
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

from ..egress import PendingEgressRequestStore
from ..private_state import MaterializedState, PrivateStateHolder
from ..resident import capture_resident_request
from .base_executor import ExecutionError, Executor, ExecutorTask
from .episode_support import EpisodeStepResult, hydrate_delivered_outcomes
from .harness import build_adapter
from .harness.codex import legacy_codex_home

_LOG = logging.getLogger("agent-episode-executor")


class AgentEpisodeExecutor(Executor):
    """Drive one run-to-yield step of an agent's harness backend."""

    name = "agent_episode"
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset({TaskType.AGENT})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._adapter: HarnessAdapter | None = None
        self._episode_task_id: str | None = None

    def run(self, task: ExecutorTask, out_dir: Path) -> EpisodeStepResult:
        dispatch = task.agent_episode
        if dispatch is None:
            raise ExecutionError(
                f"{task.task_id} routed to the agent-episode executor without an "
                "agent-episode dispatch context"
            )
        state, holder = self._open_private_state(task, dispatch)
        facade = self._lifecycle.responses_facade if self._lifecycle else None
        prior = self._episode_task_id
        if facade is not None and prior is not None and prior != task.task_id:
            # A different task means the prior episode finished; drop its facade context
            # so a worker running episodes back-to-back does not accumulate them.
            facade.unregister_episode(prior)
        adapter = build_adapter(dispatch.backend, task, self._config, facade, state)
        self._episode_task_id = task.task_id
        missing = REQUIRED_MEDIATED_FACADES - adapter.mediated_facades()
        if missing:
            raise ExecutionError(
                f"harness backend {dispatch.backend.backend!r} does not mediate "
                + ", ".join(sorted(missing))
            )
        self._adapter = adapter
        capsule = (
            HarnessCapsule(backend=dispatch.backend, blob=dispatch.capsule_blob)
            if dispatch.capsule_blob is not None
            else None
        )
        outcomes = hydrate_delivered_outcomes(
            self._config.server_base_url, dispatch.delivered_outcomes
        )
        for outcome in outcomes:
            _LOG.info(
                "[fabric] injecting %s outcome at call %s",
                outcome.kind.value,
                outcome.call_correlation,
            )
        result = adapter.start(task.task_id, capsule=capsule, outcomes=outcomes)
        if self._is_capturable_boundary(result, dispatch.model_binding):
            if (
                adapter.egress_handoff_mode()
                is not EgressHandoffMode.DURABLE_PRE_EGRESS_YIELD
            ):
                raise ExecutionError(
                    f"backend {dispatch.backend.backend!r} deferred a mediated egress "
                    "boundary but is not durable_pre_egress_yield"
                )
            result = self._capture_local_request(
                self._pending_egress_requests(),
                task.task_id,
                result,
                dispatch.model_binding,
            )
        elif self._is_resident_boundary(result, dispatch.model_binding):
            result = capture_resident_request(
                self._resident_requests(), task.task_id, result
            )
        if result.kind is HarnessResultKind.BOUNDARY and result.request is not None:
            _LOG.info(
                "[fabric] episode yielded a %s boundary (interface=%s)",
                result.request.kind.value,
                result.request.interface or "-",
            )
        value = result.value if result.kind is HarnessResultKind.COMPLETION else None
        group = facade.take_captured_group(task.task_id) if facade is not None else None
        sealed = None
        if state is not None and holder is not None:
            # The step has run to its yield, so the components are quiescent and seal as
            # one generation the next resume binds.
            sealed = holder.seal(state, _attachment(dispatch))
        return EpisodeStepResult(
            harness_result=result,
            value=value,
            facade_group=group,
            private_state=sealed,
        )

    def _open_private_state(
        self, task: ExecutorTask, dispatch: AgentEpisodeDispatch
    ) -> tuple[MaterializedState | None, PrivateStateHolder | None]:
        """Materialize the activation's bound generation for this dispatch."""
        binding = dispatch.private_state
        if binding is None:
            return None, None
        holder = PrivateStateHolder(self._config.private_state_dir)
        try:
            state = holder.open(
                binding,
                _attachment(dispatch),
                legacy_home=legacy_codex_home(
                    self._config.results_dir, task.workflow_id, task.task_id
                ),
            )
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
            store.put(task_id, req.call_correlation, model)
            digest = model_request_digest(model.interface, model.url, model.body)
        else:
            parsed = parse_search_request(req.request_payload)
            store.put(task_id, req.call_correlation, parsed)
            digest = tool_request_digest(
                parsed.interface, parsed.query, parsed.max_results
            )
        stripped = req.model_copy(
            update={"request_payload": None, "request_digest": digest}
        )
        return result.model_copy(update={"request": stripped})

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
        if self._adapter is not None:
            self._adapter.cancel(task_id)

    def cleanup_after_run(self) -> None:
        facade = self._lifecycle.responses_facade if self._lifecycle else None
        if facade is not None and self._episode_task_id is not None:
            facade.unregister_episode(self._episode_task_id)
        self._episode_task_id = None
        self._adapter = None


def _attachment(dispatch: AgentEpisodeDispatch) -> PrivateStateAttachment:
    """The dispatch's write authority, without which no holder may materialize state."""
    if dispatch.private_state_attachment is None:
        raise ExecutionError("an agent private-state binding ships with its attachment")
    return dispatch.private_state_attachment
