"""The off-lane executor for a worker-originated mediated tool operation.

The agent's own worker captured the tool request in worker-private state and proposed
only its digest; central control returned a one-use permit and dispatched this operation
back to the same worker. This executor reads the raw request from that local state,
validates the permit and digest against the same worker fence the attachment path uses,
egresses through the local provider, and reports a permit-fenced typed outcome or an
immutable outcome reference. The network-fenced harness never runs here: this is a
separate, ordinary worker executor, and the raw request never crossed to the control
plane.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

from shared.outcome import FabricContentStore, OutcomeManifest
from shared.schemas.result import BaseExecutorResult
from shared.tasks.task_type import TaskType
from shared.tools.contract import (
    MediatedOperationPermit,
    ToolOperationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from shared.tools.search.egress import ExternalToolSidecar
from shared.tools.search.providers import LazySearchProvider
from shared.tools.search.schema import SEARCH_INTERFACE, ToolRequest

from ..content_store import build_content_store
from ..tool_fence import ProviderBinding, fence_reason, materialize_tool_outcome
from .base_executor import ExecutionError, Executor, ExecutorTask

# The durable worker logger (console + rotating worker.log), so the execution-locus
# audit marker below is observable in the worker's own logs — like the attachment-path
# executor, which is handed the same logger — rather than only on the root task stream.
_LOG = logging.getLogger("worker")

_INTERFACES = frozenset({SEARCH_INTERFACE})


class ToolOperationResult(BaseExecutorResult):
    """One off-lane tool operation's fenced outcome, for the server to settle.

    Exactly one of ``outcome`` (a bounded typed control datum) or ``outcome_ref`` (a
    reference to materialized content) is set; the server injects it at the boundary.
    """

    outcome: ToolOutcome | None = None
    outcome_ref: OutcomeManifest | None = None


class ToolOperationExecutor(Executor):
    """Validate a permit and egress one worker-originated tool operation locally."""

    name = "tool_operation"
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.TOOL_OPERATION}
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sidecar = ExternalToolSidecar(
            LazySearchProvider(
                ProviderBinding(
                    self._config.web_search_provider, self._config.web_search_api_key
                )
            ),
            _LOG,
        )

    def run(self, task: ExecutorTask, out_dir: Path) -> ToolOperationResult:
        permit = task.tool_operation
        if permit is None:
            # A real operation is server-minted with a permit; a bare submission has
            # none, so fail closed rather than egressing.
            raise ExecutionError(
                f"{task.task_id} routed to the tool-operation executor without a permit"
            )
        store = build_content_store(self._config.server_base_url)
        # Recover an already-produced outcome first: a re-drive after a materialize that
        # the server never recorded (a crash between materialize and settle) returns the
        # prior result by its idempotency key, before consuming the worker-private
        # request — so a successful operation is never re-failed for a missing request.
        if (prior := self._prior_manifest(permit, store)) is not None:
            return ToolOperationResult(outcome_ref=prior)
        request = self._pending_tool_requests().take(
            permit.agent_task_id, permit.call_correlation
        )
        if request is None:
            # No prior outcome and the capturing worker incarnation is gone: the request
            # cannot be recovered here. Fail deterministically so the boundary settles
            # rather than egressing an unfenced request.
            raise ExecutionError(
                f"no worker-private request for {permit.call_correlation}"
            )
        if (reason := self._fence_reject(permit, request)) is not None:
            raise ExecutionError(f"tool permit fence rejected: {reason}")
        outcome = self._egress(permit, request)
        materialized = materialize_tool_outcome(
            outcome, idempotency_key=permit.idempotency_key, content_store=store
        )
        if isinstance(materialized, OutcomeManifest):
            return ToolOperationResult(outcome_ref=materialized)
        return ToolOperationResult(outcome=materialized)

    def _fence_reject(
        self, permit: MediatedOperationPermit, request: ToolRequest
    ) -> str | None:
        worker_id, generation = self._audience()
        return fence_reason(
            interface=permit.interface,
            target_id=permit.target_id,
            target_generation=permit.target_generation,
            policy_class=permit.policy_class,
            deadline_epoch=permit.deadline_epoch,
            request_digest=permit.request_digest,
            request=request,
            worker_id=worker_id,
            worker_generation=generation,
            allowed_interfaces=_INTERFACES,
            # Policy-class enforcement for this path lands when 7e/7f bind real policy
            # generations; the permit carries the class as a forward contract.
            expected_policy_class=None,
        )

    def _egress(
        self, permit: MediatedOperationPermit, request: ToolRequest
    ) -> ToolOutcome:
        envelope = ToolOperationEnvelope(
            interface=permit.interface,
            idempotency_key=permit.idempotency_key,
            max_results=min(request.max_results, permit.max_results),
            timeout_sec=permit.timeout_sec,
            result_char_cap=permit.result_char_cap,
        )
        _LOG.info(
            "tool egress in worker=%s interface=%s", permit.target_id, permit.interface
        )
        try:
            return self._sidecar.execute(envelope, request)
        except ValueError as exc:
            # A misprovisioned provider is a deterministic fault: report it as a typed
            # terminal outcome rather than raising into an ambiguous retry loop.
            _LOG.warning("tool provider unavailable: %s", exc)
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the external-tool provider is unavailable",
            )

    @staticmethod
    def _prior_manifest(
        permit: MediatedOperationPermit, store: FabricContentStore | None
    ) -> OutcomeManifest | None:
        """A prior materialization under this operation's idempotency key, if any."""
        if store is None or permit.idempotency_key is None:
            return None
        return store.find(permit.idempotency_key)

    def _audience(self) -> tuple[str, int]:
        """This worker's id and incarnation, resolved lazily once it is registered."""
        if self._lifecycle is None:
            raise ExecutionError("tool-operation executor has no worker lifecycle")
        client = self._lifecycle.client
        return client.worker_id, client.incarnation
