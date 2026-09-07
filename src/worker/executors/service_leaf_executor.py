"""The run-to-yield executor for a resident service-backed inference leaf.

A leaf whose binding requires resident capacity runs here instead of loading a model in
the worker. Its first step builds the model request from the task spec, keeps it in
worker-private resident custody, and yields one resident model boundary carrying only a
digest; the fabric admits a ``ServiceClaim`` and drives the worker-originated resident
protocol. A resume injects the settled completion and finishes the leaf. There is no
harness, scope, or child region — the model request is the leaf's whole body.
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from shared.harness import (
    BoundaryEventKind,
    BoundaryRequest,
    DeliveredOutcome,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)
from shared.tasks.specs import EmbeddingSpecStrict, InferenceSpecStrict
from shared.tasks.task_type import TaskType
from shared.tools.model.schema import MODEL_INTERFACE

from ..resident import capture_resident_request
from .base_executor import ExecutionError, Executor, ExecutorTask
from .episode_support import EpisodeStepResult, hydrate_delivered_outcomes

_LOG = logging.getLogger("service-leaf-executor")

# The single resident model boundary a service leaf emits; stable across a re-drive so
# the worker-private request and its settled outcome key to one occurrence.
_CALL_CORRELATION = "resident-model/0"

_PROMPT_FIELDS = ("prompt", "input", "content", "text")

# The resident transport serves a chat/completions interface.
_CHAT_INTERFACE = "chat"


class ServiceLeafExecutor(Executor):
    """Drive one run-to-yield step of a resident service-backed leaf."""

    name = "service_leaf"
    # Selected by the service-episode dispatch signal, never by task-type capability: a
    # worker running this executor relays a resident invocation, so it advertises no
    # local inference/embedding capability of its own.
    supported_task_types: ClassVar[frozenset[TaskType]] = frozenset()

    def run(self, task: ExecutorTask, out_dir: Path) -> EpisodeStepResult:
        dispatch = task.service_episode
        if dispatch is None:
            raise ExecutionError(
                f"{task.task_id} routed to the service-leaf executor without a "
                "service-episode dispatch context"
            )
        outcomes = hydrate_delivered_outcomes(
            self._config.server_base_url, dispatch.delivered_outcomes
        )
        if dispatch.interface != _CHAT_INTERFACE:
            raise ExecutionError(
                f"resident {dispatch.interface} execution is not supported; the "
                "resident transport serves a chat/completions interface"
            )
        settled = next(
            (o for o in outcomes if o.call_correlation == _CALL_CORRELATION), None
        )
        if settled is not None:
            return self._complete(settled)
        return self._yield_boundary(task, dispatch.interface)

    def _yield_boundary(self, task: ExecutorTask, interface: str) -> EpisodeStepResult:
        payload = _resident_request_payload(task, interface)
        request = BoundaryRequest(
            kind=BoundaryEventKind.INVOCATION,
            interface=MODEL_INTERFACE,
            call_correlation=_CALL_CORRELATION,
            request_payload=payload,
        )
        result = capture_resident_request(
            self._resident_requests(),
            task.task_id,
            HarnessResult(kind=HarnessResultKind.BOUNDARY, request=request),
        )
        _LOG.info(
            "[fabric] service leaf %s yielded a resident model boundary", task.task_id
        )
        return EpisodeStepResult(harness_result=result)

    @staticmethod
    def _complete(settled: DeliveredOutcome) -> EpisodeStepResult:
        if settled.kind is OutcomeKind.DENIED:
            return EpisodeStepResult(
                harness_result=HarnessResult(
                    kind=HarnessResultKind.FAILURE,
                    error=f"resident invocation denied: {settled.denial}",
                )
            )
        if settled.kind is OutcomeKind.CANCELLED:
            return EpisodeStepResult(
                harness_result=HarnessResult(kind=HarnessResultKind.CANCELLATION)
            )
        value = settled.value or ""
        return EpisodeStepResult(
            harness_result=HarnessResult(
                kind=HarnessResultKind.COMPLETION, value=value
            ),
            value=value,
        )

    def cancel(self, task_id: str) -> None:
        return None


def _resident_request_payload(task: ExecutorTask, interface: str) -> str:
    """Build the resident engine request payload from the leaf's spec.

    The payload is the model request the replica serves: an explicit chat ``messages``
    array, or a bare prompt the engine wraps as one user message. The prompt is read
    from ``spec.inference`` or ``spec.data``.
    """
    spec = task.spec
    data: dict[str, Any] = (
        spec.data
        if isinstance(spec, (InferenceSpecStrict, EmbeddingSpecStrict))
        and isinstance(spec.data, dict)
        else {}
    )
    inference: dict[str, Any] = (
        spec.inference
        if isinstance(spec, InferenceSpecStrict) and isinstance(spec.inference, dict)
        else {}
    )

    for source in (inference, data):
        if isinstance(messages := source.get("messages"), list):
            return json.dumps({"messages": messages})

    for source in (data, inference):
        for field in _PROMPT_FIELDS:
            if isinstance(value := source.get(field), str) and value:
                return value
        if isinstance(prompts := source.get("prompts"), list) and prompts:
            return str(prompts[0])

    raise ExecutionError(
        f"resident {interface} leaf {task.task_id} declares no prompt or messages "
        "in spec.data or spec.inference"
    )
