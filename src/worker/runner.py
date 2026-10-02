"""Task runner that executes assignments relayed by the server."""

import logging
import socket
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from shared.content import (
    ContentReference,
    ContentStoreError,
    FabricObjectStore,
)
from shared.harness.adapter import HarnessResultKind
from shared.inference import (
    CanonicalInferenceRequest,
    InputResolutionError,
    ResolvedCanonicalInferenceRequest,
    ResolvedInputMaterialization,
    canonical_result,
    parse_resolved_input,
    write_resolved_input,
)
from shared.network.mtls import MutualTlsMaterial
from shared.network.relay_frame import SSH_FRAME_KIND
from shared.outcome import FabricContentStore
from shared.schemas.result import RESULT_MEDIA_TYPE, BaseExecutorResult
from shared.tasks.credentials import dispatched_credentials
from shared.tasks.specs import (
    EmbeddingSpecStrict,
    InferenceBackend,
    InferenceSpecStrict,
    SSHSpecStrict,
    TaskSpecStrictBase,
)
from shared.tasks.worker_message import HardwareUsage, WorkerHardware, WorkerTaskMessage
from shared.telemetry.config import (
    DISABLED_TELEMETRY_CONFIG,
    TelemetryConfig,
    TelemetryLevel,
)
from shared.telemetry.propagation import extract_context
from shared.telemetry.provider import payload_free_span
from shared.telemetry.semconv import (
    LOGICAL_WORKFLOW_ID,
    PHYSICAL_TASK_ID,
    PHYSICAL_WORKER_ID,
    SPAN_TASK,
)
from shared.tools.contract import MediatedOperationPermit
from shared.tools.model.schema import MODEL_INTERFACE
from shared.tools.search.schema import DEFAULT_SEARCH_PROVIDER
from shared.utils.hardware import available_devices, gpus_fit_dispatch
from shared.utils.manifest import prepare_output_dir, sync_manifest
from shared.utils.recent import RecentSet
from shared.utils.redact import credential_scrubber
from shared.utils.time import now_iso

from .content.inputs import TaskInputHydrator, input_unreadable, read_input
from .egress import MediatedEgressSidecar, ModelEgress, SearchEgress
from .executors.base_executor import ExecutionError, Executor, TaskCancelledError
from .executors.episode_support import EpisodeStepResult, discard_step_captures
from .executors.inference.projection import generated_outputs
from .executors.inference.resolution import resolve_task_contract
from .executors.utils.checkpoints import write_executor_result
from .lifecycle import Lifecycle
from .model_turn import HeldModelEgress, ModelTurnRendezvous, ResponsesFacade
from .resident.lane_host import ResidentLaneHost
from .telemetry import otel
from .utils.logging import TaskLogEmitter


def _publishes_result(result: BaseExecutorResult) -> bool:
    """Whether a result is the task's terminal value rather than an episode yield.

    A run-to-yield step that is not the episode's completion hands its lane back and
    resumes later, so only the completing step carries the value the task settles with.
    """
    if isinstance(result, EpisodeStepResult):
        return result.harness_result.kind is HarnessResultKind.COMPLETION
    return True


# The supervisor kills a worker's container 30 seconds after stopping it, so a shutdown
# unregisters within this budget, leaving the process time to exit.
_STOP_BUDGET_SEC = 25.0
# How long, within the budget, a shutdown waits for the boundaries it holds to finish;
# the rest is left to the teardown.
_BOUNDARY_DRAIN_SEC = 15.0
_BOUNDARY_DRAIN_POLL_SEC = 0.1
# A revoke or stop may name a dispatch that never reaches this worker, so only the
# most recent are remembered.
_ENDED_DISPATCH_MEMORY = 1024


def _note_end(
    by_task: set[str],
    by_dispatch: RecentSet[str],
    task_id: str,
    dispatch_id: str | None,
) -> None:
    """Record a cancel or stop for the dispatch it names, or for the task when it
    names none."""
    if dispatch_id is None:
        by_task.add(task_id)
    else:
        by_dispatch.add(dispatch_id)


def _declared_result(
    result: BaseExecutorResult, request: CanonicalInferenceRequest | None
) -> BaseExecutorResult | None:
    """Rewrite a result into the shape its contract declares, or None to store it as is.

    A leaf whose contract the fabric resolves stores one result shape wherever it ran,
    so a consumer reading its output cannot tell which embodiment produced it. It runs
    before the result is stored, so the shape does not depend on the result reaching any
    other node. A step that generated nothing yet has nothing to declare.
    """
    if request is None:
        return None
    outputs = generated_outputs(result, request)
    return None if outputs is None else canonical_result(request, outputs)


class Runner:
    def __init__(
        self,
        lifecycle: Lifecycle,
        task_stream: Iterable[WorkerTaskMessage],
        results_dir: Path,
        hardware: WorkerHardware,
        executors: dict[str, Executor],
        default_executor: Executor,
        logger: logging.Logger,
        executor_idle_cleanup_sec: float | None = None,
        web_search_provider: str = DEFAULT_SEARCH_PROVIDER,
        web_search_api_key: str | None = None,
        model_api_key: str | None = None,
        model_egress_timeout_sec: float = 120.0,
        peer_enabled: bool = False,
        peer_material: MutualTlsMaterial | None = None,
        peer_listener_sock: socket.socket | None = None,
        telemetry: TelemetryConfig | None = None,
    ):
        otel.configure(telemetry or DISABLED_TELEMETRY_CONFIG)
        self.lifecycle = lifecycle
        self.task_stream = task_stream
        self.results_dir = results_dir
        self.hardware = hardware
        self.executors = executors
        self.logger = logger
        self.default_executor = default_executor
        self._input_hydrator = TaskInputHydrator(lifecycle.content_plane)
        self._peer_enabled = peer_enabled
        self._peer_material = peer_material
        self._peer_listener_sock = peer_listener_sock
        # How long to keep an executor alive (seconds) after its last use before calling
        # `cleanup_after_run()`. None or <=0 disables delayed cleanup.
        assert (
            executor_idle_cleanup_sec is None or executor_idle_cleanup_sec >= 0
        ), "executor_idle_cleanup_sec must be None or non-negative"
        self.executor_idle_cleanup_sec = executor_idle_cleanup_sec

        # Track a single active executor instance for reuse and its metadata
        self._active_executor: Executor | None = None
        self._active_executor_key: str | None = None
        self._active_executor_last_used_at: float | None = None
        # Whether the loaded executor has run anything GPU-bound since it came up.
        self._active_executor_used_gpu = False
        # Lock to protect concurrent access to active executor state
        self._active_executor_lock = threading.Lock()

        # Background idle checker thread and control event
        self._idle_checker_thread: threading.Thread | None = None
        self._idle_checker_stop_event: threading.Event | None = None
        # Poll interval in seconds for the idle checker
        self._idle_check_interval: float = 1.0

        # Interrupt monitoring thread and control event
        self._interrupt_thread: threading.Thread | None = None
        self._interrupt_stop_event: threading.Event | None = None
        self._mediated_op_thread: threading.Thread | None = None
        self._mediated_op_stop = threading.Event()
        self._current_task_id: str | None = None
        self._pending_cancels: set[str] = set()
        # The dispatch the task loop holds.
        self._current_dispatch_id: str | None = None
        # The task and dispatch the active executor runs. A cancel reaches the executor
        # only for the dispatch it runs; one landing earlier ends that dispatch before
        # it starts.
        self._executing: tuple[str, str | None] | None = None
        # Dispatches revoked or given up by a re-registration, cancelled, or stopped,
        # which never run.
        self._revoked_dispatches: RecentSet[str] = RecentSet(_ENDED_DISPATCH_MEMORY)
        self._cancelled_dispatches: RecentSet[str] = RecentSet(_ENDED_DISPATCH_MEMORY)
        self._stopped_dispatches: RecentSet[str] = RecentSet(_ENDED_DISPATCH_MEMORY)
        self._pending_stops: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._shutdown_requested = threading.Event()
        self._shutdown_thread: threading.Thread | None = None
        self._stop_deadline: float | None = None
        self._boundaries_closed = threading.Event()
        # Orders building a boundary lane against the drain closing them, so a lane
        # built by a racing frame is one the shutdown sees and stops.
        self._boundary_lanes_lock = threading.Lock()

        self._web_search_provider = web_search_provider
        self._web_search_api_key = web_search_api_key
        self._model_api_key = model_api_key
        self._model_egress_timeout_sec = model_egress_timeout_sec
        # The worker-local mediated-egress sidecar, built on the first permit relayed
        # over the attachment (once the worker id and incarnation are known).
        self._mediated_sidecar: MediatedEgressSidecar | None = None
        # Rendezvous for a held model turn's permit: a facade arms a waiter before it
        # proposes, so a permit relayed over the attachment wakes the held turn instead
        # of driving the async sidecar lane.
        self._model_turn_rendezvous = ModelTurnRendezvous()
        # The worker-local Responses facade, built alongside the sidecar once the worker
        # id is known and the first agent episode arrives.
        self._responses_facade: ResponsesFacade | None = None
        # The worker-local resident lanes (origin driver + replica sidecar) on their own
        # asyncio loop, built on the first resident control frame relayed over the
        # attachment (once the worker id is known).
        self._resident_host: ResidentLaneHost | None = None

    def has_active_gpu_executor(self) -> bool:
        """Whether the loaded executor may still hold GPU memory.

        Executors stay warm between tasks, so a reading taken while one is resident
        includes the worker's own model. The flag comes from each task's own dispatch
        rather than the executor class: the wrapper an executor loads behind carries no
        such attribute, and a transformers executor's device depends on the spec it
        ran. Read lock-free, since the availability monitor needs only a best-effort
        snapshot.
        """
        return self._active_executor is not None and self._active_executor_used_gpu

    def _refuse_if_gpu_is_held(self, msg: WorkerTaskMessage) -> None:
        """Refuse a dispatch whose GPUs another tenant holds.

        The dispatcher filters on what this worker last reported, so a task can still
        arrive for a device taken since; refusing beats dying in executor init. Only a
        reading just taken counts: a latched one advises the dispatcher, but a latch
        cannot clear while a GPU executor stays warm.
        """
        availability = self.lifecycle.live_gpu_availability()
        if not availability:
            return
        gpu = self.hardware.gpu
        devices = [
            (
                device.model_copy(update={"gpu_available": reported.available})
                if (reported := availability.get(device.uuid)) is not None
                else device
            )
            for device in gpu.devices
        ]
        hardware = self.hardware.model_copy(
            update={"gpu": gpu.model_copy(update={"devices": devices})}
        )
        if gpus_fit_dispatch(hardware, msg.spec, msg.relays_only):
            return
        held = len(devices) - len(available_devices(devices))
        raise ExecutionError(
            f"{held} of this worker's {len(devices)} GPU(s) are held by a process "
            "outside FlowMesh, so this task cannot run here",
            retryable=True,
        )

    def _note_gpu_usage(self, msg: WorkerTaskMessage) -> None:
        """Record that the loaded executor has run something GPU-bound.

        Called for every task, since one reusing a warm executor never re-enters the
        load branch and the memory it allocates outlives it. Monotone until teardown:
        a later CPU task does not free what an earlier GPU task allocated. A declared
        GPU request alone allocates nothing, so only what the spec loads counts, and an
        SSH session releases its devices when it ends, so it leaves nothing warm.
        """
        spec = msg.spec
        self._active_executor_used_gpu |= (
            not msg.relays_only
            and not isinstance(spec, SSHSpecStrict)
            and spec.uses_gpu()
        )

    def _cancel_active_executor(self) -> None:
        with self._active_executor_lock:
            executor = self._active_executor
            if executor is None:
                return
            current_task_id = self._current_task_id
            if current_task_id is not None:
                try:
                    executor.cancel(current_task_id)
                except Exception as exc:
                    self.logger.debug(
                        "Error cancelling active executor during shutdown: %s", exc
                    )

    def _cleanup_active_executor_in_budget(self) -> None:
        """Clean up the active executor, within what is left of a requested stop."""
        if self._stop_deadline is None:
            self._cleanup_active_executor()
            return
        cleanup = threading.Thread(
            target=self._cleanup_active_executor, name="executor-cleanup", daemon=True
        )
        cleanup.start()
        cleanup.join(self._stop_time_left())
        if cleanup.is_alive():
            self.logger.warning("Leaving the executor's cleanup unfinished at shutdown")

    def _cleanup_active_executor(self) -> None:
        self._cancel_active_executor()
        with self._active_executor_lock:
            executor = self._active_executor
            if executor is None:
                return
            try:
                executor.cleanup_after_run()
            except Exception as exc:
                self.logger.debug(
                    "Error cleaning up active executor during shutdown: %s", exc
                )
            finally:
                self._active_executor = None
                self._active_executor_key = None
                self._active_executor_last_used_at = None
                self._active_executor_used_gpu = False

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    @property
    def stop_deadline(self) -> float | None:
        """The monotonic time a requested shutdown must unregister by."""
        return self._stop_deadline

    def stop(self) -> None:
        """Request shutdown; safe from a signal handler.

        The work runs on its own thread, since the frame a signal interrupts may hold a
        lock that cancelling the active executor takes.
        """
        if self._shutdown_requested.is_set():
            return
        self._stop_deadline = time.monotonic() + _STOP_BUDGET_SEC
        self._shutdown_requested.set()
        self.lifecycle.begin_draining()
        thread = threading.Thread(
            target=self._shut_down, name="worker-shutdown", daemon=True
        )
        self._shutdown_thread = thread
        thread.start()

    def _shut_down(self) -> None:
        drain_deadline = time.monotonic() + _BOUNDARY_DRAIN_SEC
        self.logger.info("Shutdown requested; giving up the running task")
        self.lifecycle.set_draining()
        self.lifecycle.stop()
        self._cancel_active_executor()
        self._finish_held_boundaries(drain_deadline)
        with self._boundary_lanes_lock:
            self._boundaries_closed.set()
            sidecar = self._mediated_sidecar
            facade = self._responses_facade
            host = self._resident_host
        if sidecar is not None:
            sidecar.stop()
        if facade is not None:
            facade.stop(min(5.0, self._stop_time_left()))
        if host is not None:
            host.stop(min(15.0, self._stop_time_left()))

    def _stop_time_left(self) -> float:
        """Seconds left of the stop budget, or the budget when no stop was asked."""
        if self._stop_deadline is None:
            return _STOP_BUDGET_SEC
        return max(0.0, self._stop_deadline - time.monotonic())

    def _finish_held_boundaries(self, deadline: float) -> None:
        """Wait until ``deadline`` for control to commit the outcome of each boundary
        this worker holds for a suspended step.

        The permits that run them and the reaps that acknowledge their outcomes keep
        arriving until the worker unregisters, and an outcome reported after it
        unregisters would find the boundary already failed as this worker's loss.
        """
        while held := self._drained_boundaries():
            if time.monotonic() >= deadline:
                self.logger.warning(
                    "Leaving %d unfinished boundaries at shutdown: %s", len(held), held
                )
                return
            time.sleep(_BOUNDARY_DRAIN_POLL_SEC)

    def _drained_boundaries(self) -> list[tuple[str, str]]:
        """The held boundaries a shutdown waits for: a held model turn belongs to the
        running step, which the shutdown gives up."""
        return [
            (task_id, call)
            for task_id, call in self.lifecycle.held_boundaries()
            if not self._model_turn_rendezvous.has_waiter(task_id, call)
        ]

    def _ensure_mediated_sidecar(self) -> MediatedEgressSidecar | None:
        """Build the mediated-egress sidecar once the worker id is known."""
        if self._mediated_sidecar is not None:
            return self._mediated_sidecar
        client = self.lifecycle.client
        try:
            client.worker_id
        except RuntimeError:
            return None
        self._mediated_sidecar = MediatedEgressSidecar(
            pending_requests=self.lifecycle.pending_egress_requests,
            audience=lambda: (client.worker_id, client.incarnation),
            egresses=(
                SearchEgress(
                    self._web_search_provider, self._web_search_api_key, self.logger
                ),
                ModelEgress(self._model_api_key, self.logger),
            ),
            outcome_sink=client.push_mediated_outcome,
            content_store_for=self._outcome_store,
            logger=self.logger,
        )
        return self._mediated_sidecar

    def _ensure_responses_facade(self) -> ResponsesFacade | None:
        """Build and start the worker-local Responses facade once the worker id is set.

        A held Codex episode runs each model turn through this facade: it proposes the
        request digest, awaits the one-use permit over the rendezvous, and egresses
        synchronously through the same mediated-egress sidecar. It is reachable to the
        agent-episode executor through the lifecycle.
        """
        if self._responses_facade is not None:
            return self._responses_facade
        sidecar = self._ensure_mediated_sidecar()
        if sidecar is None:
            return None
        client = self.lifecycle.client
        held_egress = HeldModelEgress(
            rendezvous=self._model_turn_rendezvous,
            pending=self.lifecycle.pending_egress_requests,
            propose=client.push_mediated_propose,
            sidecar=sidecar,
            timeout_sec=self._model_egress_timeout_sec,
            logger=self.logger,
        )
        facade = ResponsesFacade(
            held_egress=held_egress,
            pending=self.lifecycle.pending_egress_requests,
            logger=self.logger,
        )
        facade.start()
        self._responses_facade = facade
        self.lifecycle.responses_facade = facade
        return facade

    def _ensure_resident_host(self) -> ResidentLaneHost | None:
        """Build the resident lane host once the worker id is known."""
        with self._boundary_lanes_lock:
            return self._ensure_resident_host_locked()

    def _ensure_resident_host_locked(self) -> ResidentLaneHost | None:
        if self._resident_host is not None:
            return self._resident_host
        if self._boundaries_closed.is_set():
            # A host built now would outlive the shutdown that stops the lanes.
            self.logger.warning(
                "Dropping a resident frame that arrived after the shutdown's "
                "boundary drain"
            )
            return None
        client = self.lifecycle.client
        try:
            client.worker_id
        except RuntimeError:
            return None
        host = ResidentLaneHost(
            push_frame=client.push_resident_frame,
            report_ack=client.push_resident_ack,
            report_outcome=client.push_resident_outcome,
            report_observation=client.push_resident_route_observation,
            content_store_for=self._outcome_store,
            peek_request=self.lifecycle.resident_requests.peek,
            delete_request=self.lifecycle.resident_requests.delete,
            peer_enabled=self._peer_enabled,
            peer_material=self._peer_material,
            peer_listener_sock=self._peer_listener_sock,
            logger=self.logger,
        )
        host.start()
        self._resident_host = host
        return host

    def _route_mediated_op(self, frame_kind: str, frame: dict[str, Any]) -> None:
        if frame_kind.startswith("resident_"):
            if (host := self._ensure_resident_host()) is not None:
                host.route(frame_kind, frame)
            return
        if frame_kind == SSH_FRAME_KIND:
            if (lane := self.lifecycle.ssh_relay) is not None:
                lane.route(frame_kind, frame)
            return
        if frame_kind.startswith("content_"):
            if (plane := self.lifecycle.content_plane) is not None:
                plane.route(frame_kind, frame)
            return
        if frame_kind == "deny":
            # A held model turn's denial: only a facade waiter consumes it.
            self._model_turn_rendezvous.deliver_deny(
                str(frame["agent_task_id"]),
                str(frame["call_correlation"]),
                str(frame.get("reason", "denied")),
            )
            return
        if frame_kind == "permit":
            permit = MediatedOperationPermit.model_validate(frame)
            # A held facade armed a waiter before proposing: hand it the permit for a
            # synchronous in-turn egress. Otherwise it drives the async sidecar lane.
            if self._model_turn_rendezvous.deliver_permit(permit):
                return
            # A held-turn MODEL permit whose waiter is gone is stale (its turn timed
            # out, or a duplicate): the deferred-op lane would egress it again and reap
            # a concurrent retry's private request, so drop it — recovery re-proposes it
            # under a fresh permit. A durable-yield MODEL permit never armed a waiter,
            # so it drives the async lane like any other.
            stale_held = permit.interface == MODEL_INTERFACE and (
                self._model_turn_rendezvous.was_held(
                    permit.agent_task_id, permit.call_correlation
                )
            )
            if stale_held:
                return
            with self._boundary_lanes_lock:
                if self._boundaries_closed.is_set():
                    self.logger.warning(
                        "Dropping a permit for %s:%s that arrived after the shutdown's "
                        "boundary drain",
                        permit.agent_task_id,
                        permit.call_correlation,
                    )
                    return
                if (sidecar := self._ensure_mediated_sidecar()) is not None:
                    sidecar.submit_permit(permit)
            return
        if frame_kind == "reap":
            if (sidecar := self._ensure_mediated_sidecar()) is not None:
                sidecar.reap(
                    str(frame["agent_task_id"]), str(frame["call_correlation"])
                )
            return
        self.logger.warning("Unknown mediated-op frame kind: %s", frame_kind)

    def _object_store(self, task_id: str) -> FabricObjectStore | None:
        """The content surface this task reads and writes through."""
        if (plane := self.lifecycle.content_plane) is None:
            return None
        return plane.for_task(task_id)

    def _outcome_store(self, task_id: str) -> FabricContentStore | None:
        """Where this task's outcomes materialize and deduplicate."""
        if (plane := self.lifecycle.content_plane) is None:
            return None
        return plane.outcome_store(task_id)

    def _prepare_inputs(self, msg: WorkerTaskMessage) -> ResolvedInputMaterialization:
        """Resolve a task's declared contract and store the request it materialized.

        The bytes are written before the binding that names them is reported, so a loss
        anywhere here leaves an object no resolution claims rather than a resolution
        pointing at nothing.
        """
        store = self._object_store(msg.task_id)
        if store is None:
            raise ExecutionError(
                f"task {msg.task_id} prepares its inputs, and this worker reaches no "
                "fabric content store to store the request in",
                retryable=True,
            )
        resolved = self._resolve_contract(msg)
        if resolved is None:
            raise ExecutionError(
                f"task {msg.task_id} was dispatched to prepare inputs and carries no "
                "contract to resolve",
                retryable=False,
            )
        reference = write_resolved_input(store, msg.content_scope, resolved)
        return ResolvedInputMaterialization(
            binding=resolved.binding, reference=reference
        )

    def _resolve_contract(
        self, msg: WorkerTaskMessage
    ) -> ResolvedCanonicalInferenceRequest | None:
        """The request a task's contract names, resolved against its pinned upstream."""
        if msg.declared_contract is None:
            return None
        try:
            return resolve_task_contract(msg)
        except InputResolutionError as exc:
            raise ExecutionError(str(exc), retryable=False) from exc

    def _raise_if_cancel_pending(self, task_id: str) -> None:
        """Honor a cancel, a revoke of its dispatch, or the worker's shutdown that
        landed before the task's executor runs it, such as while it read its inputs."""
        with self._cancel_lock:
            ended = self._ended_before_execution_locked(task_id)
        if ended is not None:
            raise TaskCancelledError(ended)

    def _begin_execution(self, task_id: str) -> bool:
        """Hand the task to its executor unless it ended before execution; return
        whether a stop for it is pending."""
        with self._cancel_lock:
            ended = self._ended_before_execution_locked(task_id)
            if ended is None:
                self._executing = (task_id, self._current_dispatch_id)
            stop_pending = (
                task_id in self._pending_stops
                or self._current_dispatch_id in self._stopped_dispatches
            )
        if ended is not None:
            raise TaskCancelledError(ended)
        return stop_pending

    def _ended_before_execution_locked(self, task_id: str) -> str | None:
        cancelled = (
            task_id in self._pending_cancels
            or self._current_dispatch_id in self._cancelled_dispatches
        )
        self._pending_cancels.discard(task_id)
        if cancelled:
            return f"Task {task_id} was cancelled before execution"
        if self._current_dispatch_id in self._revoked_dispatches:
            return f"Task {task_id} was revoked before execution"
        if self._shutdown_requested.is_set():
            return f"Task {task_id} was given up by the worker shutting down"
        return None

    def _materialize_contract(self, msg: WorkerTaskMessage) -> None:
        """Settle the one request a task runs, before its embodiment reaches a model.

        A task carrying a prepared request hydrates and verifies it, so the run issues
        exactly what the preparation produced and reads no source syntax. Otherwise the
        contract resolves here, and the binding recording how it was reached is durable
        before a local generation or a resident service issue. Either way a source that
        does not resolve within what the leaf declared fails the task where no model has
        run and no claim exists.
        """
        if msg.declared_contract is None:
            return
        if (prepared := msg.recorded_input) is not None:
            msg.resolved_contract = self._hydrate_prepared_request(
                msg, prepared
            ).request
            return
        resolved = self._resolve_contract(msg)
        if resolved is None:
            return
        committed = msg.recorded_resolution
        if committed is not None and not committed.matches(resolved.binding):
            raise ExecutionError(
                f"task {msg.task_id} is committed to the inputs it already resolved, "
                "and its source resolves to a different request now",
                retryable=False,
            )
        msg.resolved_contract = resolved.request
        self.lifecycle.notify_task_update(
            msg.task_id, {"input_resolution": resolved.binding.model_dump(mode="json")}
        )

    def _hydrate_prepared_request(
        self, msg: WorkerTaskMessage, reference: ContentReference
    ) -> ResolvedCanonicalInferenceRequest:
        """Fetch the prepared request this task runs, failing closed on anything else.

        Verification is what makes the reference safe to run from: content that is
        missing, out of the task's scope, or not the bytes its digest names fails the
        task before any model I/O and before any admission.
        """
        data = read_input(
            self.lifecycle.content_plane, msg.task_id, msg.content_scope, reference
        )
        try:
            hydrated = parse_resolved_input(reference, data)
        except ContentStoreError as exc:
            raise input_unreadable(
                f"task {msg.task_id} cannot hydrate the request its preparation "
                f"recorded: {exc}"
            ) from exc
        committed = msg.recorded_resolution
        if committed is not None and not committed.matches(hydrated.binding):
            raise ExecutionError(
                f"task {msg.task_id} is committed to the inputs it already resolved, "
                "and the request it hydrated records a different resolution",
                retryable=False,
            )
        return hydrated

    def _run_executor(
        self,
        executor: Executor,
        msg: WorkerTaskMessage,
        out_dir: Path,
    ) -> BaseExecutorResult | EpisodeStepResult | None:
        """Run the executor inside ``flowmesh.task``, entering the propagated context.

        Wraps every task type; an executor's own shipped ``task`` span, where it opens
        one, nests inside this one. At ``off`` this opens nothing and calls the executor
        directly, leaving the shipped span as the analyzer's only root.
        """
        if not otel.emits(TelemetryLevel.COARSE):
            return executor.run(msg, out_dir)
        parent_context = extract_context(msg.traceparent)
        attributes = otel.new_span_attributes(
            {
                PHYSICAL_TASK_ID: msg.task_id,
                LOGICAL_WORKFLOW_ID: msg.workflow_id,
                PHYSICAL_WORKER_ID: self.lifecycle.worker_id,
            }
        )
        with otel.workflow_trace_context(msg.workflow_id):
            with payload_free_span(
                otel.get_tracer(),
                SPAN_TASK,
                context=parent_context,
                attributes=attributes,
            ):
                return executor.run(msg, out_dir)

    def _resolve_output_dir(self, task_id: str) -> Path:
        """Prepare and return the canonical output directory for a task's results."""
        out_dir = self.results_dir / task_id
        prepare_output_dir(out_dir)
        return out_dir

    def _write_results(
        self,
        msg: WorkerTaskMessage,
        out_dir: Path,
        result: BaseExecutorResult | None,
    ) -> dict[str, Any]:
        """Store a task's result and each merged child's, returning their references.

        Every result lands in the shared store before the success that reports it, so a
        reference control binds always names bytes that outlive this worker. The
        task's own result carries none of its children's. A child the executor produced
        no result for reports none, and control runs it again on its own.
        """
        if result is None:
            return {}
        own = result.model_copy(update={"children": {}})
        declared = _declared_result(own, msg.resolved_contract) or own
        reference = self._write_single_result(
            msg, msg.task_id, msg.spec, out_dir, declared
        )
        if reference is None:
            return {}
        references: dict[str, Any] = {
            "result_reference": reference.model_dump(mode="json")
        }
        child_lookup = {entry.task_id: entry for entry in msg.merged_children or []}
        children: dict[str, Any] = {}
        for child_id, child_result in result.children.items():
            child_info = child_lookup.get(child_id)
            if child_info is None:
                continue
            child_reference = self._write_single_result(
                msg,
                child_id,
                child_info.spec,
                self._resolve_output_dir(child_id),
                child_result,
            )
            if child_reference is not None:
                children[child_id] = child_reference.model_dump(mode="json")
        if children:
            references["child_result_references"] = children
        return references

    def _write_single_result(
        self,
        msg: WorkerTaskMessage,
        task_id: str,
        spec: TaskSpecStrictBase,
        out_dir: Path,
        payload: BaseExecutorResult | None,
    ) -> ContentReference | None:
        if payload is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        envelope = write_executor_result(
            out_dir / "results.json", task_id, spec, payload
        )
        sync_manifest(out_dir, task_id, spec.get_artifacts())
        if not _publishes_result(payload):
            return None
        store = self._object_store(msg.task_id)
        if store is None:
            raise ExecutionError(
                f"task {task_id} produced a result and this worker reaches no fabric "
                "content store to store it in",
                retryable=True,
            )
        try:
            return store.write(
                msg.content_scope,
                envelope.encode("utf-8"),
                media_type=RESULT_MEDIA_TYPE,
            )
        except ContentStoreError as exc:
            raise ExecutionError(
                f"task {task_id} could not store its result: {exc}", retryable=True
            ) from exc

    def _select_inference_executor_key(self, spec: InferenceSpecStrict) -> str:
        if spec.backend() is InferenceBackend.TRANSFORMERS:
            return "default"
        if (model_cfg := spec.model) and model_cfg.adapters:
            return "vllm_lora"
        return "vllm"

    def _select_embedding_executor_key(self, spec: EmbeddingSpecStrict) -> str:
        if (model_cfg := spec.model) and model_cfg.vllm is not None:
            return "vllm_embedding"
        return "default"

    def _maybe_expire_active_executor(self) -> None:
        """Expire the active executor if it has been idle past the configured
        timeout.
        """
        with self._active_executor_lock:
            if not self._active_executor:
                return
            if not self.executor_idle_cleanup_sec:
                return
            if not self._active_executor_last_used_at:
                return
            idle = time.time() - self._active_executor_last_used_at
            if idle >= self.executor_idle_cleanup_sec:
                self.logger.info(
                    "Executor %s idle for %.1f sec, calling cleanup_after_run()",
                    self._active_executor_key,
                    idle,
                )
                self._active_executor.cleanup_after_run()
                self._active_executor = None
                self._active_executor_key = None
                self._active_executor_last_used_at = None
                self._active_executor_used_gpu = False

    def _idle_check_loop(self, stop_event: threading.Event) -> None:
        """Background loop that periodically checks for idle executors.

        The loop waits on `stop_event` with a timeout equal to
        `self._idle_check_interval` and calls `_maybe_expire_active_executor` each tick.
        """
        try:
            while not stop_event.wait(self._idle_check_interval):
                try:
                    self._maybe_expire_active_executor()
                except Exception as exc:
                    self.logger.debug("Idle checker encountered error: %s", exc)
        except Exception:
            # Ensure background thread doesn't propagate exceptions
            pass

    def _start_idle_checker(self) -> None:
        """Start the idle checker background thread if not already running.

        Only starts if `executor_idle_cleanup_sec` is configured and positive.
        """
        if not self.executor_idle_cleanup_sec or self.executor_idle_cleanup_sec <= 0:
            return
        if self._idle_checker_thread and self._idle_checker_thread.is_alive():
            return
        self._idle_checker_stop_event = threading.Event()
        # Use a small poll interval; ensure it's not larger than the cleanup timeout so
        # expiration happens reasonably soon after timed out.
        self._idle_check_interval = min(
            1.0, max(0.5, float(self.executor_idle_cleanup_sec) / 10.0)
        )
        t = threading.Thread(
            target=self._idle_check_loop, args=(self._idle_checker_stop_event,)
        )
        t.daemon = True
        t.name = "flowmesh-idle-checker"
        self._idle_checker_thread = t
        t.start()

    def _stop_idle_checker(self, timeout: float = 2.0) -> None:
        """Signal the idle checker thread to stop and wait briefly for join."""
        if not self._idle_checker_thread:
            return
        if self._idle_checker_stop_event:
            self._idle_checker_stop_event.set()
        try:
            self._idle_checker_thread.join(timeout=timeout)
        except Exception:
            pass
        finally:
            self._idle_checker_thread = None
            self._idle_checker_stop_event = None

    def abandon_running(self, dispatch_id: str | None) -> None:
        """Give up the dispatch this worker runs, as its registration has ended.

        Control resolves the dispatch through the previous registration's loss, so the
        task's executor is cancelled to end what it started, and its reports go
        nowhere. A serve task's replica stops admitting claims.
        """
        with self._cancel_lock:
            if dispatch_id is not None:
                self._revoked_dispatches.add(dispatch_id)
            task_id = (
                self._current_task_id
                if self._current_dispatch_id == dispatch_id
                else None
            )
            executing = task_id is not None and self._runs_locked(task_id, dispatch_id)
        with self._boundary_lanes_lock:
            resident_host = self._resident_host
            sidecar = self._mediated_sidecar
        if resident_host is not None:
            resident_host.unbind_replicas()
        if sidecar is not None:
            sidecar.forget_outcomes()
        if task_id is None:
            return
        self.logger.warning("Abandoning task %s: this worker re-registered", task_id)
        if executing:
            self._cancel_executor_run(task_id)

    def _cancel_executor_run(self, task_id: str) -> None:
        with self._active_executor_lock:
            executor = self._active_executor
        if executor is not None:
            try:
                executor.cancel(task_id)
            except Exception as exc:
                self.logger.warning("Executor cancel() raised: %s", exc)

    def _runs_locked(self, task_id: str, dispatch_id: str | None) -> bool:
        """Whether the active executor runs ``task_id``, as the dispatch named if one
        is."""
        executing = self._executing
        return (
            executing is not None
            and executing[0] == task_id
            and dispatch_id in (None, executing[1])
        )

    def _interrupt_monitor_loop(self, stop_event: threading.Event) -> None:
        try:
            while not stop_event.wait(0.5):
                try:
                    for (
                        task_id,
                        reason,
                        dispatch_id,
                    ) in self.lifecycle.client.iter_interrupts():
                        with self._cancel_lock:
                            _note_end(
                                self._pending_cancels,
                                self._cancelled_dispatches,
                                task_id,
                                dispatch_id,
                            )
                            running = self._runs_locked(task_id, dispatch_id)
                        if not running:
                            continue
                        self.logger.info(
                            "Interrupt for running task %s (reason=%s)", task_id, reason
                        )
                        self._cancel_executor_run(task_id)
                    for task_id, dispatch_id in self.lifecycle.client.iter_revokes():
                        with self._cancel_lock:
                            self._revoked_dispatches.add(dispatch_id)
                            running = self._runs_locked(task_id, dispatch_id)
                        if not running:
                            continue
                        self.logger.info(
                            "Revoked dispatch %s of running task %s",
                            dispatch_id,
                            task_id,
                        )
                        self._cancel_executor_run(task_id)
                    for (
                        task_id,
                        reason,
                        dispatch_id,
                    ) in self.lifecycle.client.iter_stops():
                        with self._cancel_lock:
                            _note_end(
                                self._pending_stops,
                                self._stopped_dispatches,
                                task_id,
                                dispatch_id,
                            )
                            running = self._runs_locked(task_id, dispatch_id)
                        if not running:
                            continue
                        self.logger.info(
                            "Graceful stop for running task %s (reason=%s)",
                            task_id,
                            reason,
                        )
                        with self._active_executor_lock:
                            executor = self._active_executor
                        if executor is not None:
                            try:
                                executor.stop(task_id)
                            except Exception as exc:
                                self.logger.warning("Executor stop() raised: %s", exc)
                except Exception as exc:
                    self.logger.warning("Interrupt monitor encountered error: %s", exc)
        except Exception:
            pass

    def _start_interrupt_monitor(self) -> None:
        if self._interrupt_thread and self._interrupt_thread.is_alive():
            return
        self._interrupt_stop_event = threading.Event()
        thread = threading.Thread(
            target=self._interrupt_monitor_loop,
            args=(self._interrupt_stop_event,),
            daemon=True,
            name="flowmesh-interrupt-monitor",
        )
        self._interrupt_thread = thread
        thread.start()

    def _stop_interrupt_monitor(self, timeout: float = 2.0) -> None:
        if not self._interrupt_thread:
            return
        if self._interrupt_stop_event is not None:
            self._interrupt_stop_event.set()
        try:
            self._interrupt_thread.join(timeout=timeout)
        except Exception:
            pass
        finally:
            self._interrupt_thread = None
            self._interrupt_stop_event = None

    def _route_mediated_ops(self) -> None:
        """Route each mediated op as it arrives, in arrival order."""
        client = self.lifecycle.client
        while not self._mediated_op_stop.is_set():
            if (item := client.next_mediated_op(0.5)) is None:
                continue
            try:
                self._route_mediated_op(*item)
            except Exception as exc:
                self.logger.warning("Mediated-op routing failed: %s", exc)

    def _start_mediated_op_router(self) -> None:
        if self._mediated_op_thread and self._mediated_op_thread.is_alive():
            return
        self._mediated_op_stop.clear()
        thread = threading.Thread(
            target=self._route_mediated_ops,
            daemon=True,
            name="flowmesh-mediated-ops",
        )
        self._mediated_op_thread = thread
        thread.start()

    def _stop_mediated_op_router(self, timeout: float = 2.0) -> None:
        if self._mediated_op_thread is None:
            return
        self._mediated_op_stop.set()
        self._mediated_op_thread.join(timeout=timeout)
        self._mediated_op_thread = None

    def start(self) -> None:
        self._start_idle_checker()
        self._start_interrupt_monitor()
        self._start_mediated_op_router()
        try:
            for msg in self.task_stream:
                if self._shutdown_requested.is_set():
                    break
                assigned_worker = msg.assigned_worker
                if assigned_worker and assigned_worker != self.lifecycle.worker_id:
                    self.logger.info(
                        "Skipping task %s assigned to %s (this worker id=%s)",
                        msg.task_id,
                        assigned_worker,
                        self.lifecycle.worker_id,
                    )
                    continue

                task_id = msg.task_id
                with self._cancel_lock:
                    self._current_dispatch_id = msg.dispatch_id
                spec = msg.spec
                task_type = spec.taskType
                scrub = _task_scrubber(msg)

                parent_task_id = msg.parent_task_id
                shard_index = msg.shard_index
                shard_total = msg.shard_total
                dispatched_at = msg.dispatched_at

                extra = []
                if parent_task_id:
                    extra.append(f"parent={parent_task_id}")
                if shard_index is not None and shard_total is not None:
                    extra.append(f"shard={shard_index}/{shard_total}")
                extra_info = ", ".join(extra) if extra else "no-parent"

                self.logger.info(
                    "Received task %s (type=%s, %s) with spec keys %s",
                    task_id,
                    task_type or "unknown",
                    extra_info,
                    sorted(key for key, value in spec if value is not None),
                )

                task_log_emitter: TaskLogEmitter | None = None
                log_handler_attached: bool = False
                prev_root_log_level: int | None = None
                out_dir = self._resolve_output_dir(task_id)
                self.lifecycle.set_busy(task_id)
                # Set task start time in case of early failure
                start_iso = now_iso()
                start_wall = time.time()
                notified_task_started: bool = False
                # A step's captures reach control only on its success report; until it
                # is sent, the worker drops them if the task ends another way.
                unreported_step: EpisodeStepResult | None = None
                try:
                    self._raise_if_cancel_pending(task_id)
                    self._current_task_id = task_id
                    self._refuse_if_gpu_is_held(msg)
                    self._input_hydrator.hydrate(msg)
                    if msg.input_preparation:
                        self._raise_if_cancel_pending(task_id)
                        self.lifecycle.notify_task_started(
                            task_id,
                            task_type=task_type,
                            dispatched_at=dispatched_at,
                            started_at=start_iso,
                        )
                        notified_task_started = True
                        prepared = self._prepare_inputs(msg)
                        metadata = self._build_task_metadata(
                            task_type,
                            dispatched_at,
                            start_iso,
                            start_wall,
                            shard_index=shard_index,
                            shard_total=shard_total,
                        )
                        metadata["input_materialization"] = prepared.model_dump(
                            mode="json"
                        )
                        self.lifecycle.set_succeeded(task_id, metadata=metadata)
                        self.logger.info("Task %s prepared its inputs", task_id)
                        continue
                    if msg.service_episode is not None:
                        # A resident service-backed leaf runs the service-episode path
                        # (capture the model request, yield a resident boundary, resume
                        # on the settled completion) rather than loading a local model.
                        desired_key = "service_leaf"
                    elif task_type == "inference":
                        assert isinstance(spec, InferenceSpecStrict)
                        desired_key = self._select_inference_executor_key(spec)
                    elif task_type == "diffusion":
                        desired_key = "diffusers"
                    elif task_type == "embedding":
                        assert isinstance(spec, EmbeddingSpecStrict)
                        desired_key = self._select_embedding_executor_key(spec)
                    elif task_type == "serve":
                        desired_key = "vllm_serve"
                    elif task_type == "agent":
                        if msg.agent_episode is None:
                            raise ExecutionError(
                                "agent task reached the worker without a resolved "
                                "harness binding; every agent runs the harness "
                                "episode path"
                            )
                        # A held backend runs its model turns through the worker-local
                        # facade; build it before the executor binds an adapter.
                        self._ensure_responses_facade()
                        desired_key = "agent_episode"
                    else:
                        desired_key = "default" if task_type is None else task_type
                    # Acquire lock before accessing/modifying active executor
                    with self._active_executor_lock:
                        if (
                            self._active_executor
                            and self._active_executor_key != desired_key
                        ):
                            self.logger.info(
                                "New task requests executor %s, shutting down active "
                                "executor %s",
                                desired_key,
                                self._active_executor_key,
                            )
                            self._active_executor.cleanup_after_run()
                            self._active_executor = None
                            self._active_executor_key = None
                            self._active_executor_used_gpu = False

                        if not self._active_executor:
                            if (
                                desired_key == "service_leaf"
                                and "service_leaf" not in self.executors
                            ):
                                # A resident leaf must run the service-episode path;
                                # never fall back to a local model executor, which would
                                # run the model on this worker and yield a boundary the
                                # resident path never settles.
                                raise ExecutionError(
                                    f"task {task_id} requires the service-leaf "
                                    "executor for its resident service binding, but "
                                    "it is not available on this worker"
                                )
                            self._active_executor = self.executors.get(
                                desired_key, self.default_executor
                            )
                            self._active_executor_key = desired_key
                        self._note_gpu_usage(msg)

                        (
                            task_log_emitter,
                            log_handler_attached,
                            prev_root_log_level,
                        ) = self._create_task_logger(task_id, msg, out_dir, scrub)

                        # Notify task started just before execution
                        start_iso = now_iso()
                        start_wall = time.time()
                        self.lifecycle.notify_task_started(
                            task_id,
                            task_type=task_type,
                            dispatched_at=dispatched_at,
                            started_at=start_iso,
                        )
                        notified_task_started = True

                        self._materialize_contract(msg)

                        # Disable idle checker during execution
                        self._active_executor_last_used_at = None
                        executor_to_run = self._active_executor
                        # A stop that landed before the executor was bound is handed to
                        # it here, under the lock the stop's delivery reads it with.
                        if self._begin_execution(task_id):
                            executor_to_run.stop(task_id)
                    try:
                        out = self._run_executor(executor_to_run, msg, out_dir)
                    finally:
                        # A cancel past this point would wait in the warm executor and
                        # end the task's next dispatch.
                        with self._cancel_lock:
                            self._executing = None
                    if isinstance(out, EpisodeStepResult):
                        unreported_step = out
                    references = self._write_results(msg, out_dir, out)
                    metadata = self._build_task_metadata(
                        task_type,
                        dispatched_at,
                        start_iso,
                        start_wall,
                        shard_index=shard_index,
                        shard_total=shard_total,
                    )
                    metadata.update(references)
                    if isinstance(out, EpisodeStepResult):
                        # The step rides the success metadata so the server routes the
                        # boundary and re-dispatches; the attempt still ends here, which
                        # is what releases the lane. A captured facade group rides the
                        # same metadata so control routes it with the completion.
                        step = out.harness_result
                        if step.error is not None:
                            step = step.model_copy(update={"error": scrub(step.error)})
                        metadata["agent_episode"] = step.model_dump(mode="json")
                        if out.facade_group is not None:
                            metadata["agent_episode_facade_group"] = (
                                out.facade_group.model_dump(mode="json")
                            )
                        if out.private_state is not None:
                            metadata["agent_episode_private_state"] = (
                                out.private_state.model_dump(mode="json")
                            )
                    self.lifecycle.set_succeeded(task_id, metadata=metadata)
                    unreported_step = None
                    self.logger.info("Task %s completed successfully", task_id)
                except TaskCancelledError as e:
                    if not notified_task_started:
                        self.lifecycle.notify_task_started(
                            task_id,
                            task_type=task_type,
                            dispatched_at=dispatched_at,
                            started_at=start_iso,
                        )
                        notified_task_started = True
                    metadata = self._build_task_metadata(
                        task_type,
                        dispatched_at,
                        start_iso,
                        start_wall,
                        shard_index=shard_index,
                        shard_total=shard_total,
                    )
                    self.lifecycle.set_cancelled(task_id, metadata=metadata)
                    self.logger.info("Task %s cancelled: %s", task_id, scrub(str(e)))
                except Exception as e:
                    if not notified_task_started:
                        self.lifecycle.notify_task_started(
                            task_id,
                            task_type=task_type,
                            dispatched_at=dispatched_at,
                            started_at=start_iso,
                        )
                        notified_task_started = True
                    metadata = self._build_task_metadata(
                        task_type,
                        dispatched_at,
                        start_iso,
                        start_wall,
                        shard_index=shard_index,
                        shard_total=shard_total,
                    )
                    controlled = e if isinstance(e, ExecutionError) else None
                    self.lifecycle.set_failed(
                        task_id,
                        scrub(str(e)),
                        metadata=metadata,
                        retryable=controlled is None or controlled.retryable,
                        failure_kind=controlled.failure_kind if controlled else None,
                        unavailable_inputs=(
                            controlled.unavailable_inputs if controlled else ()
                        ),
                    )
                    if isinstance(e, ExecutionError):
                        self.logger.error("Task %s failed: %s", task_id, scrub(str(e)))
                    else:
                        self.logger.error(
                            "Task %s failed\n%s",
                            task_id,
                            scrub(traceback.format_exc()),
                        )
                finally:
                    if unreported_step is not None:
                        discard_step_captures(self.lifecycle, task_id, unreported_step)
                    self._current_task_id = None
                    with self._cancel_lock:
                        self._executing = None
                        self._current_dispatch_id = None
                        self._pending_cancels.discard(task_id)
                        self._pending_stops.discard(task_id)
                        if (dispatch_id := msg.dispatch_id) is not None:
                            self._revoked_dispatches.discard(dispatch_id)
                            self._cancelled_dispatches.discard(dispatch_id)
                            self._stopped_dispatches.discard(dispatch_id)
                    with self._active_executor_lock:
                        self._active_executor_last_used_at = time.time()
                    self.lifecycle.set_idle(task_id)
                    if task_log_emitter is not None:
                        if log_handler_attached:
                            logging.getLogger().removeHandler(task_log_emitter)
                            self.logger.removeHandler(task_log_emitter)
                        try:
                            task_log_emitter.close()
                        except Exception:
                            pass
                    if prev_root_log_level is not None:
                        logging.getLogger().setLevel(prev_root_log_level)
        except KeyboardInterrupt:
            self.logger.info("Runner interrupted by user; shutting down task loop")
        finally:
            if self._shutdown_thread is not None:
                self._shutdown_thread.join()
            self._cleanup_active_executor_in_budget()
            self._stop_interrupt_monitor(min(2.0, self._stop_time_left()))
            self._stop_mediated_op_router(min(2.0, self._stop_time_left()))
            self._stop_idle_checker(min(2.0, self._stop_time_left()))

    def _create_task_logger(
        self,
        task_id: str,
        msg: WorkerTaskMessage,
        out_dir: Path,
        scrub: Callable[[str], str],
    ) -> tuple[TaskLogEmitter | None, bool, int | None]:
        task_log_emitter: TaskLogEmitter | None = None
        log_handler_attached = False
        prev_root_log_level: int | None = None
        try:
            owner_mismatch = False
            log_paths: dict[str, Path] = {task_id: self._get_log_path(out_dir)}
            task_refs: list[dict[str, str]] = [
                {"task_id": task_id, "workflow_id": msg.workflow_id}
            ]
            if msg.merged_children:
                for entry in msg.merged_children:
                    child_id = entry.task_id
                    child_workflow_id = entry.workflow_id or msg.workflow_id
                    task_refs.append(
                        {"task_id": child_id, "workflow_id": child_workflow_id}
                    )
                    child_out_dir = self._resolve_output_dir(child_id)
                    log_paths[child_id] = self._get_log_path(child_out_dir)
                    if entry.owner_id != msg.owner_id:
                        owner_mismatch = True
                        continue

            if owner_mismatch:
                task_log_emitter = self.lifecycle.client.create_task_log_emitter(
                    task_id=task_id,
                    workflow_id=msg.workflow_id,
                    owner_id=msg.owner_id,
                    task_refs=task_refs,
                    log_paths=log_paths,
                )
                if task_log_emitter is not None:
                    task_log_emitter.emit_warning_only(
                        "Task-level logs are not supported for merged tasks "
                        "with different owners."
                    )
                else:
                    self.logger.warning(
                        "Task-level logs are not supported for merged tasks "
                        "with different owners."
                    )
                task_log_emitter = None
            else:
                task_log_emitter = self.lifecycle.client.create_task_log_emitter(
                    task_id=task_id,
                    workflow_id=msg.workflow_id,
                    owner_id=msg.owner_id,
                    task_refs=task_refs,
                    log_paths=log_paths,
                    scrub=scrub,
                )
                if task_log_emitter is not None:
                    root_logger = logging.getLogger()
                    root_logger.addHandler(task_log_emitter)
                    self.logger.addHandler(task_log_emitter)
                    log_handler_attached = True
                    prev_root_log_level = root_logger.level
                    desired_level = self.logger.level
                    if prev_root_log_level > desired_level:
                        root_logger.setLevel(desired_level)
        except Exception:
            task_log_emitter = None
            log_handler_attached = False
            prev_root_log_level = None

        return task_log_emitter, log_handler_attached, prev_root_log_level

    @staticmethod
    def _get_log_path(out_dir: Path) -> Path:
        return out_dir / "logs" / "logs.jsonl"

    def _build_task_metadata(
        self,
        task_type: str | None,
        dispatched_at: str | None,
        started_at: str,
        start_wall: float,
        shard_index: int | None = None,
        shard_total: int | None = None,
    ) -> dict[str, Any]:
        finished_at = now_iso()
        runtime_sec = max(0.0, time.time() - start_wall)
        hw_usage = HardwareUsage.from_hw(self.hardware).model_dump()
        cost_per_hour = self.lifecycle.cost_per_hour
        metadata = {
            "taskType": task_type,
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_sec": runtime_sec,
            "hardware": hw_usage,
            "cost_per_hour": cost_per_hour,
            "total_cost": (cost_per_hour / 3600.0) * runtime_sec,
        }
        if dispatched_at:
            metadata["dispatched_at"] = dispatched_at
        if shard_index is not None:
            metadata["shard_index"] = shard_index
        if shard_total is not None:
            metadata["shard_total"] = shard_total
        return metadata


def _task_scrubber(msg: WorkerTaskMessage) -> Callable[[str], str]:
    """Masks the credentials the dispatch restored into its task and merged children."""
    specs: dict[str, BaseModel] = {msg.task_id: msg.spec}
    specs.update((child.task_id, child.spec) for child in msg.merged_children or ())
    return credential_scrubber(
        value
        for task_id, pointers in msg.credential_pointers.items()
        if (spec := specs.get(task_id)) is not None
        for value in dispatched_credentials(spec, pointers)
    )
