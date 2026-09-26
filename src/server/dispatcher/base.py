import datetime
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from server.telemetry.tracing import NULL_CONTROL_TRACER, ControlPlaneTracer
from shared.private_state import OwnerFence, PrivateStateUnavailableReason
from shared.schemas.artifact import ArtifactRef
from shared.schemas.event import TaskEvent
from shared.schemas.result import (
    BaseExecutorResult,
    ResultEnvelope,
)
from shared.tasks import (
    MergedChildTaskStrict,
    TaskEnvelope,
    TaskEnvelopeStrict,
    TaskEnvelopeTemplate,
    TaskSpecStrict,
)
from shared.tasks.placeholders import PLACEHOLDER_PATTERN
from shared.tasks.result_binding import ResultBinding
from shared.tasks.specs import (
    ConditionSpec,
    InferenceEmbodimentKind,
    SSHSpecStrict,
    SSHSpecTemplate,
)
from shared.tasks.worker_message import WorkerStatus, WorkerTaskMessage
from shared.telemetry.semconv import ControlPlaneStage, ControlPlaneWindow
from shared.utils.ids import new_dispatch_id

from ..clients.redis import REDIS_CONN_ERRORS
from ..content import ContentAccessBroker
from ..registries.worker import Worker, WorkerRegistry
from ..services.metrics import MetricsRecorder
from ..task.metadata import extract_model_dataset_names
from ..task.models import DispatchEnd, TaskRecord, TaskStatus
from ..task.results import ResultUnavailable
from ..task.runtime import TaskRuntime
from ..task.v2.representations.plan import InferenceEmbodimentMenu
from ..utils.time import now_iso
from .embodiment import (
    EmbodimentSelector,
    EmbodimentSnapshot,
    PrimaryEmbodimentSelector,
    relay_placement_task,
)
from .worker_selector import DEFAULT_WORKER_SELECTION, select_worker

_SENTINEL: Any = object()

_NO_WORKER_BACKOFF_SEC = 0.5
_ERROR_BACKOFF_SEC = 1.0


class StageReferenceNotReady(Exception):
    """Raised when a task references a stage whose artifacts are not yet available."""


class StageResultMissing(ValueError):
    """Raised when a settled stage a task references has no result to read."""


class Dispatcher:
    """Handles FCFS task dispatching via Redis pub/sub."""

    def __init__(
        self,
        runtime: TaskRuntime,
        worker_registry: WorkerRegistry,
        logger: logging.Logger,
        worker_selection_strategy: str = DEFAULT_WORKER_SELECTION,
        enable_context_reuse: bool = True,
        enable_task_merge: bool = True,
        task_merge_max_batch_size: int = 4,
        reuse_cache_ttl_sec: int = 3600,
        lambda_config: dict[str, float] | None = None,
        selection_jitter_epsilon: float = 1e-3,
        enable_stage_weight_stickiness: bool = False,
        no_worker_grace_sec: int = 60,
        metrics_recorder: MetricsRecorder | None = None,
        resident_capacity_enabled: bool = False,
        resident_admission_slots: int = 0,
        embodiment_selector: EmbodimentSelector | None = None,
        content_access: ContentAccessBroker | None = None,
        control: ControlPlaneTracer | None = None,
    ) -> None:
        self._runtime = runtime
        self._worker_registry = worker_registry
        self._content_access = content_access
        self._logger = logger
        self._worker_selection_strategy = worker_selection_strategy
        self._context_reuse_enabled = enable_context_reuse
        self._task_merge_enabled = enable_task_merge
        self._task_merge_max_batch_size = max(1, task_merge_max_batch_size)
        self._cache_ttl_sec = max(0, reuse_cache_ttl_sec)
        self._lambda_config = lambda_config or {}
        self._selection_jitter = max(0.0, selection_jitter_epsilon)
        self._stage_weight_stickiness_enabled = enable_stage_weight_stickiness
        self._no_worker_grace_sec = max(0, no_worker_grace_sec)
        self._metrics = metrics_recorder
        self._resident_capacity_enabled = resident_capacity_enabled
        self._resident_admission_slots = max(0, resident_admission_slots)
        self._embodiment_selector = embodiment_selector or PrimaryEmbodimentSelector()
        self._control = control if control is not None else NULL_CONTROL_TRACER
        self._weight_reference_hints: tuple[str, ...] = (
            "checkpoint",
            "weight",
            "weights",
            "model",
            "adapter",
            "lora",
            "load",
            "artifact",
        )

    def eligible_worker_ids(self, record: TaskRecord, relay: bool = False) -> set[str]:
        """Worker ids whose hardware satisfies the task, honoring selected_worker."""
        task = relay_placement_task(record.task) if relay else record.task
        eligible = {
            worker.id for worker in self._worker_registry.satisfying_workers(task)
        }
        if record.selected_worker:
            eligible &= set(record.selected_worker)
        return eligible

    def _grace_then_fail(
        self,
        task_id: str,
        record: TaskRecord,
        reason: str,
        message: str,
        extra_payload: dict[str, Any] | None = None,
    ) -> bool:
        """Wait out the no-worker grace, then fail the task terminally."""
        now = time.time()
        if record.no_eligible_since is None:
            record.no_eligible_since = now
        waited = now - record.no_eligible_since
        if waited >= self._no_worker_grace_sec:
            self._logger.warning(
                "No worker for %s after %.0fs (%s); failing", task_id, waited, reason
            )
            payload = {"reason": reason}
            if extra_payload:
                payload.update(extra_payload)
            self.fail_task(
                task_id,
                record.last_error or message,
                worker_id=record.last_failed_worker,
                payload=payload,
            )
            return False
        self.requeue_task(task_id, reason=reason, count_retry=False)
        return False

    def _grace_then_fail_undeliverable(
        self,
        task_id: str,
        record: TaskRecord,
        reason: str,
        message: str,
        extra_payload: dict[str, Any] | None = None,
    ) -> bool:
        """Handle a selected worker that turned out undeliverable due to an
        infrastructure fault.

        Retry without spending the task's ``max_attempts`` budget and then fail after
        ``no_worker_grace_sec``.
        """
        self._runtime.release_merge(task_id)
        record.last_error = message
        now = time.time()
        if record.no_dispatch_since is None:
            record.no_dispatch_since = now
        waited = now - record.no_dispatch_since
        if waited >= self._no_worker_grace_sec:
            self._logger.warning(
                "Task %s undeliverable after %.0fs (%s); failing",
                task_id,
                waited,
                reason,
            )
            payload = {"reason": reason}
            if extra_payload:
                payload.update(extra_payload)
            self.fail_task(
                task_id, message, worker_id=record.last_failed_worker, payload=payload
            )
            return False
        self.requeue_task(task_id, reason=reason, front=True, count_retry=False)
        return False

    def _grace_then_fail_exhausted(
        self, task_id: str, record: TaskRecord, failed_ids: set[str]
    ) -> bool:
        """Grace-then-fail once every eligible worker has failed the task."""
        return self._grace_then_fail(
            task_id,
            record,
            reason="eligible_workers_exhausted",
            message="All eligible workers failed the task",
            extra_payload={"failed_workers": sorted(failed_ids)},
        )

    def _private_state_owner_loss(
        self, owner: OwnerFence
    ) -> PrivateStateUnavailableReason | None:
        """Why a bound generation's holder cannot supply it, or None when it can."""
        worker = self._worker_registry.get_worker(owner.worker_id)
        if worker is None or self._worker_registry.is_worker_stale(owner.worker_id):
            return PrivateStateUnavailableReason.OWNER_LOST
        if worker.incarnation != owner.incarnation:
            return PrivateStateUnavailableReason.INCARNATION_MISMATCH
        return None

    def _fail_private_state_unavailable(
        self,
        task_id: str,
        record: TaskRecord,
        owner: OwnerFence,
        reason: PrivateStateUnavailableReason,
    ) -> bool:
        """Fail an episode whose private state no live holder can supply.

        A brief grace absorbs a heartbeat flap; past it the episode fails closed rather
        than resuming against an empty home on another worker.
        """
        now = time.time()
        if record.no_eligible_since is None:
            record.no_eligible_since = now
        if now - record.no_eligible_since < self._no_worker_grace_sec:
            self.requeue_task(
                task_id, reason="private_state_owner_lost", count_retry=False
            )
            return False
        self._logger.warning(
            "Private state for %s cannot be supplied by %s (%s); failing closed",
            task_id,
            owner.worker_id,
            reason.value,
        )
        self.fail_task(
            task_id,
            f"PrivateStateUnavailable: {reason.value}",
            payload={
                "reason": reason.value,
                "private_state_owner": owner.worker_id,
                "private_state_incarnation": str(owner.incarnation),
            },
        )
        return False

    def _resolve_embodiment(self, task_id: str, record: TaskRecord) -> bool:
        """Bind a menu node to one embodiment, or defer holding no worker.

        A task whose node offers no menu passes straight through, as does one whose
        embodiment is already committed: the choice is re-resolvable only while no
        attempt has carried it to a worker and no invocation has issued.

        A primary that can never be placed would otherwise defer forever, so a
        continuously deferred menu reaches the same grace-then-fail a task no worker can
        satisfy does. It fails rather than switching: an embodiment the author did not
        declare stays unreachable.
        """
        menu = self._runtime.embodiment_menu(task_id)
        if menu is None or self._runtime.embodiment_pinned(task_id):
            return True
        snapshot = self._embodiment_snapshot(record)
        batch_size = self._selection_batch_size(task_id, menu)
        if batch_size is None:
            # The node declares no bound and no preparation reported one, so every
            # candidate would be screened against nothing. Waiting is the safe answer:
            # selecting here could admit a resident batch larger than it reserves.
            return self._grace_then_fail(
                task_id,
                record,
                reason="embodiment_deferred:unknown_batch_size",
                message=(
                    "No embodiment of the task can be placed: it declares no "
                    "conversation bound and its inputs have not been prepared"
                ),
            )
        decision = self._embodiment_selector(menu, snapshot, batch_size)
        if decision.alternative_id is None:
            self._logger.debug(
                "Deferring %s: no embodiment placeable (%s)",
                task_id,
                decision.defer_reason,
            )
            return self._grace_then_fail(
                task_id,
                record,
                reason=f"embodiment_deferred:{decision.defer_reason}",
                message=(
                    "No embodiment of the task can be placed: its declared primary "
                    f"is unavailable ({decision.defer_reason})"
                ),
            )
        bound = self._runtime.record_embodiment_selection(
            task_id,
            decision.alternative_id,
            self._embodiment_selector.name,
            snapshot.evidence(),
        )
        if bound is None:
            # The work item went away mid-dispatch; without a durable selection there is
            # no fence, so the task waits rather than running an unrecorded embodiment.
            self.requeue_task(
                task_id, reason="embodiment_unrecorded", count_retry=False
            )
            return False
        record.no_eligible_since = None
        return True

    def _selection_batch_size(
        self, task_id: str, menu: InferenceEmbodimentMenu
    ) -> int | None:
        """The conversation count a menu node's candidates are screened against.

        A declared bound screens before any value exists. A node without one is screened
        against the count its preparation actually materialized.
        """
        if menu.max_batch_size is not None:
            return menu.max_batch_size
        binding = self._runtime.input_resolution_binding(task_id)
        return binding.cardinality if binding is not None else None

    def _embodiment_snapshot(self, record: TaskRecord) -> EmbodimentSnapshot:
        """Read live feasibility for one menu node. It reserves nothing."""
        return EmbodimentSnapshot(
            local_capable_workers=len(self.eligible_worker_ids(record)),
            relay_capable_workers=len(self.eligible_worker_ids(record, relay=True)),
            resident_capacity_enabled=self._resident_capacity_enabled,
            resident_admission_slots=self._resident_admission_slots,
        )

    def _relays_only(self, task_id: str) -> bool:
        """Whether this dispatch carries an invocation rather than running a model.

        A resident-served embodiment runs its model on a replica, so the local model
        requirement its leaf declares for the other embodiment does not apply to the
        worker that carries the invocation.
        """
        resolved = self._runtime.resolved_embodiment(task_id)
        return (
            resolved is not None
            and resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED
        )

    def dispatch_once(self, task_id: str) -> bool:
        """Dispatch a single task if possible; requeue when no worker.

        Wrapped in the ``dispatch`` control-plane span for a v2 episode: the dispatcher
        loop thread has no ambient span, so the span's parent is built explicitly from
        the episode (work item) id rather than relied upon. A v1 task, or a v2 task
        whose work item is not yet known, dispatches with no span at all.
        """
        if not self._control.enabled:
            return self._dispatch_once_impl(task_id)
        record = self._runtime.get_record(task_id)
        engine = (
            self._runtime.orchestration_engine(record.workflow_id)
            if record is not None
            else None
        )
        work_item_id = (
            engine.work_item_id_for_task(task_id) if engine is not None else None
        )
        if record is None or work_item_id is None:
            return self._dispatch_once_impl(task_id)
        with self._control.episode_stage(
            ControlPlaneStage.DISPATCH,
            ControlPlaneWindow.QUEUE,
            record.workflow_id,
            work_item_id,
        ):
            return self._dispatch_once_impl(task_id)

    def _dispatch_once_impl(self, task_id: str) -> bool:
        record = self._runtime.get_record(task_id)
        if not record:
            return True

        # A leaf whose source declares no envelope is prepared first: this dispatch
        # resolves its inputs on a worker and reports the request it materialized, and
        # the dispatch after it chooses an embodiment knowing what that request holds.
        preparing = self._runtime.prepares_inputs(task_id)

        # Resolve and durably bind the embodiment before anything else this dispatch
        # does: publication precedes attempt bookkeeping, so a choice recorded after it
        # would not survive a loss between the two.
        if not preparing and not self._resolve_embodiment(task_id, record):
            return False

        # Live-feasibility handoff: defer an episode whose declared alternative is not
        # feasible to place now, holding no worker. It admits no capacity object.
        if not preparing and not self._runtime.episode_feasible(task_id):
            self.requeue_task(
                task_id, reason="infeasible_alternative", count_retry=False
            )
            return False

        task = record.task
        # Placement reads the resolved embodiment, never the leaf's own binding: a
        # resident-served dispatch carries the invocation and loads no model locally.
        # A preparation places the same way: it reads an upstream value and runs no
        # model, so it needs no accelerator either.
        relays_only = preparing or self._relays_only(task_id)
        placement_task = relay_placement_task(task) if relays_only else task

        model_names, dataset_names = extract_model_dataset_names(task)
        task_category = (
            (record.category or "other").lower()
            if hasattr(record, "category")
            else "other"
        )
        task_age = None
        if getattr(record, "last_queue_ts", None) is not None:
            task_age = max(0.0, time.time() - record.last_queue_ts)

        # 1. Get idle worker pool
        pool = self._worker_registry.idle_satisfying_pool(placement_task)

        # 2. Filter by selected_worker hint if present
        if record.selected_worker:
            pool = [c for c in pool if c.id in record.selected_worker]

        # 2b. Owner-affine private state: a bound generation is sealed on the holder
        # that produced it, so the episode waits for that incarnation instead of
        # resuming against a fresh or foreign one. Waiting holds no worker. This
        # governs a holder lost with no external effect in flight; an ambiguous
        # in-flight effect settles terminally in the ledger before placement is asked.
        if (owner := self._runtime.private_state_owner(task_id)) is not None:
            if (loss := self._private_state_owner_loss(owner)) is not None:
                return self._fail_private_state_unavailable(
                    task_id, record, owner, loss
                )
            pool = [
                c
                for c in pool
                if c.id == owner.worker_id and c.incarnation == owner.incarnation
            ]
            if not pool:
                record.no_eligible_since = None
                self.requeue_task(
                    task_id, reason="private_state_owner_busy", count_retry=False
                )
                return False

        failed_ids = set(record.failed_workers)
        if owner is not None:
            # The owner is the only holder that can supply the bound generation, so a
            # failure there is retried on it under the attempt budget. Diverting to an
            # untried worker would wait on workers 2b has already excluded, which never
            # become selectable — an unbounded requeue that reaches no terminal.
            failed_ids = set()

        # 3. No idle worker: wait for a busy one, or grace-then-fail when no worker can
        # take the task, or every eligible worker has already failed it.
        if not pool:
            eligible = self.eligible_worker_ids(record, relay=relays_only)
            if not eligible:
                return self._grace_then_fail(
                    task_id,
                    record,
                    reason="no_eligible_worker",
                    message="No worker satisfies the task hardware and capability "
                    "requirements",
                )
            if not (eligible - failed_ids):
                return self._grace_then_fail_exhausted(task_id, record, failed_ids)
            record.no_eligible_since = None
            reason = (
                "candidate_workers_busy" if record.selected_worker else "no_idle_worker"
            )
            self._logger.debug("No idle worker available for %s; requeueing", task_id)
            self.requeue_task(task_id, reason=reason, count_retry=False)
            return False

        # 4. Prefer workers that have not failed this task.
        if failed_ids:
            filtered_pool = [c for c in pool if c.id not in failed_ids]
            if filtered_pool:
                pool = filtered_pool
            elif self.eligible_worker_ids(record, relay=relays_only) - failed_ids:
                # Untried eligible workers exist but are busy; wait for them.
                record.no_eligible_since = None
                self._logger.debug(
                    "All idle candidates for %s already failed it; waiting for an "
                    "untried worker",
                    task_id,
                )
                self.requeue_task(
                    task_id, reason="untried_workers_busy", count_retry=False
                )
                return False
            else:
                # Every eligible worker has failed this task; grace-then-fail so a
                # newly joined worker can still pick it up.
                return self._grace_then_fail_exhausted(task_id, record, failed_ids)

        record.no_eligible_since = None

        # 5. Worker selection (best-fit scoring by default)
        selection_info: dict[str, Any] = {}
        worker: Worker | None = None

        sticky_worker_id: str | None = None
        if (not record.selected_worker) and self._stage_weight_stickiness_enabled:
            sticky_worker_id = self._preferred_stage_worker(record, task)
            selection_info = {
                "strategy": "stage_weight_affinity",
                "sticky_worker": sticky_worker_id,
                "candidate_pool": len(pool),
                "chosen_metrics": {"score": 1.0},
            }

        if sticky_worker_id:
            sticky_candidate = next(
                (item for item in pool if item.id == sticky_worker_id), None
            )
            if sticky_candidate:
                worker = sticky_candidate
                self._logger.debug(
                    "Using sticky worker %s for %s",
                    sticky_worker_id,
                    task_id,
                )
            else:
                if self._should_wait_for_sticky_worker(record, sticky_worker_id):
                    self._logger.debug(
                        "Preferred worker %s busy for %s; waiting for availability",
                        sticky_worker_id,
                        task_id,
                    )
                    self.requeue_task(
                        task_id, reason="sticky_worker_busy", count_retry=False
                    )
                    return False
                else:
                    self._logger.debug(
                        "Preferred worker %s unavailable for %s; falling "
                        "back to normal selection",
                        sticky_worker_id,
                        task_id,
                    )

        preferred_pool: list[Worker] = []
        if worker is None and self._context_reuse_enabled:
            preferred_pool = self._cached_worker_candidates(
                pool, model_names, dataset_names
            )
        if worker is None:
            candidate_pool = preferred_pool or pool
            if preferred_pool:
                self._logger.debug(
                    "Preferring cached worker candidates for %s "
                    "(models=%s datasets=%s)",
                    task_id,
                    model_names,
                    dataset_names,
                )

            worker, selection_info = select_worker(
                candidate_pool,
                self._worker_selection_strategy,
                logger=self._logger,
                task_category=task_category,
                lambda_overrides=self._lambda_config,
                task_id=task_id,
                jitter_epsilon=self._selection_jitter,
                task_age=task_age,
            )
            if not worker and preferred_pool:
                worker, selection_info = select_worker(
                    pool,
                    self._worker_selection_strategy,
                    logger=self._logger,
                    task_category=task_category,
                    lambda_overrides=self._lambda_config,
                    task_id=task_id,
                    jitter_epsilon=self._selection_jitter,
                    task_age=task_age,
                )
        if not worker:
            self._logger.debug(
                "No suitable worker selected for %s; requeueing", task_id
            )
            self.requeue_task(task_id, reason="no_selection")
            return False

        # Plan task merge: coalesce sibling merge candidates onto this worker
        merged_children: list[str] = []
        if (
            self._task_merge_enabled
            and self._task_merge_max_batch_size > 1
            and not preparing
        ):
            merged_children = self._runtime.plan_merge(
                task_id, self._task_merge_max_batch_size, worker.id
            )
            if merged_children:
                self._logger.debug(
                    "Coalesced task %s with siblings %s",
                    task_id,
                    ", ".join(merged_children),
                )

        # 6. Resolve stage references
        try:
            rendered_task, upstream_results = self._resolve_stage_references(
                task_id, task, record
            )
            self._validate_ssh_inputs(record, rendered_task.spec)
        except StageReferenceNotReady as exc:
            self._logger.debug("Task %s waiting on stage artifacts: %s", task_id, exc)
            self.requeue_task(
                task_id, reason="stage_reference_pending", count_retry=False
            )
            return False
        except ResultUnavailable as exc:
            # The store is unreachable, not the result lost: wait it out without
            # spending the task's attempts. A missing or corrupt result fails below.
            self._logger.warning(
                "Task %s cannot reach a referenced stage result yet: %s", task_id, exc
            )
            self.requeue_task(
                task_id, reason="stage_result_unavailable", count_retry=False
            )
            return False
        except ValidationError as exc:
            self._runtime.release_merge(task_id)
            self.fail_task(
                task_id,
                "task_schema_validation_failed",
                payload={"error": str(exc)},
            )
            return True
        except Exception as exc:
            self._logger.error(
                "Failed to resolve stage references for %s: %s", task_id, exc
            )
            self._runtime.release_merge(task_id)
            self.fail_task(task_id, str(exc), payload={"error": str(exc)})
            return True

        # Re-validate resolved task spec
        try:
            rendered_task.spec.validate_dispatchable()
        except ValueError as exc:
            self._runtime.release_merge(task_id)
            self.fail_task(
                task_id, "spec_validation_failed", payload={"error": str(exc)}
            )
            return True

        # Conditional execution: skip dispatch if condition not met
        if self._evaluate_condition_skip(task_id, rendered_task, record):
            return True

        rendered_children = self._render_merged_children(
            task_id, record, rendered_task.spec
        )

        # 7. Build WorkerTaskMessage
        agent_episode = self._runtime.agent_episode_dispatch(
            task_id, OwnerFence(worker_id=worker.id, incarnation=worker.incarnation)
        )
        dispatch_id = new_dispatch_id()
        message = WorkerTaskMessage(
            task_id=task_id,
            dispatch_id=dispatch_id,
            workflow_id=record.workflow_id,
            owner_id=record.owner_id,
            content_scope=record.org_id,
            task=rendered_task,
            task_type=record.task_type,
            assigned_worker=worker.id,
            dispatched_at=now_iso(),
            parent_task_id=None,
            shard_index=record.shard_index,
            shard_total=record.shard_total,
            merged_children=rendered_children,
            upstream_results=upstream_results,
            input_element=self._runtime.input_element(task_id),
            agent_episode=agent_episode,
            service_episode=self._runtime.service_episode_dispatch(task_id),
            declared_contract=self._runtime.declared_contract(task_id),
            recorded_resolution=self._runtime.input_resolution_binding(task_id),
            input_preparation=preparing,
            recorded_input=self._runtime.recorded_input_reference(task_id),
            traceparent=self._runtime.dispatch_traceparent(task_id),
        )

        # 8. Give the task what it reads and writes its content under, then publish it
        if not self._runtime.begin_publish(
            task_id, worker, dispatch_id, input_preparation=preparing
        ):
            return True
        try:
            if self._content_access is not None:
                self._content_access.issue(worker.id, task_id, record.org_id)
            receivers = self._worker_registry.publish_task(worker, message)
        except Exception as exc:
            if not self._runtime.abandon_publish(task_id):
                return True
            self._logger.warning(
                "Failed to publish task %s to worker %s: %s", task_id, worker.id, exc
            )
            return self._grace_then_fail_undeliverable(
                task_id,
                record,
                reason="publish_failed",
                message=f"Failed to publish task to worker {worker.id}: {exc}",
                extra_payload={"worker_id": worker.id, "node_id": worker.node_id},
            )

        if receivers <= 0:
            if not self._runtime.abandon_publish(task_id):
                return True
            self._logger.info(
                "Node %s dispatch channel has no subscriber; delaying task %s "
                "(worker %s)",
                worker.node_id,
                task_id,
                worker.id,
            )
            return self._grace_then_fail_undeliverable(
                task_id,
                record,
                reason="no_dispatch_subscriber",
                message=(
                    f"Selected worker {worker.id} on node {worker.node_id} is "
                    "undeliverable: nothing is subscribed to the node's dispatch "
                    "channel (the node may be down, restarting, or its worker "
                    "registration stale)"
                ),
                extra_payload={"worker_id": worker.id, "node_id": worker.node_id},
            )

        # 9. Mark dispatched
        record.no_dispatch_since = None
        live = self._runtime.mark_dispatched(task_id)
        if rendered_children:
            self._logger.info(
                "[TaskMerge] parent=%s merged_children=%d -> %s",
                task_id,
                len(rendered_children),
                ", ".join(child.task_id for child in rendered_children),
            )
        if live:
            try:
                self._worker_registry.update_worker_status(worker.id, WorkerStatus.BUSY)
            except Exception as exc:
                self._logger.debug(
                    "Failed to update worker %s status: %s", worker.id, exc
                )

        try:
            chosen_score = selection_info.get("chosen_metrics", {}).get("score")
            score_display = (
                f"{chosen_score:.4f}"
                if isinstance(chosen_score, (int, float))
                else "n/a"
            )
            self._logger.info(
                "Dispatch %s -> %s (strategy=%s score=%s age=%.2fs)",
                task_id,
                worker.id,
                selection_info.get("strategy"),
                score_display,
                task_age if task_age is not None else -1.0,
            )
        except Exception:
            pass
        return True

    def dispatch_loop(self, stop_event, poll_interval: float = 1.0) -> None:
        """Continuously dispatch ready tasks until stop_event is set."""
        while not stop_event.is_set():
            task_id = self._runtime.next_ready(stop_event, timeout=poll_interval)
            if not task_id:
                continue
            try:
                success = self.dispatch_once(task_id)
                if not success:
                    time.sleep(_NO_WORKER_BACKOFF_SEC)
            except REDIS_CONN_ERRORS as exc:
                # Control Redis dropped. The connection pool reconnects on the next
                # command, so back off and keep the loop alive rather than letting the
                # dispatcher thread die. Requeue is best-effort; if it also hits the
                # dead connection the watchdog re-surfaces the task once Redis is back.
                self._logger.warning(
                    "Dispatch loop lost Redis for %s (%s); backing off", task_id, exc
                )
                self._safe_requeue(task_id)
                time.sleep(_ERROR_BACKOFF_SEC)
            except Exception as exc:
                self._logger.exception("Dispatch loop error for %s: %s", task_id, exc)
                self._safe_requeue(task_id)
                time.sleep(_ERROR_BACKOFF_SEC)

    def _safe_requeue(self, task_id: str) -> None:
        """Requeue a task without letting a Redis outage kill the dispatch loop.

        The task is back in the in-memory ready queue before its return persists, so a
        persist that fails mid-outage leaves it queued; its durable state catches up at
        its next transition.
        """
        try:
            self.requeue_task(task_id, reason="dispatch_exception", front=True)
        except REDIS_CONN_ERRORS as exc:
            self._logger.warning(
                "Requeue persist for %s failed (Redis down: %s)", task_id, exc
            )

    def _render_merged_children(
        self, task_id: str, record: TaskRecord, parent_spec: TaskSpecStrict
    ) -> list[MergedChildTaskStrict] | None:
        """Render the children merged into a dispatch.

        A child that cannot run in this dispatch leaves the merge rather than failing
        it: one that is not ready yet returns to the queue still mergeable, one whose
        rendered merge key differs from the parent's returns to merge under its rendered
        key, and one whose own input is at fault or whose condition is not met returns
        to run alone and settle its own outcome.
        """
        rendered: list[MergedChildTaskStrict] = []
        for child_id in list(record.merged_children or []):
            child_record = self._runtime.merged_child_record(task_id, child_id)
            if child_record is None:
                self._runtime.release_merged_child(task_id, child_id, None)
                continue
            try:
                resolved, child_upstream = self._resolve_stage_references(
                    child_id, child_record.task, child_record
                )
                resolved.spec.validate_dispatchable()
                if (condition := resolved.spec.condition) is not None and str(
                    self._condition_actual(child_record, condition)
                ) != condition.equals:
                    # The child's own dispatch settles its skip.
                    self._runtime.release_merged_child(task_id, child_id, None)
                    continue
                key = resolved.spec.merge_key(scope=child_record.org_id)
                if key is None or key != parent_spec.merge_key(scope=record.org_id):
                    self._logger.info(
                        "Merged child %s of %s renders a different merge key; it "
                        "leaves the merge",
                        child_id,
                        task_id,
                    )
                    self._runtime.release_merged_child(task_id, child_id, key)
                    continue
                # Rendering runs off the runtime lock; the child may have left since.
                if self._runtime.merged_child_record(task_id, child_id) is None:
                    continue
                rendered.append(
                    MergedChildTaskStrict(
                        task_id=child_id,
                        owner_id=child_record.owner_id,
                        workflow_id=child_record.workflow_id,
                        spec=resolved.spec,
                        metadata=resolved.metadata,
                        upstream_results=child_upstream,
                    )
                )
            except (StageReferenceNotReady, ResultUnavailable) as exc:
                self._logger.debug(
                    "Merged child %s of %s is not ready yet: %s", child_id, task_id, exc
                )
                self._runtime.release_merged_child(
                    task_id, child_id, child_record.merge_key
                )
            except Exception as exc:
                self._logger.warning(
                    "Merged child %s of %s cannot be rendered; it runs alone: %s",
                    child_id,
                    task_id,
                    exc,
                )
                self._runtime.release_merged_child(task_id, child_id, None)
        return rendered or None

    def requeue_task(
        self,
        task_id: str,
        *,
        reason: str,
        front: bool = False,
        holder: str | None = None,
        count_retry: bool = True,
        extra_payload: dict[str, Any] | None = None,
    ) -> DispatchEnd:
        """Return a task to the ready queue, spending an attempt when ``count_retry``.

        Only a task ``holder`` holds is returned, and a settled one stays as it is. A
        task being cancelled settles CANCELLED, and one whose last attempt this spends
        fails. Returns what the return did to the task.
        """
        end = self._runtime.return_dispatch(
            task_id, holder, increment_retry=count_retry, front=front
        )
        if end is DispatchEnd.EXHAUSTED:
            record = self._runtime.get_record(task_id)
            assert record is not None
            self.fail_task(
                task_id,
                record.last_error or "max_attempts_exceeded",
                payload={
                    "reason": "max_attempts_exceeded",
                    "requeue_reason": reason,
                    "attempts": record.attempts,
                    "max_attempts": record.max_attempts,
                },
                worker_id=record.last_failed_worker or record.assigned_worker,
            )
            return DispatchEnd.FAILED
        if end is DispatchEnd.RETURNED and count_retry:
            payload = {"reason": reason}
            if extra_payload:
                payload.update(extra_payload)
            self._emit_task_event("TASK_REQUEUED", task_id, payload=payload)
        return end

    def fail_task(
        self,
        task_id: str,
        error_message: str,
        *,
        worker_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        failure_payload = payload.copy() if isinstance(payload, dict) else {}
        if error_message and "error" not in failure_payload:
            failure_payload["error"] = error_message
        impacted, _ = self._runtime.mark_failed(
            task_id,
            worker_id,
            failure_payload,
            now_iso(),
            error=error_message,
        )
        self._emit_task_event(
            "TASK_FAILED",
            task_id,
            worker_id=worker_id,
            payload=failure_payload,
            error=error_message,
        )
        for dep_id, reason in impacted:
            dependent_payload = {"dependency_failure": task_id, "error": reason}
            self._emit_task_event(
                "TASK_FAILED",
                dep_id,
                payload=dependent_payload,
                error=reason,
            )

    def _emit_task_event(
        self,
        event_type: str,
        task_id: str,
        *,
        worker_id: str | None = None,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not self._metrics:
            return
        event = TaskEvent(
            type=event_type,
            task_id=task_id,
            worker_id=worker_id,
            payload=payload or {},
            error=error,
            ts=now_iso(),
        )
        self._metrics.record_task_event(event)
        if event_type == "TASK_FAILED":
            self._metrics.finalize_task_failure(task_id)

    def _cached_worker_candidates(
        self,
        pool: list[Worker],
        model_names: list[str],
        dataset_names: list[str],
    ) -> list[Worker]:
        if not pool or (not model_names and not dataset_names):
            return []

        def _score(worker: Worker) -> int:
            score = 0
            cached_models = {
                m.lower() for m in (worker.cached_models or []) if isinstance(m, str)
            }
            cached_datasets = {
                d.lower() for d in (worker.cached_datasets or []) if isinstance(d, str)
            }
            if model_names:
                score += sum(1 for name in model_names if name.lower() in cached_models)
            if dataset_names:
                score += sum(
                    1 for name in dataset_names if name.lower() in cached_datasets
                )
            return score

        scored: list[tuple[Worker, int]] = []
        ttl = self._cache_ttl_sec
        now_ts = time.time()
        for worker in pool:
            if ttl:
                ts_raw = worker.cache_updated_ts
                if not ts_raw:
                    continue
                try:
                    parsed = datetime.datetime.fromisoformat(
                        ts_raw.rstrip("Z").replace("Z", "+00:00")
                    )
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=datetime.UTC)
                except Exception:
                    continue
                age = now_ts - parsed.timestamp()
                if age > ttl:
                    continue
            value = _score(worker)
            if value > 0:
                scored.append((worker, value))

        if not scored:
            return []

        max_score = max(score for _, score in scored)
        return [worker for worker, score in scored if score == max_score]

    def _preferred_stage_worker(
        self, record: TaskRecord, task: TaskEnvelope
    ) -> str | None:
        if not record or not record.graph_node_name:
            return None
        task_type = (record.task_type or "").strip().lower()
        if task_type.startswith("lora"):
            return None
        try:
            info = self._runtime.describe_task(record.task_id)
            depends_on = info.depends_on if info else None
        except Exception:
            depends_on = None
        if not depends_on:
            return None
        for dep_id in depends_on:
            dep_record = self._runtime.get_record(dep_id)
            if not dep_record or not dep_record.assigned_worker:
                continue
            stage_refs = [dep_record.graph_node_name, dep_id]
            for ref in stage_refs:
                if not ref:
                    continue
                if self._spec_has_weight_reference(task, ref):
                    return dep_record.assigned_worker
        return None

    def _spec_has_weight_reference(
        self, task: TaskEnvelope, stage_identifier: str
    ) -> bool:
        token = f"${{{stage_identifier}.result"
        return self._search_weight_reference(task, token, tuple())

    def _search_weight_reference(
        self, value: Any, token: str, path: tuple[str, ...]
    ) -> bool:
        if isinstance(value, BaseModel):
            for key, sub in value:
                next_path = path + (str(key).lower(),)
                if self._search_weight_reference(sub, token, next_path):
                    return True
        elif isinstance(value, dict):
            for key, sub in value.items():
                next_path = path + (str(key).lower(),)
                if self._search_weight_reference(sub, token, next_path):
                    return True
        elif isinstance(value, list):
            for item in value:
                if self._search_weight_reference(item, token, path):
                    return True
        elif isinstance(value, str):
            if token in value:
                lowered = value.lower()
                if any(hint in path for hint in self._weight_reference_hints) or any(
                    hint in lowered for hint in self._weight_reference_hints
                ):
                    return True
        return False

    def _should_wait_for_sticky_worker(self, record, worker_id: str) -> bool:
        if not record or not worker_id:
            return False
        task_type = (record.task_type or "").strip().lower()
        if not task_type:
            return False
        if task_type.startswith("lora"):
            return False
        category = (record.category or "").strip().lower()
        if category != "training":
            return False
        try:
            worker = self._worker_registry.get_worker(worker_id)
        except Exception:
            worker = None
        if not worker:
            return False
        try:
            if self._worker_registry.is_worker_stale(worker_id):
                return False
        except Exception:
            return False
        if worker.status is WorkerStatus.IDLE:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Stage reference handling
    # ------------------------------------------------------------------ #

    def _resolve_stage_references(
        self, task_id: str, task: TaskEnvelopeTemplate, record: TaskRecord
    ) -> tuple[TaskEnvelopeStrict, dict[str, ResultBinding] | None]:
        """Render a task's placeholders and name each upstream result it receives.

        Placeholders render here, against the upstream values they name; the upstream
        results themselves travel as bindings the worker hydrates.
        """
        context = self._build_stage_context(record)
        resolved_task: TaskEnvelopeTemplate = task
        if context and task.has_placeholder():
            resolved_task = self._resolve_placeholders(task, context)
        upstream = self._upstream_bindings(context, task_id) if context else {}
        return TaskEnvelopeStrict.model_validate(resolved_task), upstream or None

    def _resolve_placeholders(self, value: Any, context: dict[str, TaskRecord]) -> Any:
        if isinstance(value, str):
            exact = PLACEHOLDER_PATTERN.fullmatch(value)
            if exact:
                return self._resolve_reference(exact.group(1), context)
            return PLACEHOLDER_PATTERN.sub(
                lambda m: str(self._resolve_reference(m.group(1), context)),
                value,
            )
        if isinstance(value, dict):
            return {k: self._resolve_placeholders(v, context) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_placeholders(item, context) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_placeholders(item, context) for item in value)
        if isinstance(value, BaseModel):
            updates: dict[str, Any] = {}
            for key, current in value:
                transformed = self._resolve_placeholders(current, context)
                if transformed is not current:
                    updates[key] = transformed
            return value.model_copy(update=updates) if updates else value
        return value

    def _contains_placeholder(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(PLACEHOLDER_PATTERN.search(value))
        if isinstance(value, dict):
            return any(self._contains_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(self._contains_placeholder(item) for item in value)
        return False

    def _build_stage_context(self, record: TaskRecord) -> dict[str, TaskRecord]:
        """Collect upstream dependency records keyed by stage identity."""
        if not self._needs_stage_context(record):
            return {}

        context: dict[str, TaskRecord] = {}
        for dep_id in self._dependency_task_ids(record.task_id):
            other = self._runtime.get_record(dep_id)
            if other is None:
                continue
            for key in self._stage_context_keys(other):
                context[key] = other
        return context

    def _needs_stage_context(self, record: TaskRecord) -> bool:
        if record.graph_node_name is not None or record.local_name is not None:
            return True
        if record.task.has_placeholder():
            return True
        spec = record.task.spec
        return isinstance(spec, (SSHSpecStrict, SSHSpecTemplate)) and bool(spec.inputs)

    def _dependency_task_ids(self, task_id: str) -> set[str]:
        pending = self._task_dependencies(task_id)
        visited: set[str] = set()
        # Walk the upstream dependency graph to collect all dependency tasks
        while pending:
            dep_id = pending.pop()
            if dep_id in visited:
                continue
            visited.add(dep_id)
            pending.extend(
                upstream_id
                for upstream_id in self._task_dependencies(dep_id)
                if upstream_id not in visited
            )
        return visited

    def _task_dependencies(self, task_id: str) -> list[str]:
        info = self._runtime.describe_task(task_id)
        return [] if info is None else info.depends_on.copy()

    @staticmethod
    def _stage_context_keys(record: TaskRecord) -> tuple[str, ...]:
        keys: list[str] = []
        if record.local_name:
            keys.append(record.local_name)
        if record.graph_node_name and record.graph_node_name not in keys:
            keys.append(record.graph_node_name)
        return tuple(keys)

    def _resolve_reference(self, expr: str, context: dict[str, TaskRecord]) -> Any:
        expr = expr.strip()
        if not expr:
            raise ValueError("Empty stage reference")
        if "." not in expr:
            raise ValueError(f"Invalid stage reference '{expr}'")
        stage_name, path = expr.split(".", 1)
        stage_name = stage_name.strip()
        if not stage_name:
            raise ValueError(f"Invalid stage reference '{expr}'")
        stage_record = context.get(stage_name)
        if not stage_record:
            raise ValueError(f"Unknown stage reference '{stage_name}'")
        if path == "task_id":
            return stage_record.task_id
        if stage_record.status != "DONE":
            raise StageReferenceNotReady(f"Stage '{stage_name}' has not completed")
        envelope = self._load_stage_result(stage_record.task_id)
        value = self._dig_result_path(envelope.result, path.split("."))
        if value is None:
            raise ValueError(f"Missing value for reference '{expr}'")
        # If the referenced value is an artifact ref (an ``ArtifactRef`` or a
        # legacy ``{path: ...}`` dict), render it as a full URL (when base_url
        # is set) or an absolute filesystem path using the producing stage's
        # top-level _artifacts context.
        if rendered := self._render_artifact_ref(value, envelope):
            return rendered
        return value

    @staticmethod
    def _render_artifact_ref(value: Any, stage_result: ResultEnvelope) -> str | None:
        if isinstance(value, ArtifactRef):
            path_value: str | None = value.path
        elif isinstance(value, dict):
            path_value = value.get("path")
        else:
            return None
        if not isinstance(path_value, str) or not path_value:
            return None
        ctx = stage_result.result.artifacts_
        if ctx is None:
            return None
        base_url = ctx.base_url
        base_dir = ctx.base_dir
        if base_url and base_dir:
            task_id = Path(base_dir).name
            return f"{base_url.rstrip('/')}/api/v1/results/{task_id}/files/{path_value}"
        if base_dir:
            return (Path(base_dir) / "artifacts" / path_value).as_posix()
        return None

    def _upstream_bindings(
        self, context: dict[str, TaskRecord], current_task_id: str
    ) -> dict[str, ResultBinding]:
        bindings: dict[str, ResultBinding] = {}
        for name, record in context.items():
            if not record or record.task_id == current_task_id:
                continue
            if record.status != TaskStatus.DONE:
                continue
            binding = self._runtime.result_binding(record.task_id)
            if binding is None:
                self._logger.warning(
                    "Task %s receives no upstream result for %s: task %s has no bound "
                    "result",
                    current_task_id,
                    name,
                    record.task_id,
                )
                continue
            bindings[name] = binding
        return bindings

    def _validate_ssh_inputs(self, record: TaskRecord, spec: TaskSpecStrict) -> None:
        """Check that each SSH input names a settled upstream stage of the task."""
        if not isinstance(spec, SSHSpecStrict) or not spec.inputs:
            return
        context = self._build_stage_context(record)
        for entry in spec.inputs:
            stage_name = entry.stage.strip()
            if not stage_name:
                raise ValueError("SSH input stage names must be non-empty")
            upstream = context.get(stage_name)
            if upstream is None:
                raise ValueError(
                    f"Unknown SSH input stage '{stage_name}' for task {record.task_id}"
                )
            if upstream.task_id == record.task_id:
                raise ValueError(
                    f"SSH input stage '{stage_name}' cannot reference the current task"
                )
            if upstream.status != TaskStatus.DONE:
                raise StageReferenceNotReady(
                    f"Stage '{stage_name}' has not completed for SSH input mount"
                )

    def _load_stage_result(self, stage_task_id: str) -> ResultEnvelope:
        envelope = self._runtime.read_result(stage_task_id)
        if envelope is None:
            raise StageResultMissing(f"task {stage_task_id} has no bound result")
        return envelope

    def _dig_result_path(self, result: BaseExecutorResult, parts: list[str]) -> Any:
        current: Any = result
        for part in parts:
            part = part.strip()
            if part == "":
                continue
            if isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
                continue
            if isinstance(current, list):
                try:
                    idx = int(part)
                except ValueError as exc:
                    raise ValueError(
                        f"List index must be integer in reference path, got '{part}'"
                    ) from exc
                if idx < 0 or idx >= len(current):
                    return None
                current = current[idx]
                continue
            if isinstance(current, BaseModel):
                current = getattr(current, part, _SENTINEL)
                if current is _SENTINEL:
                    return None
                continue
            return None
        return current

    def _condition_actual(self, record: TaskRecord, condition: ConditionSpec) -> Any:
        """The upstream value a task's condition compares against."""
        stage_context = self._build_stage_context(record)
        upstream_record = stage_context.get(condition.node)
        if upstream_record is None:
            raise ValueError(
                f"Condition references unknown node '{condition.node}'; "
                f"known nodes: {list(stage_context.keys())}"
            )
        if upstream_record.status != TaskStatus.DONE:
            raise StageReferenceNotReady(
                f"Condition upstream node '{condition.node}' not yet DONE "
                f"(status={upstream_record.status})"
            )
        upstream_result = self._load_stage_result(upstream_record.task_id)
        return self._dig_result_path(upstream_result.result, condition.field.split("."))

    def _evaluate_condition_skip(
        self,
        task_id: str,
        rendered_task: TaskEnvelopeStrict,
        record: TaskRecord,
    ) -> bool:
        """Evaluate a task's condition and skip it if the condition is not met.

        Returns ``True`` if the task was skipped (caller should return early),
        ``False`` if the task should proceed with normal dispatch.
        """
        condition: ConditionSpec | None = rendered_task.spec.condition
        if condition is None:
            return False

        try:
            actual_value = self._condition_actual(record, condition)
            if str(actual_value) == condition.equals:
                return False  # Condition met — proceed with dispatch

            self._logger.info(
                "Task %s skipped: condition %s.%s == %r not met (actual=%r)",
                task_id,
                condition.node,
                condition.field,
                condition.equals,
                actual_value,
            )
            self._runtime.release_merge(task_id)
            ts = now_iso()
            self._runtime.mark_succeeded(
                task_id,
                worker_id=None,
                payload={"finished_at": ts, "started_at": ts},
                ts=ts,
                skip={
                    "skipped": True,
                    "reason": "condition_not_met",
                    "condition_node": condition.node,
                    "condition_field": condition.field,
                    "condition_expected": condition.equals,
                    "condition_actual": str(actual_value),
                },
            )
            return True
        except StageReferenceNotReady as exc:
            self._logger.debug(
                "Task %s condition: upstream %s not ready: %s",
                task_id,
                condition.node,
                exc,
            )
            self.requeue_task(
                task_id, reason="condition_upstream_pending", count_retry=False
            )
            return True
        except ResultUnavailable as exc:
            self._logger.warning(
                "Task %s condition: upstream %s result not reachable yet: %s",
                task_id,
                condition.node,
                exc,
            )
            self.requeue_task(
                task_id, reason="condition_upstream_unavailable", count_retry=False
            )
            return True
        except Exception as exc:
            self._logger.error(
                "Failed to evaluate condition for task %s: %s", task_id, exc
            )
            self._runtime.release_merge(task_id)
            self.fail_task(
                task_id,
                f"condition_evaluation_failed: {exc}",
                payload={"error": str(exc)},
            )
            return True
