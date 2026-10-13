"""Worker reports and task records read for settlement: reported references,
membership, retry eligibility, outcomes and in-flight usage."""

from typing import Any

from pydantic import ValidationError

from shared.content import ContentReference
from shared.tasks.artifact_binding import ArtifactBinding
from shared.tasks.components.exports import EXPORT_UNSAFE, ExportFailure

from ..models import (
    SETTLING_TASK_STATUSES,
    DispatchEnd,
    EventEffect,
    SettleOutcome,
    TaskRecord,
    TaskStatus,
    TaskUsage,
)


def reported_reference(raw: Any) -> ContentReference | None:
    """A result reference a worker reported, or None when it reported none."""
    if raw is None:
        return None
    try:
        return ContentReference.model_validate(raw)
    except ValidationError:
        return None


def reported_artifacts(raw: Any) -> list[ArtifactBinding] | None:
    """The artifacts a worker reported publishing, or None for a report that does
    not parse."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None
    try:
        return [ArtifactBinding.model_validate(item) for item in raw]
    except ValidationError:
        return None


def export_failure(payload: dict[str, Any]) -> ExportFailure | None:
    """Why a completing agent could not copy out the exports it declared, if it
    could not; a report that does not parse reads as a refusal."""
    if (raw := payload.get("agent_export_failure")) is None:
        return None
    try:
        return ExportFailure.model_validate(raw)
    except ValidationError:
        return ExportFailure(kind=EXPORT_UNSAFE, message="an unreadable export report")


def reported_child_artifacts(
    payload: dict[str, Any],
) -> dict[str, list[ArtifactBinding] | None]:
    """The artifacts a merged dispatch reported for its children, by child."""
    raw = payload.get("child_artifacts") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(child_id): reported_artifacts(items) for child_id, items in raw.items()}


def reset_to_pending(record: TaskRecord) -> None:
    """Clear what a task's last dispatch left on it, returning it to PENDING."""
    record.status = TaskStatus.PENDING
    record.assigned_worker = None
    record.dispatch_id = None
    record.topic = None
    record.dispatched_ts = None
    record.started_ts = None
    record.finished_ts = None
    record.error = None


def membership(record: TaskRecord) -> str:
    """The status set a task's record commits into. A task being cancelled still runs
    on its worker. A child its region's residual policy cancelled settles its workflow
    as a finished task does, never as a cancelled one."""
    if record.status == TaskStatus.CANCELLING:
        return TaskStatus.DISPATCHED
    if record.status == TaskStatus.CANCELLED and record.residual_cancel:
        return TaskStatus.DONE
    return record.status


def failed_task_can_retry(record: TaskRecord, retryable: bool | None) -> bool:
    """Whether a failed task may be requeued: retryable, within the attempt budget,
    and not settling."""
    if record.status in SETTLING_TASK_STATUSES or retryable is False:
        return False
    return record.max_attempts < 0 or record.attempts < record.max_attempts


def settle_outcome(
    effect: EventEffect,
    record: TaskRecord | None,
    merged_children: list[str],
    usages: list[tuple[str, TaskUsage]],
    impacted: tuple[tuple[str, str], ...] = (),
) -> SettleOutcome:
    return SettleOutcome(
        effect,
        record.status if record is not None else None,
        usages if effect in (EventEffect.APPLIED, EventEffect.FAILED) else [],
        merged_children,
        impacted,
    )


LOSS_EFFECTS = {
    DispatchEnd.RETURNED: EventEffect.RETURNED,
    DispatchEnd.FAILED: EventEffect.FAILED,
    DispatchEnd.STALE: EventEffect.STALE,
}


def reported_child_references(payload: dict[str, Any]) -> dict[str, ContentReference]:
    """The result references a merged dispatch reported for its children, by child."""
    return {
        str(child_id): parsed
        for child_id, raw in (payload.get("child_result_references") or {}).items()
        if (parsed := reported_reference(raw)) is not None
    }


def in_flight_usage(
    task_id: str, payload: dict[str, Any]
) -> list[tuple[str, TaskUsage]]:
    """The usage row for a dispatch whose task runs on rather than settling."""
    usage = TaskUsage.from_payload(payload, TaskStatus.DISPATCHED)
    return [(task_id, usage)] if usage is not None else []
