"""Caller-neutral substrate shared by the run-to-yield episode executors."""

from typing import TYPE_CHECKING

from shared.content import ContentStoreError, FabricObjectStore
from shared.harness import DeliveredOutcome, HarnessResult
from shared.private_state import PrivateStateSealReport
from shared.schemas.result import BaseExecutorResult
from shared.tools.facade import FacadeTurnGroup

from .base_executor import ExecutionError

if TYPE_CHECKING:
    from ..lifecycle import Lifecycle


class EpisodeStepResult(BaseExecutorResult):
    """One run-to-yield episode step's result: the step plus its terminal value.

    ``harness_result`` carries the step back to the server through the success metadata;
    ``value`` is the episode's declared output on a completion step, readable over REST.
    ``facade_group`` is a turn group a worker facade captured on this step, carried with
    the completion so control routes it ordered-with the turn. ``private_state`` reports
    the generation this holder sealed at the step's quiescence fence.
    """

    harness_result: HarnessResult
    value: str | None = None
    facade_group: FacadeTurnGroup | None = None
    private_state: PrivateStateSealReport | None = None


def hydrate_delivered_outcomes(
    lifecycle: "Lifecycle | None",
    task_id: str,
    outcomes: tuple[DeliveredOutcome, ...],
) -> tuple[DeliveredOutcome, ...]:
    """Resolve any reference-backed outcome into its injected value.

    A manifest is fetched from the content store and digest-verified before injection;
    a hydration failure fails the step for a physical retry of the same reference, never
    a re-run of the invocation, so an unverified value is never injected. An inline
    outcome passes through.
    """
    if not any(o.outcome_ref is not None for o in outcomes):
        return outcomes
    plane = lifecycle.content_plane if lifecycle is not None else None
    if plane is None:
        raise ExecutionError("cannot hydrate a reference-backed outcome: no store")
    store = plane.for_task(task_id)
    return tuple(_hydrate(o, store) for o in outcomes)


def _hydrate(outcome: DeliveredOutcome, store: FabricObjectStore) -> DeliveredOutcome:
    if outcome.outcome_ref is None:
        return outcome
    try:
        value = store.hydrate(outcome.outcome_ref.content).decode()
    except (ContentStoreError, UnicodeDecodeError) as exc:
        raise ExecutionError(
            f"outcome hydration failed at {outcome.call_correlation}: {exc}"
        ) from exc
    return outcome.model_copy(update={"value": value, "outcome_ref": None})
