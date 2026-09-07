"""Caller-neutral substrate shared by the run-to-yield episode executors."""

from shared.harness import DeliveredOutcome, HarnessResult
from shared.outcome import ContentStoreError, FabricContentStore
from shared.schemas.result import BaseExecutorResult
from shared.tools.facade import FacadeTurnGroup

from ..content_store import build_content_store
from .base_executor import ExecutionError


class EpisodeStepResult(BaseExecutorResult):
    """One run-to-yield episode step's result: the step plus its terminal value.

    ``harness_result`` carries the step back to the server through the success metadata;
    ``value`` is the episode's declared output on a completion step, readable over REST.
    ``facade_group`` is a turn group a worker facade captured on this step, carried with
    the completion so control routes it ordered-with the turn.
    """

    harness_result: HarnessResult
    value: str | None = None
    facade_group: FacadeTurnGroup | None = None


def hydrate_delivered_outcomes(
    server_base_url: str | None, outcomes: tuple[DeliveredOutcome, ...]
) -> tuple[DeliveredOutcome, ...]:
    """Resolve any reference-backed outcome into its injected value.

    A manifest is fetched from the content store and digest-verified before injection;
    a hydration failure fails the step for a physical retry of the same reference, never
    a re-run of the invocation, so an unverified value is never injected. An inline
    outcome passes through.
    """
    if not any(o.outcome_ref is not None for o in outcomes):
        return outcomes
    store = build_content_store(server_base_url)
    if store is None:
        raise ExecutionError("cannot hydrate a reference-backed outcome: no store")
    return tuple(_hydrate(o, store) for o in outcomes)


def _hydrate(outcome: DeliveredOutcome, store: FabricContentStore) -> DeliveredOutcome:
    if outcome.outcome_ref is None:
        return outcome
    try:
        value = store.hydrate(outcome.outcome_ref).decode()
    except (ContentStoreError, UnicodeDecodeError) as exc:
        raise ExecutionError(
            f"outcome hydration failed at {outcome.call_correlation}: {exc}"
        ) from exc
    return outcome.model_copy(update={"value": value, "outcome_ref": None})
