"""Staging and settling an agent's declared inputs from upstream results."""

from dataclasses import dataclass

from shared.content import ContentReference
from shared.harness import InputBinding, InputBindingMember
from shared.schemas.result import ResultEnvelope
from shared.schemas.result.binding import value_text

from ...orchestration import (
    AcceptedInput,
    AcceptedInputMember,
    Advance,
    OrchestrationEngine,
    PublicationOutcome,
    ValueRef,
)
from ...orchestration.tool_dispatch import InputMemberPlan
from ..models import TaskRecord
from ..redrive import StoreRedriveScheduler
from ..results import ResultReader, ResultUnavailable, ResultUnreadable
from . import content_bindings


@dataclass(frozen=True)
class _PortSnapshot:
    """One agent input port whose members are each frozen to what supplies them."""

    target_port: str
    provenance: str
    members: tuple[tuple[InputMemberPlan, ValueRef], ...]


@dataclass(frozen=True)
class _AgentInputSnapshot:
    """An agent's pending input ports as of one look at the ledger."""

    task_id: str
    activation_id: str
    ports: tuple[_PortSnapshot, ...]
    # Why the agent's input can never be read, when a producer settled unbound.
    unreadable: str | None = None

    @property
    def references(self) -> dict[str, ContentReference]:
        """Each stored result the ports read, keyed by the producer it names."""
        return {
            value_ref.legacy_task_id or "": value_ref.content
            for port in self.ports
            for _member, value_ref in port.members
            if value_ref.content is not None
        }


def _literal_text(value_ref: ValueRef | None) -> str | None:
    """The value an inline input member carries itself."""
    if value_ref is None:
        return None
    if value_ref.kind == "inline":
        return value_ref.literal or ""
    if value_ref.kind == "empty":
        return ""
    return None


def _member_text(
    value_ref: ValueRef, values: dict[ContentReference, ResultEnvelope | Exception]
) -> str | None:
    """The string an input member resolves to from the results read for it."""
    if value_ref.content is None:
        return _literal_text(value_ref)
    envelope = values.get(value_ref.content)
    if not isinstance(envelope, ResultEnvelope):
        return None
    return value_text(envelope, content_bindings.element_of(value_ref))


def mint_fanout_facet_locked(
    engine: OrchestrationEngine,
    child_task_id: str,
    producer_task_id: str,
    index: int,
    value_ref: ValueRef,
) -> None:
    """Record a fan-out child's typed entry-port input from its producer element."""
    wi = engine.work_item(child_task_id)
    if wi is None:
        return
    entry_port = engine.agent_entry_port(wi.operator_id)
    if entry_port is None:
        return
    engine.record_accepted_input(
        AcceptedInput(
            activation_id=wi.activation_id,
            target_port=entry_port,
            occurrence_key=str(index),
            provenance="spawn_element",
            members=(
                AcceptedInputMember(
                    source_operator_id=producer_task_id,
                    source_activation_id=wi.activation_id,
                    child_index=index,
                    outcome=PublicationOutcome.SUCCESS,
                    value_ref=value_ref,
                ),
            ),
        )
    )


def stage_agent_inputs_locked(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    redrive: StoreRedriveScheduler,
    input_budget_bytes: int,
    workflow_id: str,
    engine: OrchestrationEngine,
    advance: Advance,
) -> None:
    """Record each edge-bound agent's accepted inputs that need no stored read.

    An agent whose bound inputs have to be read from the store is left for an
    immediate off-lock re-drive, which reads them and records them there.
    """
    drive = False
    for task_id in engine.blocked_input_agents():
        snapshot = agent_input_snapshot_locked(tasks, engines, engine, task_id)
        if snapshot is None:
            continue
        if snapshot.references:
            drive = True
            continue
        settle_agent_inputs_locked(
            redrive, input_budget_bytes, workflow_id, engine, snapshot, {}, advance
        )
    if drive:
        redrive.drive_now(workflow_id)


def agent_input_snapshot_locked(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    engine: OrchestrationEngine,
    task_id: str,
) -> _AgentInputSnapshot | None:
    """What an agent's pending input ports resolve from, as the ledger stands.

    Each member of a port whose producers are all bound is frozen to the result its
    producer settled with. A port with a member whose producer has nothing bound
    waits for a later advance; a producer that settled with nothing bound makes the
    whole input unreadable.
    """
    plan = engine.agent_input_plan(task_id)
    if plan is None:
        return None
    ports: list[_PortSnapshot] = []
    unreadable: str | None = None
    for port in plan.ports:
        members: list[tuple[InputMemberPlan, ValueRef]] = []
        for member in port.members:
            value_ref = ValueRef(
                kind=member.value_ref_kind,
                legacy_task_id=member.legacy_task_id,
                collection_key=member.collection_key,
                literal=member.literal,
            )
            if value_ref.kind == "legacy_task_result":
                producer = value_ref.legacy_task_id or ""
                binding = content_bindings.result_binding_locked(
                    tasks, engines, producer
                )
                if binding is None or binding.reference is None:
                    if content_bindings.settled_unbound_locked(
                        tasks, engines, producer
                    ):
                        unreadable = f"task {producer} settled with no bound result"
                    break
                value_ref = value_ref.model_copy(update={"content": binding.reference})
            elif value_ref.kind not in ("inline", "empty"):
                break
            members.append((member, value_ref))
        if unreadable is not None:
            break
        if len(members) == len(port.members):
            ports.append(
                _PortSnapshot(
                    target_port=port.target_port,
                    provenance=port.provenance,
                    members=tuple(members),
                )
            )
    return _AgentInputSnapshot(
        task_id=task_id,
        activation_id=plan.activation_id,
        ports=tuple(ports),
        unreadable=unreadable,
    )


def read_input_values(
    results: ResultReader, snapshots: list[_AgentInputSnapshot]
) -> dict[ContentReference, ResultEnvelope | Exception]:
    """Read every result the snapshots' inputs are frozen to, off the lock."""
    values: dict[ContentReference, ResultEnvelope | Exception] = {}
    for snapshot in snapshots:
        for reference in snapshot.references.values():
            if reference in values:
                continue
            try:
                values[reference] = results.read_reference(reference)
            except (ResultUnavailable, ResultUnreadable) as exc:
                values[reference] = exc
    return values


def settle_agent_inputs_locked(
    redrive: StoreRedriveScheduler,
    input_budget_bytes: int,
    workflow_id: str,
    engine: OrchestrationEngine,
    snapshot: _AgentInputSnapshot,
    values: dict[ContentReference, ResultEnvelope | Exception],
    advance: Advance,
) -> None:
    """Record an agent's accepted inputs from their read values, then re-admit.

    The budget counts the bytes of the resolved member strings; inputs over it fail
    the agent as a declared failure.
    """
    task_id = snapshot.task_id
    if snapshot.unreadable is not None:
        advance.extend(
            engine.on_failed(
                task_id, f"input_unreadable: {snapshot.unreadable}", retryable=False
            )
        )
        return
    read = [values.get(ref) for ref in snapshot.references.values()]
    if any(isinstance(value, ResultUnavailable) for value in read):
        redrive.schedule(workflow_id)
        return
    if unreadable := next(
        (value for value in read if isinstance(value, ResultUnreadable)), None
    ):
        advance.extend(
            engine.on_failed(
                task_id, f"input_unreadable: {unreadable}", retryable=False
            )
        )
        return
    total_bytes = 0
    accepted: list[AcceptedInput] = []
    for port in snapshot.ports:
        texts = [_member_text(value_ref, values) for _member, value_ref in port.members]
        if any(text is None for text in texts):
            continue
        total_bytes += sum(len((text or "").encode("utf-8")) for text in texts)
        accepted.append(
            AcceptedInput(
                activation_id=snapshot.activation_id,
                target_port=port.target_port,
                provenance=port.provenance,
                members=tuple(
                    AcceptedInputMember(
                        source_operator_id=member.source_operator_id,
                        source_activation_id=member.source_activation_id,
                        child_index=member.child_index,
                        outcome=PublicationOutcome(member.outcome),
                        value_ref=value_ref,
                        ordinal=member.ordinal,
                    )
                    for member, value_ref in port.members
                ),
            )
        )
    if total_bytes > input_budget_bytes:
        advance.extend(
            engine.on_failed(
                task_id,
                f"input_too_large: resolved input is {total_bytes} bytes, "
                f"over the {input_budget_bytes}-byte budget",
                retryable=False,
            )
        )
        return
    for entry in accepted:
        engine.record_accepted_input(entry)
    advance.extend(engine.reconsider_admission(task_id))


def agent_input_bindings(
    tasks: dict[str, TaskRecord],
    engines: dict[str, OrchestrationEngine],
    engine: OrchestrationEngine,
    task_id: str,
) -> tuple[InputBinding, ...]:
    """The first-turn input bindings for an agent's input ports.

    A member a producer's result supplies names that result for the worker to
    hydrate; an inline member carries its own literal.
    """
    bindings: list[InputBinding] = []
    for ordinal, accepted in enumerate(engine.accepted_inputs_for_task(task_id)):
        members = tuple(
            InputBindingMember(
                source_operator_id=member.source_operator_id,
                source_activation_id=member.source_activation_id,
                child_index=member.child_index,
                outcome=member.outcome.value,
                value=_literal_text(member.value_ref),
                source=content_bindings.member_source_locked(
                    tasks, engines, member.value_ref
                ),
                ordinal=member.ordinal,
            )
            for member in accepted.members
        )
        bindings.append(
            InputBinding(
                port=accepted.target_port,
                provenance=accepted.provenance,
                ordinal=accepted.ordinal or ordinal,
                members=members,
            )
        )
    return tuple(bindings)
