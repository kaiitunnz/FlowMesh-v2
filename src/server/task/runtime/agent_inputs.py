"""Staging and settling an agent's declared inputs from upstream results."""

from dataclasses import dataclass

from shared.content import ContentReference
from shared.harness import InputBinding, InputBindingMember
from shared.schemas.result import ResultEnvelope
from shared.schemas.result.binding import binding_text, skip_envelope
from shared.tasks.result_binding import BindingKind, ResultBinding

from ...orchestration import (
    AcceptedInput,
    AcceptedInputMember,
    Advance,
    OrchestrationEngine,
    PublicationOutcome,
    ValueRef,
)
from ...orchestration.tool_dispatch import InputMemberPlan
from ..redrive import StoreRedriveScheduler
from ..results import ResultReader, ResultUnavailable, ResultUnreadable
from . import content_bindings
from .content_bindings import ContentBindings, UnreadableInput


@dataclass(frozen=True)
class _PortSnapshot:
    """One agent input port whose members are each frozen to the value it reads."""

    target_port: str
    provenance: str
    members: tuple[tuple[InputMemberPlan, ValueRef, ResultBinding | None], ...]


@dataclass(frozen=True)
class _AgentInputSnapshot:
    """An agent's pending input ports as of one look at the ledger."""

    task_id: str
    activation_id: str
    ports: tuple[_PortSnapshot, ...]
    # Why the agent's input can never be read, when a value has nothing to read.
    unreadable: str | None = None

    @property
    def references(self) -> tuple[ContentReference, ...]:
        """Each stored result the ports read, once."""
        return tuple(
            dict.fromkeys(
                reference
                for port in self.ports
                for _member, _value, binding in port.members
                if binding is not None
                for reference in content_bindings.references(binding)
            )
        )


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
    value_ref: ValueRef,
    binding: ResultBinding | None,
    values: dict[ContentReference, ResultEnvelope | Exception],
) -> str | None:
    """The string an input member resolves to from the results read for it."""
    if binding is None:
        return _literal_text(value_ref)

    def envelope_of(read: ResultBinding) -> ResultEnvelope:
        if read.reference is None:
            return skip_envelope(read)
        envelope = values.get(read.reference)
        if not isinstance(envelope, ResultEnvelope):
            raise IndexError(f"the stored result of task {read.task_id} is unread")
        return envelope

    try:
        return binding_text(binding, envelope_of)
    except IndexError:
        return None


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


def read_input_values(
    results: ResultReader, snapshots: list[_AgentInputSnapshot]
) -> dict[ContentReference, ResultEnvelope | Exception]:
    """Read every result the snapshots' inputs are frozen to, off the lock."""
    values: dict[ContentReference, ResultEnvelope | Exception] = {}
    for snapshot in snapshots:
        for reference in snapshot.references:
            if reference in values:
                continue
            try:
                values[reference] = results.read_reference(reference)
            except (ResultUnavailable, ResultUnreadable) as exc:
                values[reference] = exc
    return values


class AgentInputs:
    """Stages and settles each agent's declared inputs from upstream results,
    re-driving a read the store could not serve."""

    def __init__(
        self,
        content_bindings: ContentBindings,
        redrive: StoreRedriveScheduler,
        input_budget_bytes: int,
    ) -> None:
        self._content_bindings = content_bindings
        self._redrive = redrive
        self._input_budget_bytes = input_budget_bytes

    def stage_agent_inputs_locked(
        self, workflow_id: str, engine: OrchestrationEngine, advance: Advance
    ) -> None:
        """Record each edge-bound agent's accepted inputs that need no stored read.

        An agent whose bound inputs have to be read from the store is left for an
        immediate off-lock re-drive, which reads them and records them there.
        """
        drive = False
        for task_id in engine.blocked_input_agents():
            snapshot = self.agent_input_snapshot_locked(engine, task_id)
            if snapshot is None:
                continue
            if snapshot.references:
                drive = True
                continue
            self.settle_agent_inputs_locked(workflow_id, engine, snapshot, {}, advance)
        if drive:
            self._redrive.drive_now(workflow_id)

    def agent_input_snapshot_locked(
        self, engine: OrchestrationEngine, task_id: str
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
            members: list[tuple[InputMemberPlan, ValueRef, ResultBinding | None]] = []
            for member in port.members:
                value_ref = member.value_ref
                if value_ref.kind in ("inline", "empty"):
                    members.append((member, value_ref, None))
                    continue
                try:
                    binding = self._content_bindings.value_binding_locked(value_ref)
                except UnreadableInput as exc:
                    unreadable = str(exc)
                    break
                if binding.kind is BindingKind.RESULT and binding.reference is None:
                    if binding.skip is None:
                        producer = value_ref.legacy_task_id or ""
                        if self._content_bindings.settled_unbound_locked(producer):
                            unreadable = f"task {producer} settled with no bound result"
                        break
                elif value_ref.kind == "legacy_task_result":
                    value_ref = value_ref.model_copy(
                        update={"content": binding.reference}
                    )
                members.append((member, value_ref, binding))
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

    def settle_agent_inputs_locked(
        self,
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
        read = [values.get(ref) for ref in snapshot.references]
        if any(isinstance(value, ResultUnavailable) for value in read):
            self._redrive.schedule(workflow_id)
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
            texts = [
                _member_text(value_ref, binding, values)
                for _member, value_ref, binding in port.members
            ]
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
                        for member, value_ref, _binding in port.members
                    ),
                )
            )
        if total_bytes > self._input_budget_bytes:
            advance.extend(
                engine.on_failed(
                    task_id,
                    f"input_too_large: resolved input is {total_bytes} bytes, "
                    f"over the {self._input_budget_bytes}-byte budget",
                    retryable=False,
                )
            )
            return
        for entry in accepted:
            engine.record_accepted_input(entry)
        advance.extend(engine.reconsider_admission(task_id))

    def agent_input_bindings(
        self, engine: OrchestrationEngine, task_id: str
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
                    source=self._content_bindings.member_source_locked(
                        member.value_ref
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
