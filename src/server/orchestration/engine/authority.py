"""Authority decisions and delegated grants of one workflow instance."""

from shared.utils import new_authority_grant_id

from ...task.v2.representations.operators import AgentOperator, SpawnRegion
from ..outcomes import attenuate
from ..state import (
    AuthorityDecision,
    AuthorityDecisionKind,
    AuthorityGrant,
    DelegatedAuthorityGrant,
    DenialKind,
    WorkItem,
)
from ..tool_dispatch import GrantSnapshot
from .ledger import OrchestrationLedger
from .topology import PlanTopology


class AuthorityLedger:
    """Holds the instance's authority decisions, delegated grants and denied spawn
    sites, mints delegated grants, and resolves the faces a scope may invoke or
    delegate."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self.decisions: list[AuthorityDecision] = []
        self.grants: dict[str, DelegatedAuthorityGrant] = {}
        self.denied_spawns: set[str] = set()

    def record_decision(self, decision: AuthorityDecision) -> None:
        self.decisions.append(decision)

    def store_grant(self, grant: DelegatedAuthorityGrant) -> None:
        self.grants[grant.grant_id] = grant

    def grant_snapshot_for(self, wi: WorkItem) -> GrantSnapshot:
        act = self._ledger.activations.get(wi.activation_id)
        grant = (
            self.grant_for_scope(act.scope_id)
            if act is not None
            else self._ledger.root_grant
        )
        return GrantSnapshot(
            grant_id=grant.grant_id,
            policy_envelope=self._ledger.workflow_instance.policy_envelope,
        )

    def agent_faces(
        self, op: AgentOperator, wi: WorkItem
    ) -> tuple[frozenset[str], frozenset[str]]:
        """The agent's effective invoke/delegate faces: its ceiling under policy."""
        invoke, delegate = self.agent_face_tuples(
            op, self._ledger.activations[wi.activation_id].scope_id
        )
        return frozenset(invoke), frozenset(delegate)

    def agent_face_tuples(
        self, op: AgentOperator, agent_scope_id: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The agent's invoke/delegate faces: the scope grant under ceiling+policy."""
        scope_grant = self.grant_for_scope(agent_scope_id)
        envelope = self._policy_interfaces()
        invoke = attenuate(scope_grant.invoke, op.authority.invoke, envelope)
        delegate = attenuate(invoke, op.authority.delegate, envelope)
        return invoke, delegate

    def deny_spawn(
        self, spawn_op: str, interface: str, *, kind: DenialKind = DenialKind.AUTHORITY
    ) -> None:
        """Record a definitive dynamic authorization denial at a spawn site."""
        self.denied_spawns.add(spawn_op)
        scope_id = self._ledger.scope_id_for(spawn_op)
        self.decisions.append(
            AuthorityDecision(
                grant_id=self.grant_for_scope(scope_id or "").grant_id,
                interface=interface,
                kind=AuthorityDecisionKind.DENIED,
                operator_id=spawn_op,
                scope_id=scope_id,
                denial_kind=kind,
                reason=f"interface {interface!r} outside spawn-site {kind.value} face",
            )
        )
        self._ledger.emit(
            "policy_denied" if kind is DenialKind.POLICY else "authority_denied",
            operator_id=spawn_op,
        )

    def can_delegate(self, region_op: str, interface: str) -> bool:
        """Whether a child of ``region_op`` may itself delegate ``interface``."""
        scope_id = self._ledger.scope_id_for(region_op)
        return (
            scope_id is not None
            and interface in self.grant_for_scope(scope_id).delegate
        )

    def mint_delegated_grant(
        self,
        opener_op: str,
        parent_scope_id: str,
        *,
        parent_delegate: tuple[str, ...] | None = None,
    ) -> DelegatedAuthorityGrant:
        parent = self.grant_for_scope(parent_scope_id)
        # An agent-selected region attenuates from the agent's delegate face, supplied
        # here, rather than the enclosing scope's raw delegate face.
        base = parent.delegate if parent_delegate is None else parent_delegate
        opener = self._topology.operators[opener_op]
        ceiling = (
            opener.authority
            if isinstance(opener, (SpawnRegion, AgentOperator))
            else None
        )
        ceiling_invoke = ceiling.invoke if ceiling else base
        ceiling_delegate = ceiling.delegate if ceiling else base
        envelope = self._policy_interfaces()
        invoke = attenuate(base, ceiling_invoke, envelope)
        delegate = attenuate(invoke, ceiling_delegate, envelope)
        grant = DelegatedAuthorityGrant(
            grant_id=new_authority_grant_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            scope_id="",
            parent_grant_id=parent.grant_id,
            policy_id=parent.policy_id,
            invoke=invoke,
            delegate=delegate,
            epoch=parent.epoch + 1,
        )
        self._ledger.emit(
            "grant_delegated",
            operator_id=opener_op,
            detail={"invoke": ",".join(invoke), "delegate": ",".join(delegate)},
        )
        return grant

    def effective_invoke_face(self, task_id: str) -> tuple[str, ...]:
        """The interfaces this agent activation may invoke."""
        wi = self._ledger.work_item_for_task(task_id)
        act = self._ledger.activations.get(wi.activation_id) if wi is not None else None
        op = self._topology.operators.get(wi.operator_id) if wi is not None else None
        if act is None or not isinstance(op, AgentOperator):
            return ()
        invoke, _delegate = self.agent_face_tuples(op, act.scope_id)
        return invoke

    def grant_for_scope(
        self, scope_id: str
    ) -> AuthorityGrant | DelegatedAuthorityGrant:
        """The grant a scope runs under: its own, else its nearest enclosing scope's.

        A loop or a definition child's scope holds no grant of its own, so it runs
        under the grant of the scope it entered from; only the root falls back to the
        root grant.
        """
        current: str | None = scope_id
        while (scope := self._ledger.scopes.get(current or "")) is not None:
            if scope.grant_id and scope.grant_id in self.grants:
                return self.grants[scope.grant_id]
            current = scope.parent_scope_id
        return self._ledger.root_grant

    def _policy_interfaces(self) -> tuple[str, ...]:
        # The pinned policy envelope caps every face; the root grant projects it.
        return self._ledger.root_grant.delegate or self._ledger.root_grant.invoke

    def grant_for(self, region_op: str) -> DelegatedAuthorityGrant | None:
        scope_id = self._ledger.scope_id_for(region_op)
        if scope_id is None:
            return None
        grant_id = self._ledger.scopes[scope_id].grant_id
        return self.grants.get(grant_id) if grant_id else None
