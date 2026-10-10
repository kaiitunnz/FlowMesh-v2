"""Scopes and progress capabilities of one workflow instance."""

from shared.utils import new_scope_id

from ..guardrails import ScopeBudget
from ..state import (
    TERMINAL_WORK_ITEM_STATUSES,
    CapabilityStatus,
    ProgressAxis,
    ProgressCapability,
    Scope,
)
from .advance import RegionError
from .authority import AuthorityLedger
from .failures import FailureLedger
from .ledger import OrchestrationLedger


class ScopeProgress:
    """Opens scopes and their child-init and loop-time capabilities, charges and
    checks the scope budget, and closes a scope's frontier."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        failures: FailureLedger,
        authority: AuthorityLedger,
        budget: ScopeBudget,
    ) -> None:
        self._ledger = ledger
        self._failures = failures
        self._authority = authority
        self._budget = budget

    def revoke_progress(self, scope_id: str) -> None:
        """Revoke a scope's open child-init and loop-time capabilities."""
        scope = self._ledger.scopes[scope_id]
        for axis in (ProgressAxis.CHILD_INIT, ProgressAxis.LOOP_TIME):
            cap = self._ledger.capabilities.get((scope_id, axis))
            if cap is not None and cap.status is CapabilityStatus.OPEN:
                cap.status = CapabilityStatus.REVOKED
                self._ledger.emit(
                    (
                        "child_init_revoked"
                        if axis is ProgressAxis.CHILD_INIT
                        else "loop_revoked"
                    ),
                    operator_id=scope.owner_operator_id,
                    detail={"scope": scope_id},
                )

    def open_child_init_scope(
        self,
        opener_activation: str,
        *,
        parent_scope_id: str | None = None,
        parent_delegate: tuple[str, ...] | None = None,
    ) -> str:
        if opener_activation in self._ledger.scope_by_activation:
            return self._ledger.scope_by_activation[opener_activation]
        scope = self._new_child_scope(
            opener_activation, parent_scope_id, parent_delegate=parent_delegate
        )
        self._register_scope_owner(scope)
        self._acquire_capability(scope.scope_id, ProgressAxis.CHILD_INIT)
        self._ledger.emit(
            "child_init_acquired",
            operator_id=scope.owner_operator_id,
            detail={"scope": scope.scope_id},
        )
        return scope.scope_id

    def open_loop(self, opener_activation: str, parent_scope_id: str) -> str:
        """Open a loop's progress scope under its enclosing scope.

        Entering a loop moves progress, never authority: the scope carries no grant of
        its own and runs under its enclosing scope's.
        """
        if opener_activation in self._ledger.scope_by_activation:
            return self._ledger.scope_by_activation[opener_activation]
        parent = self._ledger.scopes[parent_scope_id]
        scope = Scope(
            scope_id=new_scope_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            parent_scope_id=parent.scope_id,
            owner_operator_id=self._ledger.activations[opener_activation].operator_id,
            owner_activation_id=opener_activation,
            depth=parent.depth,
        )
        self._ledger.add_scope(scope)
        self._register_scope_owner(scope)
        self._acquire_capability(scope.scope_id, ProgressAxis.LOOP_TIME, coordinate=0)
        self._ledger.emit(
            "loop_ingress",
            operator_id=scope.owner_operator_id,
            detail={"scope": scope.scope_id},
        )
        return scope.scope_id

    def open_context_scope(self, child_activation: str, parent_scope_id: str) -> str:
        """Open the progress scope of a child that runs a region definition.

        It inherits the grant of the child-init scope that created the child.
        """
        parent = self._ledger.scopes[parent_scope_id]
        scope = Scope(
            scope_id=new_scope_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            parent_scope_id=parent.scope_id,
            owner_activation_id=child_activation,
            depth=parent.depth,
        )
        self._ledger.add_scope(scope)
        self._register_scope_owner(scope)
        return scope.scope_id

    def _acquire_capability(
        self, scope_id: str, axis: ProgressAxis, *, coordinate: int | None = None
    ) -> ProgressCapability:
        cap = ProgressCapability(scope_id=scope_id, axis=axis, coordinate=coordinate)
        self._ledger.capabilities[(scope_id, axis)] = cap
        return cap

    def require_capability(
        self, scope_id: str, axis: ProgressAxis
    ) -> ProgressCapability:
        cap = self._ledger.capabilities.get((scope_id, axis))
        if cap is None:
            raise RegionError(f"scope {scope_id!r} holds no {axis.value} capability")
        return cap

    def check_scope_depth(self, parent_scope_id: str) -> None:
        if (
            self._ledger.scopes[parent_scope_id].depth + 1
            > self._budget.max_scope_depth
        ):
            self.exhaust_budget("scope_depth", self._budget.max_scope_depth)

    def _new_child_scope(
        self,
        opener_activation: str,
        parent_scope_id: str | None,
        *,
        parent_delegate: tuple[str, ...] | None = None,
    ) -> Scope:
        opener_op = self._ledger.activations[opener_activation].operator_id
        parent = self._ledger.scopes[
            parent_scope_id or self._ledger.root_scope.scope_id
        ]
        self.check_scope_depth(parent.scope_id)
        grant = self._authority.mint_delegated_grant(
            opener_op, parent.scope_id, parent_delegate=parent_delegate
        )
        scope = Scope(
            scope_id=new_scope_id(),
            instance_id=self._ledger.workflow_instance.instance_id,
            parent_scope_id=parent.scope_id,
            owner_operator_id=opener_op,
            owner_activation_id=opener_activation,
            grant_id=grant.grant_id,
            depth=parent.depth + 1,
        )
        self._ledger.add_scope(scope)
        self._authority.store_grant(
            grant.model_copy(update={"scope_id": scope.scope_id})
        )
        return scope

    def _register_scope_owner(self, scope: Scope) -> None:
        if scope.owner_activation_id:
            self._ledger.scope_by_activation[scope.owner_activation_id] = scope.scope_id
        if scope.owner_operator_id and scope.owner_activation_id:
            self._ledger.owner_acts_by_operator.setdefault(
                scope.owner_operator_id, []
            ).append(scope.owner_activation_id)

    def scope_drained(self, scope_id: str) -> bool:
        """Whether nothing can still arrive in a scope: every occurrence and child it
        accounts for is terminal, no spawn of it can add a child, and every scope
        nested under it has drained too."""
        ledger = self._ledger
        occurrences = ledger.open_occurrences.get(scope_id, {})
        for key in list(occurrences):
            if (wi_id := ledger.wi_by_occurrence.get(key)) is not None:
                if ledger.work_items[wi_id].status not in TERMINAL_WORK_ITEM_STATUSES:
                    return False
            elif not ledger.control_terminal(key):
                return False
            del occurrences[key]
        children = ledger.open_children.get(scope_id, {})
        for child in list(children):
            if (wi_id := ledger.wi_by_activation.get(child)) is None:
                continue
            if ledger.work_items[wi_id].status not in TERMINAL_WORK_ITEM_STATUSES:
                return False
            del children[child]
        cap = ledger.capabilities.get((scope_id, ProgressAxis.CHILD_INIT))
        if cap is not None and cap.status is CapabilityStatus.OPEN:
            return False
        subscopes = ledger.open_subscopes.get(scope_id, {})
        for sub in list(subscopes):
            if not self.scope_drained(sub):
                return False
            # A released scope admits nothing more, so once drained it stays drained.
            if sub in ledger.released_scopes:
                del subscopes[sub]
        return True

    def frontier_closed(self, scope_id: str) -> None:
        self._ledger.emit("frontier_closed", detail={"scope": scope_id})

    def charge_activation(self) -> None:
        self.charge_activations(1)

    def charge_activations(self, count: int) -> None:
        """Check a batch of new activations against the activation budget."""
        if self._ledger.dynamic_activations + count > self._budget.max_activations:
            self.exhaust_budget("activations", self._budget.max_activations)

    def exhaust_budget(self, budget: str, limit: int) -> None:
        """Record a durable scope-budget breach, distinct from an authority denial."""
        self._ledger.emit(
            "scope_budget_exhausted", detail={"budget": budget, "limit": str(limit)}
        )
        raise RegionError(f"{budget} budget {limit} exhausted")

    def capability(
        self, scope_id: str | None, axis: ProgressAxis
    ) -> ProgressCapability | None:
        return self._ledger.capabilities.get((scope_id, axis)) if scope_id else None

    def spawn_awaits_children(self, spawn_op: str) -> bool:
        """Whether a spawn has yet to fan out: unopened, or open and not sealed."""
        if self._failures.region_failed(spawn_op):
            return False
        scope_id = self._ledger.scope_id_for(spawn_op)
        if scope_id is None:
            return True
        cap = self.require_capability(scope_id, ProgressAxis.CHILD_INIT)
        return cap.status is CapabilityStatus.OPEN

    def spawn_is_open(self, spawn_op: str) -> bool:
        """Whether a spawn's child-init capability still admits new children."""
        scope_id = self._ledger.scope_id_for(spawn_op)
        if scope_id is None:
            return False
        cap = self.require_capability(scope_id, ProgressAxis.CHILD_INIT)
        return cap.status is CapabilityStatus.OPEN

    def require_child_init_scope(self, handle: str) -> str:
        if (scope_id := self._ledger.scope_id_for(handle)) is not None:
            return scope_id
        opener = self._ledger.resolve_opener_activation(handle)
        if opener is None:
            raise RegionError(f"{handle!r} has no opener activation")
        # A lazily opened scope (an agent's first spawn) nests under the opener's own
        # enclosing scope, so a nested agent's recursion depth is counted correctly.
        parent = (
            self._ledger.activations[opener].scope_id
            if opener in self._ledger.activations
            else None
        )
        return self.open_child_init_scope(opener, parent_scope_id=parent)
