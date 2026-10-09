"""The loops and definition children that run region-definition occurrences."""

from ...task.v2.representations.template import DefinitionKind
from ..state import LoopInstanceStatus, Occurrence
from .advance import Advance
from .edges import Incoming
from .ledger import OrchestrationLedger
from .loops import LoopProgress
from .spawns import SpawnRegions
from .topology import PlanTopology

_OPEN_LOOP = frozenset({LoopInstanceStatus.OPEN, LoopInstanceStatus.EXITED})


class RegionContexts:
    """Routes a region-definition occurrence's returns and failures to the loop or
    child running it, and closes each loop and child once its scope drains."""

    def __init__(
        self,
        ledger: OrchestrationLedger,
        topology: PlanTopology,
        loops: LoopProgress,
        spawns: SpawnRegions,
    ) -> None:
        self._ledger = ledger
        self._topology = topology
        self._loops = loops
        self._spawns = spawns

    def _kind(self, operator_id: str) -> DefinitionKind | None:
        definition_id = self._topology.definition_of.get(operator_id)
        definition = self._topology.definitions.get(definition_id or "")
        return definition.kind if definition else None

    def ingress(self, key: str, inputs: list[Incoming], advance: Advance) -> None:
        self._loops.ingress(key, inputs, advance)

    def on_return(self, key: str, advance: Advance) -> None:
        operator_id = self._ledger.occurrence(key).operator_id
        match self._kind(operator_id):
            case DefinitionKind.LOOP_BODY:
                self._loops.on_return(key, advance)
            case DefinitionKind.CHILD:
                self._spawns.on_child_return(key, advance)

    def on_failure(self, occurrence: Occurrence, reason: str, advance: Advance) -> None:
        match self._kind(occurrence.operator_id):
            case DefinitionKind.LOOP_BODY if occurrence.time:
                instance = self._ledger.loop_instances.get(occurrence.time[-1].loop)
                if instance is not None:
                    self._loops.fail(instance, reason, advance)
            case DefinitionKind.CHILD:
                context = self._ledger.child_contexts.get(occurrence.context_id)
                if context is not None:
                    self._spawns.fail_child(context, reason, advance)

    def on_cancelled(self, scope_id: str) -> None:
        self._loops.cancelled(scope_id)

    def sweep(self, advance: Advance) -> Advance:
        """Close every loop and definition child whose scope has drained.

        Closing one can drain an enclosing one, so this repeats until a pass closes
        nothing.
        """
        while True:
            before = self._closed_count()
            for scope_id in sorted(self._ledger.active_loops):
                instance = self._ledger.loop_instances[scope_id]
                if instance.status not in _OPEN_LOOP:
                    self._ledger.active_loops.discard(scope_id)
                    continue
                self._loops.maybe_release(instance, advance)
                self._loops.check_drained_time(instance, advance)
            for context_id in sorted(self._ledger.active_contexts):
                self._spawns.maybe_settle_child(
                    self._ledger.child_contexts[context_id], advance
                )
            if self._closed_count() == before:
                return advance

    def _closed_count(self) -> int:
        return len(self._ledger.released_scopes) - len(self._ledger.active_contexts)
