"""In-memory durable task states and ledgers, stored and loaded through the
registry's codecs."""

from collections.abc import Sequence

from server.orchestration import LedgerSnapshot
from server.orchestration.ledger_fields import (
    LedgerChanges,
    StoredLedger,
    decode_ledger,
)
from server.registries.workflow import (
    PersistedTask,
    load_task_state,
    task_sources,
)
from server.task.models import TaskRecord


class StoredTaskStates:
    """Task states as the registry stores them: each record without its workflow
    source, which is stored once per workflow and must be held before a task names
    it."""

    def __init__(self) -> None:
        super().__init__()
        self.task_blobs: dict[str, str] = {}
        self.sources: dict[str, dict[str, str]] = {}
        self.task_workflows: dict[str, str] = {}

    def put_sources(self, items: Sequence[PersistedTask]) -> None:
        for workflow_id, sources in task_sources(items).items():
            self.sources.setdefault(workflow_id, {}).update(sources)

    def keep_sources(self, workflow_id: str, items: Sequence[PersistedTask]) -> None:
        self.put_sources(items)

    async def keep_sources_async(
        self, workflow_id: str, items: Sequence[PersistedTask]
    ) -> None:
        self.put_sources(items)

    def put_tasks(self, items: Sequence[PersistedTask]) -> None:
        for item in items:
            record = item.record
            digest = record.source_digest()
            assert digest in self.sources.get(record.workflow_id, {}), digest
            self.task_blobs[record.task_id] = item.model_dump_json()
            self.task_workflows[record.task_id] = record.workflow_id

    def load_task_states(
        self, workflow_id: str, *task_ids: str
    ) -> list[PersistedTask | None]:
        sources = self.sources.get(workflow_id, {})
        return [
            load_task_state(blob, sources) if (blob := self.task_blobs.get(t)) else None
            for t in task_ids
        ]

    async def load_task_states_async(
        self, workflow_id: str, *task_ids: str
    ) -> list[PersistedTask | None]:
        return self.load_task_states(workflow_id, *task_ids)

    def stored_task(self, task_id: str) -> PersistedTask | None:
        if (workflow_id := self.task_workflows.get(task_id)) is None:
            return None
        return self.load_task_states(workflow_id, task_id)[0]

    def stored_record(self, task_id: str) -> TaskRecord:
        stored = self.stored_task(task_id)
        assert stored is not None, task_id
        return stored.record

    def source_texts(self) -> list[str]:
        """Each workflow source as stored."""
        return [text for sources in self.sources.values() for text in sources.values()]

    def forget_workflow_tasks(self, workflow_id: str, task_ids: Sequence[str]) -> None:
        for task_id in task_ids:
            self.task_blobs.pop(task_id, None)
            self.task_workflows.pop(task_id, None)
        self.sources.pop(workflow_id, None)


class StoredLedgers:
    """Ledgers as the registry stores them: each workflow's ledger as its fields,
    written by the changes each write carries."""

    def __init__(self) -> None:
        super().__init__()
        self.ledgers: dict[str, dict[str, str]] = {}

    def put_ledger(self, workflow_id: str, changes: LedgerChanges) -> None:
        # A new dict per write, so a test holding an earlier ledger keeps it.
        fields = {} if changes.reset else dict(self.ledgers.get(workflow_id, {}))
        fields.update(changes.fields)
        for name in changes.deleted:
            fields.pop(name, None)
        self.ledgers[workflow_id] = fields

    def load_ledger(self, workflow_id: str) -> StoredLedger | None:
        fields = self.ledgers.get(workflow_id)
        return decode_ledger(fields) if fields else None

    async def load_ledger_async(self, workflow_id: str) -> StoredLedger | None:
        return self.load_ledger(workflow_id)

    def ledger(self, workflow_id: str) -> LedgerSnapshot | None:
        stored = self.load_ledger(workflow_id)
        return stored.snapshot if stored is not None else None

    def ledger_texts(self) -> list[str]:
        """Each stored ledger's fields as one text."""
        return ["".join([*f.keys(), *f.values()]) for f in self.ledgers.values()]
