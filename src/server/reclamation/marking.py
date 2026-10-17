"""Marking every reference the control store binds.

One pass reads each family of keys that binds content — workflow ledgers, task states,
workflow blueprints, artifact catalogs, freed workflows' histories and outcome
finalizations — decoding each value with the codec that wrote it and extracting
references only through typed fields, into a fresh ``Mark``. One scan of the keyspace
classifies each key into its family, a page of keys at a time and a large hash a page of
fields at a time, holding no runtime lock and reading nothing the runtime holds in
memory. The pass pauses between pages and every few decoded values, so it never holds
the interpreter for long.

A value that does not decode leaves the scope it names incomplete, or every scope when
its scope cannot be read. The pass names each tree manifest it finds, for its members
to be marked through it.
"""

import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import translate
from functools import cache
from typing import Any, Self

from pydantic import BaseModel, ValidationError
from redis.exceptions import ResponseError

from shared.content import ContentReference
from shared.content.tree import TREE_MEDIA_TYPE
from shared.outcome import OutcomeManifest
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind

from ..clients.redis import (
    RedisClient,
    content_finalization_key,
    task_artifacts_key,
    task_state_key,
    workflow_blueprints_key,
    workflow_ds_key,
    workflow_history_key,
    workflow_sources_key,
)
from ..orchestration.ledger_layout import HISTORIES, KEYED, KEYED_TEXTS, META, SETS
from ..registries.task_states import PersistedTask, load_context
from ..registries.workflow import TaskBlueprints
from ..task.history import (
    CERTIFICATE_FIELD,
    DECLARATIONS_FIELD,
    OutputHistory,
    TaskHistory,
    output_index_field,
    output_member_field,
)
from .mark import Mark
from .references import reference_paths, references_of

# Keys one SCAN page asks for and fields one HSCAN page asks for.
SCAN_COUNT = 500
FIELD_COUNT = 256
# The pause between pages, and after every so many decoded values.
PAGE_PAUSE_SEC = 0.005
VALUES_PER_PAUSE = 100

_OUTPUT_INDEX_PREFIX = output_index_field("")
_OUTPUT_MEMBER_PREFIX = output_member_field("", None, None, None).partition("[")[0]


class Family(StrEnum):
    """A family of keys that binds content."""

    LEDGER = "ledger"
    TASK_STATE = "task_state"
    BLUEPRINTS = "blueprints"
    ARTIFACTS = "artifacts"
    HISTORY = "history"
    FINALIZATION = "finalization"


_PATTERNS: dict[Family, str] = {
    Family.LEDGER: workflow_ds_key("*"),
    Family.TASK_STATE: task_state_key("*"),
    Family.BLUEPRINTS: workflow_blueprints_key("*"),
    Family.ARTIFACTS: task_artifacts_key("*"),
    Family.HISTORY: workflow_history_key("*"),
    Family.FINALIZATION: content_finalization_key("*", "*"),
}
_HASHES = frozenset({Family.LEDGER, Family.ARTIFACTS, Family.HISTORY})

# The control store's key builders by the family each marks, None for one whose keys
# bind no content, so a key family added later is classified before it ships.
KEY_FAMILIES: dict[str, Family | None] = {
    "workflow_ds_key": Family.LEDGER,
    "task_state_key": Family.TASK_STATE,
    "workflow_blueprints_key": Family.BLUEPRINTS,
    "task_artifacts_key": Family.ARTIFACTS,
    "workflow_history_key": Family.HISTORY,
    "content_finalization_key": Family.FINALIZATION,
    "TASK_EVENT_STREAM_KEY": None,
    "TASK_EVENT_CURSOR_KEY": None,
    "WORKFLOWS_SET_KEY": None,
    "WORKFLOWS_BY_SUBMISSION_KEY": None,
    "WORKFLOWS_OPEN_KEY": None,
    "TASK_HISTORY_MEMBERS_KEY": None,
    "HISTORY_COUNTS_KEY": None,
    "HISTORY_SEQ_KEY": None,
    "HISTORY_INDEX_KEY": None,
    "WORKERS_SET_KEY": None,
    "WORKER_ID_SEQ_KEY": None,
    "WORKERS_CORDONED_SET_KEY": None,
    "NODES_SET_KEY": None,
    "NODE_ID_SEQ_KEY": None,
    "workflow_key": None,
    "workflow_tasks_key": None,
    "workflow_dispatched_tasks_key": None,
    "workflow_dynamic_tasks_key": None,
    "workflow_failed_tasks_key": None,
    "workflow_cancelled_tasks_key": None,
    "workflow_uploads_key": None,
    "workflow_sched_key": None,
    "workflow_sources_key": None,
    "workflow_v2_key": None,
    "workflow_revision_key": None,
    "task_history_key": None,
    "workflow_credential_key": None,
    "resident_cs_key": None,
    "resident_retired_key": None,
    "resident_replica_history_key": None,
    "resident_family_history_key": None,
    "serve_cs_key": None,
    "worker_key": None,
    "worker_hb_key": None,
    "node_key": None,
    "node_hb_key": None,
    "node_alias_lease_key": None,
    "node_dispatch_channel": None,
    "node_cmd_channel": None,
    "task_log_stream_key": None,
    "workflow_log_stream_key": None,
    "task_log_archive_last_id_key": None,
    "task_log_closed_key": None,
    "workflow_log_closed_key": None,
    "resident_relay_up_key": None,
    "resident_relay_down_key": None,
    "resident_relay_session_key": None,
    "resident_relay_down_cursor_key": None,
    "RESIDENT_RELAY_ROOT_CURSOR_KEY": None,
    "content_relay_up_key": None,
    "content_relay_down_key": None,
    "content_relay_session_key": None,
    "content_relay_down_cursor_key": None,
    "CONTENT_RELAY_ROOT_CURSOR_KEY": None,
    "ssh_relay_up_key": None,
    "ssh_relay_down_key": None,
    "ssh_relay_session_key": None,
    "ssh_relay_down_cursor_key": None,
    "SSH_RELAY_ROOT_CURSOR_KEY": None,
    "content_holders_key": None,
    "content_held_key": None,
    "content_held_seq_key": None,
    "CONTENT_HELD_WORKERS_KEY": None,
    "content_finalization_scope_key": None,
    "RECLAIM_AUDIT_KEY": None,
    "ssh_connection_key": None,
    "SSH_CONNECTION_IDS_KEY": None,
}

_FAMILY_KEY = re.compile(
    "|".join(f"(?P<{family.name}>{translate(p)})" for family, p in _PATTERNS.items())
)


def _family_of(key: str, families: Container[Family]) -> Family | None:
    """The family among ``families`` a key belongs to, matched as a key-pattern scan
    would match it."""
    if (match := _FAMILY_KEY.match(key)) is None or match.lastgroup is None:
        return None
    family = Family[match.lastgroup]
    return family if family in families else None


class Undecodable(Exception):
    """A stored value its codec refuses; ``scope`` is the scope it names, when known."""

    def __init__(self, message: str, scope: str | None = None) -> None:
        super().__init__(message)
        self.scope = scope


@dataclass
class Found:
    """What one decoded value binds: its references and the tree manifests among
    them."""

    references: list[ContentReference] = field(default_factory=list)
    trees: list[ContentReference] = field(default_factory=list)

    @classmethod
    def of(cls, *values: Any, trees: Iterable[ContentReference] = ()) -> Self:
        references = list(references_of(*values))
        manifests = [r for r in references if r.media_type == TREE_MEDIA_TYPE]
        return cls(references, [*manifests, *trees])


@cache
def _bears(model: type[BaseModel]) -> bool:
    return bool(reference_paths(model))


def _validated(model: type[BaseModel], raw: str, scope: str | None = None) -> Found:
    try:
        return Found.of(model.model_validate_json(raw))
    except ValidationError as exc:
        raise Undecodable(f"{model.__name__}: {exc}", scope) from exc


def ledger_field(name: str, value: str) -> Found:
    """The references one stored ledger field binds, decoded with its entity's model.

    A field of a collection whose model holds no reference binds nothing whatever it
    stores, so it is not decoded.
    """
    collection, sep, _ = name.partition(":")
    if not sep:
        raise Undecodable(f"malformed ledger field {name!r}")
    if collection == META or collection in SETS or collection in KEYED_TEXTS:
        return Found()
    if collection in KEYED:
        model = KEYED[collection][0]
        if not _bears(model):
            return Found()
        ordinal, sep, body = value.partition(":")
        if not sep or not ordinal.isascii() or not ordinal.isdigit():
            raise Undecodable(f"ledger field {name!r} has no ordinal")
        return _validated(model, body)
    if collection in HISTORIES:
        model = HISTORIES[collection]
        return _validated(model, value) if _bears(model) else Found()
    raise Undecodable(f"unknown ledger collection {collection!r}")


def history_field(name: str, value: str) -> Found:
    """The references one stored history field binds."""
    if name in (CERTIFICATE_FIELD, DECLARATIONS_FIELD) or name.startswith(
        _OUTPUT_INDEX_PREFIX
    ):
        return Found()
    if name.startswith(_OUTPUT_MEMBER_PREFIX):
        return _validated(OutputHistory, value)
    if name.startswith("@"):
        raise Undecodable(f"unknown history field {name!r}")
    return _validated(TaskHistory, value)


def artifact_field(name: str, value: str) -> Found:
    """The references one catalog binding names; a tree's manifest, which an archive
    view names too, binds its members."""
    try:
        binding = ArtifactBinding.model_validate_json(value)
    except ValidationError as exc:
        raise Undecodable(f"artifact binding {name!r}: {exc}") from exc
    trees = (
        [binding.reference]
        if binding.kind in (ArtifactKind.TREE, ArtifactKind.ARCHIVE)
        else []
    )
    return Found.of(binding, trees=trees)


def finalization_scope(key: str) -> str:
    """The scope a finalization key is recorded under."""
    prefix = content_finalization_key("", "")[:-1]
    return key[len(prefix) :].rpartition(":")[0]


def finalization_value(key: str, value: str) -> Found:
    return _validated(OutcomeManifest, value, finalization_scope(key) or None)


_FIELD_DECODERS: dict[Family, Callable[[str, str], Found]] = {
    Family.LEDGER: ledger_field,
    Family.ARTIFACTS: artifact_field,
    Family.HISTORY: history_field,
}


@dataclass
class FamilyReport:
    """What one family's pass read and marked."""

    keys: int = 0
    values: int = 0
    references: int = 0
    trees: int = 0
    faults: int = 0


@dataclass
class MarkPass:
    """One pass over the control store: what it marked and what it could not read."""

    mark: Mark
    started: float
    """The wall-clock time the pass began."""
    trees: dict[tuple[str, str], ContentReference] = field(default_factory=dict)
    """Every bound tree manifest, by scope and digest."""
    faults: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Values that did not decode, by the scope they name."""
    unattributed: int = 0
    """Values that did not decode and name no readable scope."""
    complete: bool = False
    """Whether the pass read the whole keyspace."""
    reports: dict[str, FamilyReport] = field(default_factory=dict)
    seconds: float = 0.0

    def gap(self, scope: str) -> str:
        """Why the pass leaves ``scope``'s coverage incomplete, or "" when it does
        not."""
        if not self.complete:
            return "the marking pass did not finish"
        if self.unattributed:
            return f"{self.unattributed} undecodable value(s) of unknown scope"
        if count := self.faults.get(scope, 0):
            return f"{count} undecodable value(s)"
        return ""


class _Pacer:
    """Pauses after every so many values, so a pass yields the interpreter."""

    def __init__(self, stop: threading.Event, every: int, pause_sec: float) -> None:
        self._stop = stop
        self._every = every
        self._pause = pause_sec
        self._count = 0

    def tick(self) -> bool:
        """Count one value; True once the pass is to stop."""
        self._count += 1
        if self._count % self._every:
            return self._stop.is_set()
        return self._stop.wait(self._pause)

    def pause(self) -> bool:
        return self._stop.wait(self._pause)


class _Stopped(Exception):
    pass


class Marker:
    """Marks every reference the control store's binding keys name."""

    def __init__(
        self,
        rds: RedisClient,
        logger: logging.Logger,
        *,
        page_pause_sec: float = PAGE_PAUSE_SEC,
        values_per_pause: int = VALUES_PER_PAUSE,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._rds = rds
        self._logger = logger
        self._page_pause = page_pause_sec
        self._values_per_pause = values_per_pause
        self._clock = clock

    def run(
        self, stop: threading.Event, capacities: Mapping[str, int] | None = None
    ) -> MarkPass:
        """Read the keyspace once from its start into a fresh mark sized by
        ``capacities``; a set ``stop`` ends the pass incomplete."""
        started = time.monotonic()
        result = MarkPass(Mark(capacities), self._clock())
        result.reports = {family.value: FamilyReport() for family in Family}
        pacer = _Pacer(stop, self._values_per_pause, self._page_pause)
        try:
            cursor = 0
            while True:
                cursor, keys = self._scan_page(cursor)
                classified: dict[Family, list[str]] = defaultdict(list)
                for key in keys:
                    if (family := _family_of(key, Family)) is not None:
                        classified[family].append(key)
                for family, members in classified.items():
                    report = result.reports[family.value]
                    report.keys += len(members)
                    if family in _HASHES:
                        for key in members:
                            self._scan_hash(family, key, result, report, pacer)
                    elif family in (Family.TASK_STATE, Family.BLUEPRINTS):
                        self._tasks(family, members, result, report, pacer)
                    else:
                        self._finalizations(members, result, report, pacer)
                if cursor == 0:
                    result.complete = True
                    break
                if pacer.pause():
                    break
        except _Stopped:
            pass
        result.seconds = time.monotonic() - started
        return result

    def _scan_page(self, cursor: int) -> tuple[int, list[str]]:
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            pipe.scan(cursor, count=SCAN_COUNT)
            next_cursor, keys = pipe.execute()[0]
        return int(next_cursor), list(keys)

    def _hscan_page(self, key: str, cursor: int) -> tuple[int, dict[str, str]]:
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            pipe.hscan(key, cursor, count=FIELD_COUNT)
            next_cursor, fields = pipe.execute()[0]
        return int(next_cursor), dict(fields)

    def _get_many(self, keys: Sequence[str]) -> list[Any]:
        """Each key's value, or the error reading it raised."""
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            for key in keys:
                pipe.get(key)
            return pipe.execute(raise_on_error=False)

    def _sources(
        self, wanted: Sequence[tuple[str, list[str]]]
    ) -> dict[str, dict[str, str]]:
        """The stored sources each workflow's task states name, by digest."""
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            for workflow_id, digests in wanted:
                pipe.hmget(workflow_sources_key(workflow_id), digests)
            replies = pipe.execute()
        return {
            workflow_id: {
                digest: text
                for digest, text in zip(digests, texts, strict=True)
                if text is not None
            }
            for (workflow_id, digests), texts in zip(wanted, replies, strict=True)
        }

    def _fault(
        self,
        result: MarkPass,
        report: FamilyReport,
        key: str,
        exc: Exception,
        pacer: _Pacer | None = None,
    ) -> None:
        scope = exc.scope if isinstance(exc, Undecodable) else None
        report.faults += 1
        if scope:
            result.faults[scope] += 1
        else:
            result.unattributed += 1
        self._logger.error(
            "Content audit cannot decode %s; %s stays incomplete: %s",
            key,
            f"scope {scope}" if scope else "every scope",
            exc,
        )
        if pacer is not None and pacer.tick():
            raise _Stopped

    def _add(
        self, result: MarkPass, report: FamilyReport, found: Found, pacer: _Pacer
    ) -> None:
        for reference in found.references:
            result.mark.add(reference.authorization_scope, reference.content_digest)
        for tree in found.trees:
            result.trees[(tree.authorization_scope, tree.content_digest)] = tree
        report.references += len(found.references)
        report.trees += len(found.trees)
        if pacer.tick():
            raise _Stopped

    def _scan_hash(
        self,
        family: Family,
        key: str,
        result: MarkPass,
        report: FamilyReport,
        pacer: _Pacer,
    ) -> None:
        decode = _FIELD_DECODERS[family]
        cursor = 0
        while True:
            try:
                cursor, fields = self._hscan_page(key, cursor)
            except ResponseError as exc:
                if "WRONGTYPE" not in str(exc):
                    raise
                self._fault(result, report, key, exc, pacer)
                return
            for name, value in fields.items():
                report.values += 1
                try:
                    found = decode(name, value)
                except Undecodable as exc:
                    self._fault(result, report, f"{key} field {name!r}", exc, pacer)
                    continue
                self._add(result, report, found, pacer)
            if cursor == 0:
                return
            if pacer.pause():
                raise _Stopped

    def _tasks(
        self,
        family: Family,
        keys: Sequence[str],
        result: MarkPass,
        report: FamilyReport,
        pacer: _Pacer,
    ) -> None:
        """Decode task states or blueprints, which load under their workflow's stored
        sources."""
        raws = self._get_many(keys)
        loaded: list[tuple[str, str, Any, str | None]] = []
        wanted: dict[str, set[str]] = defaultdict(set)
        for key, raw in zip(keys, raws, strict=True):
            if raw is None:
                continue
            report.values += 1
            try:
                if isinstance(raw, Exception):
                    raise raw
                data = json.loads(raw)
                records = data["tasks"] if family is Family.BLUEPRINTS else [data]
                workflow_id = _workflow_of(records, key)
            except Exception as exc:
                self._fault(result, report, key, exc, pacer)
                continue
            org = next(
                (r["record"].get("org_id") for r in records if "record" in r), None
            )
            loaded.append(
                (key, workflow_id, raw, org if isinstance(org, str) else None)
            )
            for record in records:
                if isinstance(digest := record.get("source_digest"), str):
                    wanted[workflow_id].add(digest)
        sources = self._sources(
            [(workflow_id, sorted(digests)) for workflow_id, digests in wanted.items()]
        )
        model: type[BaseModel] = (
            TaskBlueprints if family is Family.BLUEPRINTS else PersistedTask
        )
        for key, workflow_id, raw, org in loaded:
            try:
                value = model.model_validate_json(
                    raw, context=load_context(sources.get(workflow_id, {}))
                )
            except ValidationError as exc:
                self._fault(result, report, key, Undecodable(str(exc), org), pacer)
                continue
            self._add(result, report, Found.of(value), pacer)

    def _finalizations(
        self,
        keys: Sequence[str],
        result: MarkPass,
        report: FamilyReport,
        pacer: _Pacer,
    ) -> None:
        raws = self._get_many(keys)
        for key, raw in zip(keys, raws, strict=True):
            if raw is None:
                continue
            report.values += 1
            try:
                if isinstance(raw, Exception):
                    raise Undecodable(str(raw), finalization_scope(key) or None)
                found = finalization_value(key, raw)
            except Undecodable as exc:
                self._fault(result, report, key, exc, pacer)
                continue
            self._add(result, report, found, pacer)


def _workflow_of(records: Sequence[Any], key: str) -> str:
    """The workflow the decoded task states of ``key`` belong to."""
    if not records:
        prefix = workflow_blueprints_key("\0").partition("\0")
        if key.startswith(prefix[0]) and key.endswith(prefix[2]):
            return key[len(prefix[0]) : len(key) - len(prefix[2])]
        raise ValueError("a task state names no workflow")
    workflow_ids = {record["record"]["workflow_id"] for record in records}
    if len(workflow_ids) != 1 or not isinstance(
        workflow_id := next(iter(workflow_ids)), str
    ):
        raise ValueError("task states name no single workflow")
    return workflow_id


__all__ = [
    "KEY_FAMILIES",
    "Family",
    "FamilyReport",
    "Found",
    "MarkPass",
    "Marker",
    "Undecodable",
    "artifact_field",
    "finalization_value",
    "history_field",
    "ledger_field",
]
