"""Protecting every binding committed before the protection hooks went live.

The backfill reads each family of keys that binds content — workflow ledgers, task
states, artifact catalogs, freed workflows' histories and outcome finalizations — and
protects every reference they name, decoding each with the codec that wrote it and
extracting references only through typed fields. One scan of the keyspace classifies
each key into its family, a page of keys at a time and a large hash a page of fields at
a time, and protects each page in one transaction, holding no runtime lock and reading
nothing the runtime holds in memory. A generation's families complete once, and a later
root reads only those its generation has not completed; only completion is recorded,
never a scan position, so an interrupted scan starts again from the beginning. A key
that does not decode leaves the scope it names, or every scope when its scope cannot be
read, uncertified for the generation.
"""

import json
import logging
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Container, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import translate
from functools import cache
from typing import Any

from pydantic import BaseModel, ValidationError
from redis.exceptions import ResponseError

from shared.content import ContentReference
from shared.content.tree import TREE_MEDIA_TYPE
from shared.outcome import OutcomeManifest
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind

from ..clients.redis import (
    RedisClient,
    content_finalization_key,
    reclaim_coverage_key,
    reclaim_unexpanded_key,
    task_artifacts_key,
    task_state_key,
    workflow_ds_key,
    workflow_history_key,
    workflow_sources_key,
)
from ..orchestration.ledger_layout import HISTORIES, KEYED, KEYED_TEXTS, META, SETS
from ..registries.task_states import PersistedTask, load_context
from ..task.history import (
    CERTIFICATE_FIELD,
    DECLARATIONS_FIELD,
    OutputHistory,
    TaskHistory,
    output_index_field,
    output_member_field,
)
from .audit import AuditTrail
from .certification import CoverageStore
from .protection import queue_protection
from .references import reference_paths, references_of

# Keys one SCAN page asks for, fields one HSCAN page asks for, and how many identities a
# page accumulates before its protection is written.
SCAN_COUNT = 500
FIELD_COUNT = 256
FLUSH_IDENTITIES = 1000
# The pause between pages, and before a scan that failed starts again.
PAGE_PAUSE_SEC = 0.005
RETRY_PAUSE_SEC = 5.0

_OUTPUT_INDEX_PREFIX = output_index_field("")
_OUTPUT_MEMBER_PREFIX = output_member_field("", None, None, None).partition("[")[0]


class Family(StrEnum):
    """A family of keys that binds content."""

    LEDGER = "ledger"
    TASK_STATE = "task_state"
    ARTIFACTS = "artifacts"
    HISTORY = "history"
    FINALIZATION = "finalization"


FAMILIES: tuple[str, ...] = tuple(family.value for family in Family)

_PATTERNS: dict[Family, str] = {
    Family.LEDGER: workflow_ds_key("*"),
    Family.TASK_STATE: task_state_key("*"),
    Family.ARTIFACTS: task_artifacts_key("*"),
    Family.HISTORY: workflow_history_key("*"),
    Family.FINALIZATION: content_finalization_key("*", "*"),
}
_HASHES = frozenset({Family.LEDGER, Family.ARTIFACTS, Family.HISTORY})


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
    def of(cls, *values: Any, trees: Iterable[ContentReference] = ()) -> "Found":
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
    view names too, awaits expansion."""
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
    """What one family's scan read and protected."""

    keys: int = 0
    values: int = 0
    references: int = 0
    trees: int = 0
    faults: int = 0
    pages: int = 0


class _Batch:
    """The identities a page protects, written in one transaction."""

    def __init__(self) -> None:
        self.references: dict[tuple[str, str, str], ContentReference] = {}
        self.trees: dict[str, set[str]] = defaultdict(set)

    def add(self, found: Found) -> None:
        for reference in found.references:
            self.references[reference.identity] = reference
        for tree in found.trees:
            self.trees[tree.authorization_scope].add(tree.content_digest)

    def __len__(self) -> int:
        return len(self.references)


class Backfill:
    """Protects every reference the control store's binding keys name."""

    def __init__(
        self,
        rds: RedisClient,
        coverage: CoverageStore,
        logger: logging.Logger,
        audit: AuditTrail | None = None,
        *,
        page_pause_sec: float = PAGE_PAUSE_SEC,
        retry_pause_sec: float = RETRY_PAUSE_SEC,
    ) -> None:
        self._rds = rds
        self._coverage = coverage
        self._logger = logger
        self._audit = audit
        self._page_pause = page_pause_sec
        self._retry_pause = retry_pause_sec

    def run(
        self, generation: int, stop: threading.Event
    ) -> dict[str, FamilyReport] | None:
        """Scan for every family not yet complete in ``generation``, retrying a failed
        scan from its start, until all are complete; None when stopped or a later root
        moved the generation on."""
        done = self._coverage.completed(generation)
        pending = [family for family in Family if family.value not in done]
        if not pending:
            return {}
        while True:
            try:
                reports = self.scan(pending, generation, stop)
            except Exception:
                self._logger.warning(
                    "Content reclamation backfill failed; scanning again",
                    exc_info=True,
                )
                if stop.wait(self._retry_pause):
                    return None
                continue
            break
        if reports is None:
            return None
        for family, report in reports.items():
            if not self._coverage.complete(generation, family.value):
                self._logger.info(
                    "Content reclamation backfill stopped: a later root holds the "
                    "generation past %d",
                    generation,
                )
                return None
            self._logger.info(
                "Content reclamation backfill of %s complete: %d keys, %d "
                "references, %d trees, %d undecodable",
                family.value,
                report.keys,
                report.references,
                report.trees,
                report.faults,
            )
            if self._audit is not None:
                self._audit.record(
                    "backfill",
                    generation=str(generation),
                    family=family.value,
                    keys=report.keys,
                    values=report.values,
                    references=report.references,
                    trees=report.trees,
                    faults=report.faults,
                )
        return {family.value: report for family, report in reports.items()}

    def scan(
        self, families: Sequence[Family], generation: int, stop: threading.Event
    ) -> dict[Family, FamilyReport] | None:
        """Scan the keyspace once from its start, protecting what each key of
        ``families`` binds; None when stopped."""
        reports = {family: FamilyReport() for family in families}
        cursor = 0
        while True:
            cursor, keys = self._scan_page(cursor)
            classified: dict[Family, list[str]] = defaultdict(list)
            for key in keys:
                if (family := _family_of(key, reports)) is not None:
                    classified[family].append(key)
            for family in families:
                report = reports[family]
                report.pages += 1
                if not (members := classified.get(family)):
                    continue
                batch = _Batch()
                report.keys += len(members)
                if family in _HASHES:
                    for key in members:
                        self._scan_hash(family, key, generation, batch, report)
                elif family is Family.TASK_STATE:
                    self._task_states(members, generation, batch, report)
                else:
                    self._finalizations(members, generation, batch, report)
                self._flush(family, generation, batch, report)
            if cursor == 0:
                return reports
            if stop.wait(self._page_pause):
                return None

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
        self, generation: int, report: FamilyReport, key: str, exc: Exception
    ) -> None:
        scope = exc.scope if isinstance(exc, Undecodable) else None
        report.faults += 1
        self._coverage.record_fault(generation, scope)
        self._logger.error(
            "Content reclamation cannot decode %s; %s stays uncertified: %s",
            key,
            f"scope {scope}" if scope else "every scope",
            exc,
        )

    def _scan_hash(
        self,
        family: Family,
        key: str,
        generation: int,
        batch: _Batch,
        report: FamilyReport,
    ) -> None:
        decode = _FIELD_DECODERS[family]
        cursor = 0
        while True:
            try:
                cursor, fields = self._hscan_page(key, cursor)
            except ResponseError as exc:
                if "WRONGTYPE" not in str(exc):
                    raise
                self._fault(generation, report, key, exc)
                return
            for name, value in fields.items():
                report.values += 1
                try:
                    batch.add(decode(name, value))
                except Undecodable as exc:
                    self._fault(generation, report, f"{key} field {name!r}", exc)
            if len(batch) >= FLUSH_IDENTITIES:
                self._flush(family, generation, batch, report)
            if cursor == 0:
                return

    def _task_states(
        self,
        keys: Sequence[str],
        generation: int,
        batch: _Batch,
        report: FamilyReport,
    ) -> None:
        raws = self._get_many(keys)
        loaded: list[tuple[str, dict[str, Any], str | None]] = []
        wanted: dict[str, set[str]] = defaultdict(set)
        for key, raw in zip(keys, raws, strict=True):
            if raw is None:
                continue
            report.values += 1
            try:
                if isinstance(raw, Exception):
                    raise raw
                data = json.loads(raw)
                workflow_id = data["record"]["workflow_id"]
                if not isinstance(workflow_id, str):
                    raise ValueError("a task state names no workflow")
            except Exception as exc:
                self._fault(generation, report, key, exc)
                continue
            org = data["record"].get("org_id")
            loaded.append((key, data, org if isinstance(org, str) else None))
            if isinstance(digest := data.get("source_digest"), str):
                wanted[workflow_id].add(digest)
        sources = self._sources(
            [(workflow_id, sorted(digests)) for workflow_id, digests in wanted.items()]
        )
        for key, data, org in loaded:
            workflow_id = data["record"]["workflow_id"]
            try:
                task = PersistedTask.model_validate(
                    data, context=load_context(sources.get(workflow_id, {}))
                )
            except ValidationError as exc:
                self._fault(generation, report, key, Undecodable(str(exc), org))
                continue
            batch.add(Found.of(task))

    def _finalizations(
        self,
        keys: Sequence[str],
        generation: int,
        batch: _Batch,
        report: FamilyReport,
    ) -> None:
        raws = self._get_many(keys)
        for key, raw in zip(keys, raws, strict=True):
            if raw is None:
                continue
            report.values += 1
            try:
                if isinstance(raw, Exception):
                    raise Undecodable(str(raw), finalization_scope(key) or None)
                batch.add(finalization_value(key, raw))
            except Undecodable as exc:
                self._fault(generation, report, key, exc)

    def _flush(
        self, family: Family, generation: int, batch: _Batch, report: FamilyReport
    ) -> None:
        if not batch.references and not batch.trees:
            return
        scopes = {identity[0] for identity in batch.references} | set(batch.trees)
        with self._rds.sync.control_pipeline() as pipe:
            queue_protection(pipe, batch.references.values())
            for scope, digests in batch.trees.items():
                pipe.sadd(reclaim_unexpanded_key(scope), *sorted(digests))
            for scope in sorted(scopes):
                pipe.hset(reclaim_coverage_key(scope), family.value, str(generation))
            pipe.execute()
        report.references += len(batch.references)
        report.trees += sum(len(digests) for digests in batch.trees.values())
        batch.references.clear()
        batch.trees.clear()


__all__ = [
    "FAMILIES",
    "Backfill",
    "Family",
    "FamilyReport",
    "Found",
    "Undecodable",
    "artifact_field",
    "finalization_value",
    "history_field",
    "ledger_field",
]
