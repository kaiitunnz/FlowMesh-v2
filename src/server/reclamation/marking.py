"""Marking every reference the control store binds.

One pass reads each family of keys that binds content — workflow ledgers, task states,
workflow blueprints, artifact catalogs, freed workflows' histories and outcome
finalizations — decoding each value with the codec that wrote it and extracting
references only through typed fields, into a fresh ``Mark``. One scan of the keyspace
classifies each key into its family, a page of keys at a time and a large hash a page of
fields at a time, holding no runtime lock and reading nothing the runtime holds in
memory. The pass pauses between pages and every few decoded values, for long enough
that it takes only a small share of the interpreter.

A value that does not decode leaves incomplete the scope it names, or else the scope of
the workflow or task its key belongs to, and every scope when neither can be read. A
pass across which the control store restarted or another server took over leaves every
scope incomplete, since a scan cursor does not survive that. The pass names each tree
manifest it finds, packed, for its members to be marked through it.
"""

import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import translate
from typing import Any, Self

from pydantic import BaseModel, ValidationError
from redis.exceptions import RedisError, ResponseError

from shared.content import ContentReference, DigestAlgorithm
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
    workflow_key,
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
from .mark import Mark, digest_bytes, is_content_digest
from .pacing import DUTY, MIN_PAUSE_SEC, Pacer
from .references import model_bears, references_of

# Keys one SCAN page asks for and fields one HSCAN page asks for.
SCAN_COUNT = 500
FIELD_COUNT = 256
# How many decoded values run between pauses.
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
    "CONTENT_AUDIT_KEY": None,
    "ssh_connection_key": None,
    "SSH_CONNECTION_IDS_KEY": None,
}

_FAMILY_KEY = re.compile(
    "|".join(f"(?P<{family.name}>{translate(p)})" for family, p in _PATTERNS.items())
)


def _family_of(key: str) -> Family | None:
    """The family a key belongs to, matched as a key-pattern scan would match it."""
    if (match := _FAMILY_KEY.match(key)) is None or match.lastgroup is None:
        return None
    return Family[match.lastgroup]


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
        named = tuple(trees)
        manifests = {
            r.identity: r
            for r in (*references, *named)
            if r.media_type == TREE_MEDIA_TYPE or r in named
        }
        return cls(references, list(manifests.values()))


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
        if not model_bears(model):
            return Found()
        ordinal, sep, body = value.partition(":")
        if not sep or not ordinal.isascii() or not ordinal.isdigit():
            raise Undecodable(f"ledger field {name!r} has no ordinal")
        return _validated(model, body)
    if collection in HISTORIES:
        model = HISTORIES[collection]
        return _validated(model, value) if model_bears(model) else Found()
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

# A packed tree reference: its digest, then its size as 8 big-endian bytes.
_TREE_RECORD = 40


@dataclass
class MarkPass:
    """One pass over the control store: what it marked and what it could not read."""

    mark: Mark
    started: float
    """The wall-clock time the pass began."""
    faults: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Values that did not decode, by the scope they belong to."""
    unattributed: int = 0
    """Values that did not decode and belong to no readable scope."""
    complete: bool = False
    """Whether the pass read the whole keyspace."""
    interrupted: str = ""
    """Why the pass cannot vouch for what it read, when it cannot."""
    seconds: float = 0.0
    _trees: dict[str, bytearray] = field(default_factory=lambda: defaultdict(bytearray))

    def add_tree(self, tree: ContentReference) -> None:
        self._trees[tree.authorization_scope] += digest_bytes(
            tree.content_digest
        ) + tree.size_bytes.to_bytes(8, "big")

    def tree_references(self) -> Iterator[ContentReference]:
        """Every bound tree manifest the pass found, once per binding that names it."""
        for scope, records in self._trees.items():
            for at in range(0, len(records), _TREE_RECORD):
                record = records[at : at + _TREE_RECORD]
                yield ContentReference(
                    authorization_scope=scope,
                    digest_algorithm=DigestAlgorithm.SHA256,
                    content_digest=record[:32].hex(),
                    size_bytes=int.from_bytes(record[32:], "big"),
                    media_type=TREE_MEDIA_TYPE,
                )

    def tree_scopes(self) -> Iterable[str]:
        """Every scope the pass found a bound tree in."""
        return self._trees.keys()

    @property
    def trees_nbytes(self) -> int:
        return sum(len(records) for records in self._trees.values())

    def gap(self, scope: str) -> str:
        """Why the pass leaves ``scope``'s coverage incomplete, or "" when it does
        not."""
        if self.interrupted:
            return self.interrupted
        if not self.complete:
            return "the marking pass did not finish"
        if self.unattributed:
            return f"{self.unattributed} undecodable value(s) of unknown scope"
        if count := self.faults.get(scope, 0):
            return f"{count} undecodable value(s)"
        return ""


class _BlankSources(Mapping[str, str]):
    """Stored sources that answer every digest with a blank source."""

    def __getitem__(self, digest: str) -> str:
        return ""

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


_BLANK_SOURCE_CONTEXT = load_context(_BlankSources())


class _Stopped(Exception):
    pass


class _Pass:
    """The state one marking pass reads through: its result, its pacer, and what it
    has already looked up."""

    def __init__(self, result: MarkPass, pacer: Pacer) -> None:
        self.result = result
        self.pacer = pacer
        self.task_scopes: dict[str, str | None] = {}
        self.workflow_scopes: dict[str, str | None] = {}

    def tick(self) -> None:
        if self.pacer.tick():
            raise _Stopped

    def pause(self) -> None:
        if self.pacer.pause():
            raise _Stopped


def _owner(key: str) -> tuple[str, str] | None:
    """Whether ``key`` belongs to a task or a workflow, and which."""
    prefix, _, rest = key.partition(":")
    owner_id, _, _ = rest.partition(":")
    if prefix in ("task", "workflow") and owner_id:
        return prefix, owner_id
    return None


class Marker:
    """Marks every reference the control store's binding keys name."""

    def __init__(
        self,
        rds: RedisClient,
        logger: logging.Logger,
        *,
        min_pause_sec: float = MIN_PAUSE_SEC,
        values_per_pause: int = VALUES_PER_PAUSE,
        duty: float = DUTY,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._rds = rds
        self._logger = logger
        self._min_pause = min_pause_sec
        self._values_per_pause = values_per_pause
        self._duty = duty
        self._clock = clock

    def run(
        self, stop: threading.Event, capacities: Mapping[str, int] | None = None
    ) -> MarkPass:
        """Read the keyspace once from its start into a fresh mark sized by
        ``capacities``; a set ``stop`` ends the pass incomplete."""
        started = time.monotonic()
        result = MarkPass(Mark(capacities), self._clock())
        state = _Pass(
            result,
            Pacer(
                stop,
                self._values_per_pause,
                duty=self._duty,
                min_pause_sec=self._min_pause,
            ),
        )
        identity = self._identity(result)
        if result.interrupted:
            return result
        try:
            cursor = 0
            while True:
                cursor, keys = self._rds.sync.scan_step(cursor, SCAN_COUNT)
                classified: dict[Family, list[str]] = defaultdict(list)
                for key in keys:
                    if (family := _family_of(key)) is not None:
                        classified[family].append(key)
                for family, members in classified.items():
                    if decode := _FIELD_DECODERS.get(family):
                        self._hashes(members, decode, identity, state)
                    elif family in (Family.TASK_STATE, Family.BLUEPRINTS):
                        self._tasks(family, members, state)
                    else:
                        self._finalizations(members, state)
                if cursor == 0:
                    self._check_identity(identity, result)
                    result.complete = not result.interrupted
                    break
                state.pause()
        except _Stopped:
            pass
        result.seconds = time.monotonic() - started
        return result

    def _identity(self, result: MarkPass) -> str:
        """The control store's server identity; one it cannot vouch by leaves
        ``result`` interrupted."""
        try:
            identity = self._rds.sync.server_identity()
        except RedisError as exc:
            self._interrupt(
                result, f"the control store's server identity cannot be read: {exc}"
            )
            return ""
        if not identity.partition(":")[0]:
            self._interrupt(result, "the control store reports no server identity")
        return identity

    def _check_identity(self, identity: str, result: MarkPass) -> None:
        if result.interrupted or self._identity(result) == identity:
            return
        self._interrupt(
            result, "the control store restarted or another server took over mid-pass"
        )

    def _interrupt(self, result: MarkPass, why: str) -> None:
        if not result.interrupted:
            result.interrupted = why
            self._logger.warning("Content audit pass is incomplete: %s", why)

    def _get_many(self, keys: Sequence[str]) -> list[Any]:
        """Each key's value, or the error reading it raised."""
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            for key in keys:
                pipe.get(key)
            return pipe.execute(raise_on_error=False)

    def _task_scope(self, task_id: str, state: _Pass) -> str | None:
        if task_id not in state.task_scopes:
            raw = self._rds.sync.get(task_state_key(task_id))
            scope = None
            try:
                org = json.loads(raw)["record"]["org_id"] if raw else None
                scope = org if isinstance(org, str) and org else None
            except (ValueError, KeyError, TypeError):
                pass
            state.task_scopes[task_id] = scope
        return state.task_scopes[task_id]

    def _workflow_scope(self, workflow_id: str, state: _Pass) -> str | None:
        if workflow_id not in state.workflow_scopes:
            scope = None
            raw = self._rds.sync.hash_mget(workflow_key(workflow_id), ["task_ids"])[0]
            try:
                task_ids = json.loads(raw) if raw else []
            except ValueError:
                task_ids = []
            for task_id in task_ids if isinstance(task_ids, list) else []:
                if isinstance(task_id, str) and (
                    scope := self._task_scope(task_id, state)
                ):
                    break
            state.workflow_scopes[workflow_id] = scope
        return state.workflow_scopes[workflow_id]

    def _key_scope(self, key: str, state: _Pass) -> str | None:
        """The scope of the task or workflow ``key`` belongs to, when it can be read."""
        try:
            match _owner(key):
                case ("task", task_id):
                    return self._task_scope(task_id, state)
                case ("workflow", workflow_id):
                    return self._workflow_scope(workflow_id, state)
        except RedisError:
            pass
        return None

    def _fault(self, key: str, label: str, exc: Exception, state: _Pass) -> None:
        scope = exc.scope if isinstance(exc, Undecodable) else None
        scope = scope or self._key_scope(key, state)
        if scope:
            state.result.faults[scope] += 1
        else:
            state.result.unattributed += 1
        self._logger.error(
            "Content audit cannot decode %s; %s stays incomplete: %s",
            label,
            f"scope {scope}" if scope else "every scope",
            exc,
        )
        state.tick()

    def _add(self, key: str, found: Found, state: _Pass) -> None:
        for reference in found.references:
            if self._checked(key, reference, state):
                state.result.mark.add(
                    reference.authorization_scope, reference.content_digest
                )
        for tree in found.trees:
            if is_content_digest(tree.content_digest):
                state.result.add_tree(tree)
            elif tree not in found.references:
                self._checked(key, tree, state)
        state.tick()

    def _checked(self, key: str, reference: ContentReference, state: _Pass) -> bool:
        """Whether ``reference`` names a content digest, faulting its scope if not."""
        if is_content_digest(reference.content_digest):
            return True
        why = f"{reference.content_digest!r} is not a content digest"
        self._fault(key, key, Undecodable(why, reference.authorization_scope), state)
        return False

    def _hashes(
        self,
        keys: Sequence[str],
        decode: Callable[[str, str], Found],
        identity: str,
        state: _Pass,
    ) -> None:
        """Decode every field of each hash, reading the first page of all of them in
        one round trip."""
        for key, first in zip(
            keys, self._rds.sync.scan_hashes(keys, FIELD_COUNT), strict=True
        ):
            page, paged = first, False
            while True:
                if isinstance(page, Exception):
                    if not isinstance(page, ResponseError) or "WRONGTYPE" not in str(
                        page
                    ):
                        raise page
                    self._fault(key, key, page, state)
                    break
                cursor, fields = page
                for name, value in fields.items():
                    try:
                        found = decode(name, value)
                    except Undecodable as exc:
                        self._fault(key, f"{key} field {name!r}", exc, state)
                        continue
                    self._add(key, found, state)
                if cursor == 0:
                    break
                state.pause()
                page, paged = self._rds.sync.scan_hash(key, cursor, FIELD_COUNT), True
            if paged:
                self._check_identity(identity, state.result)

    def _tasks(self, family: Family, keys: Sequence[str], state: _Pass) -> None:
        """Decode task states or blueprints. Their workflow source binds no content,
        so they load with a blank one rather than the stored source."""
        model: type[BaseModel] = (
            TaskBlueprints if family is Family.BLUEPRINTS else PersistedTask
        )
        for key, raw in zip(keys, self._get_many(keys), strict=True):
            if raw is None:
                continue
            try:
                if isinstance(raw, Exception):
                    raise raw
                value = model.model_validate_json(raw, context=_BLANK_SOURCE_CONTEXT)
            except Exception as exc:
                self._fault(key, key, exc, state)
                continue
            self._add(key, Found.of(value), state)

    def _finalizations(self, keys: Sequence[str], state: _Pass) -> None:
        for key, raw in zip(keys, self._get_many(keys), strict=True):
            if raw is None:
                continue
            try:
                if isinstance(raw, Exception):
                    raise Undecodable(str(raw), finalization_scope(key) or None)
                found = finalization_value(key, raw)
            except Undecodable as exc:
                self._fault(key, key, exc, state)
                continue
            self._add(key, found, state)


__all__ = [
    "KEY_FAMILIES",
    "Family",
    "Found",
    "MarkPass",
    "Marker",
    "Undecodable",
    "artifact_field",
    "finalization_value",
    "history_field",
    "ledger_field",
]
