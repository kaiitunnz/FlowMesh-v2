import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml
from pydantic import SecretStr

from shared.tasks.credentials import (
    TaskSpec,
    find_spec_credentials,
    holds_placeholder,
    mask_spec_values,
    set_spec_values,
    spec_value,
)
from shared.tasks.specs import AgentSpecStrict, AgentSpecTemplate
from shared.utils.ids import new_credential_ref
from shared.utils.redact import REDACTED, redact_credential_fields

from .parser import ParsedWorkflow


def pop_inline_model_secrets(parsed: ParsedWorkflow) -> dict[str, SecretStr]:
    """Remove and return each agent's inline model credential, keyed by task id.

    The api_key is stripped from the parsed spec in place, so no credential rides into
    the compiled template or the persisted task record. The returned map feeds the
    vault and the generated ref pinned on each agent's compiled model binding.
    """
    secrets: dict[str, SecretStr] = {}
    for task in parsed.tasks:
        spec = task.task.spec
        if not isinstance(spec, (AgentSpecStrict, AgentSpecTemplate)):
            continue
        binding = spec.model_binding
        if binding is not None and binding.api_key is not None:
            secrets[task.task_id] = binding.api_key
            binding.api_key = None
    return secrets


@dataclass(frozen=True)
class TaskCredentials:
    """Where one task's vaulted credentials sit in its spec.

    ``refs`` maps each credential's pointer to the ref it is vaulted under. A task one
    of whose vaulted values renders from an upstream stage is ``renders``: the ref
    names the unrendered value, so it cannot tell two rendered credentials apart.
    """

    refs: dict[str, str] = field(default_factory=dict)
    renders: bool = False


@dataclass(frozen=True)
class InlineCredentials:
    """The inline credentials taken out of one submission.

    ``values`` is what the vault stores, keyed by ref; each distinct value within the
    submission has one ref, so tasks carrying the same credential name the same ref.
    """

    values: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, TaskCredentials] = field(default_factory=dict)


class CredentialRefs:
    """Mints the refs one workflow's credentials are vaulted under, one per distinct
    value, and collects the values to vault."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self._by_identity: dict[str, str] = {}

    def ref(self, value: Any) -> str:
        identity = json.dumps(value, sort_keys=True)
        if (ref := self._by_identity.get(identity)) is None:
            ref = self._by_identity[identity] = new_credential_ref()
            self.values[ref] = value
        return ref


def take_spec_credentials(
    spec: TaskSpec, refs: CredentialRefs | None
) -> TaskCredentials:
    """Mask every inline credential in ``spec`` in place, minting a ref for each from
    ``refs``; with no ``refs`` the credentials are only masked."""
    found = find_spec_credentials(spec)
    taken = TaskCredentials(
        refs=(
            {pointer: refs.ref(value) for pointer, value in found.items()}
            if refs is not None
            else {}
        ),
        renders=any(holds_placeholder(value) for value in found.values()),
    )
    mask_spec_values(spec, found)
    return taken


def take_inline_credentials(parsed: ParsedWorkflow) -> InlineCredentials:
    """Mask every inline task-spec credential in ``parsed`` in place and return them.

    Each credential is replaced by its marker where it sits, so no value reaches a
    persisted record or the compiled template and plan.
    """
    refs = CredentialRefs()
    tasks = {
        task.task_id: take_spec_credentials(task.task.spec, refs)
        for task in parsed.tasks
    }
    return InlineCredentials(values=refs.values, tasks=tasks)


def mask_inline_credentials(parsed: ParsedWorkflow) -> None:
    """Mask every inline task-spec credential in ``parsed`` in place."""
    for task in parsed.tasks:
        take_spec_credentials(task.task.spec, None)


def credential_merge_key(
    spec: TaskSpec, refs: Mapping[str, str], **context: Any
) -> str | None:
    """The merge key of a spec whose vaulted credentials are named by their refs.

    Tasks share a key only when their vaulted credentials are the same values, and the
    key carries no credential.
    """
    if not refs:
        return spec.merge_key(**context)
    keyed = spec.model_copy(deep=True)
    set_spec_values(
        keyed,
        {
            pointer: [ref] if isinstance(spec_value(spec, pointer), list) else ref
            for pointer, ref in refs.items()
        },
    )
    return keyed.merge_key(**context)


def credential_scrubber(values: Iterable[Any]) -> Callable[[str], str]:
    """A function masking every occurrence of ``values`` in a text.

    Applied to any text produced from a task whose credentials were restored, so an
    error or log line quoting the spec carries no credential.
    """
    needles: set[str] = set()
    for value in values:
        needles.update(_strings(value))
        if not isinstance(value, str):
            needles.add(json.dumps(value))
    ordered = sorted((n for n in needles if n), key=len, reverse=True)

    def scrub(text: str) -> str:
        for needle in ordered:
            text = text.replace(needle, REDACTED)
        return text

    return scrub


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _redact_embedded_json(text: str) -> str:
    try:
        return json.dumps(redact_credential_fields(json.loads(text)))
    except (ValueError, TypeError, RecursionError):
        return REDACTED


def _redact_n8n_document(document: Any) -> Any:
    """Mask an n8n document, including the JSON each node embeds as a string.

    A Set node carries its output as a JSON string in ``parameters.jsonOutput``, which
    the parser loads into a task spec, so its credential fields are masked like any
    other; a string that is not JSON is masked whole.
    """
    redacted = redact_credential_fields(document)
    nodes = redacted.get("nodes") if isinstance(redacted, dict) else None
    for node in nodes if isinstance(nodes, list) else []:
        params = node.get("parameters") if isinstance(node, dict) else None
        if isinstance(params, dict) and isinstance(
            json_output := params.get("jsonOutput"), str
        ):
            params["jsonOutput"] = _redact_embedded_json(json_output)
    return redacted


def redact_stored_source(source: str) -> str:
    """Redact a source stored as the redacted form of either submission format."""
    return redact_source_text(
        source, "n8n" if source.lstrip().startswith("{") else "native"
    )


def redact_source_text(raw_payload: str, format: str) -> str:
    """Return the submitted source with every credential value masked.

    Redaction is structural: the payload is parsed with its submission format's parser,
    every credential value is masked, and the document is re-serialized in that format,
    so a credential never survives in the captured source however it was quoted,
    escaped, duplicated, or commented out. The re-serialized source keeps no comments or
    formatting. A payload that does not parse or re-serialize is replaced by
    ``REDACTED``.
    """
    try:
        if format == "n8n":
            return json.dumps(_redact_n8n_document(json.loads(raw_payload)), indent=2)
        return yaml.safe_dump(
            redact_credential_fields(yaml.safe_load(raw_payload)), sort_keys=False
        )
    except (ValueError, TypeError, RecursionError, yaml.YAMLError):
        return REDACTED
