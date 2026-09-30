import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml
from pydantic import SecretStr

from shared.tasks.credentials import (
    find_spec_credentials,
    holds_placeholder,
    mask_spec_values,
    set_spec_values,
    spec_value,
)
from shared.tasks.specs import AgentSpecTemplate, TaskSpecBase, TaskSpecTemplateBase
from shared.utils.ids import new_credential_ref
from shared.utils.redact import REDACTED, redact_credential_fields

from .parser import ParsedWorkflow


@dataclass(frozen=True)
class TaskCredentials:
    # Each vaulted credential's pointer in the spec, mapped to its ref.
    refs: dict[str, str] = field(default_factory=dict)
    # Set when a vaulted value renders from an upstream stage: its ref names the
    # unrendered value, so it cannot tell two rendered credentials apart.
    renders: bool = False

    def merge_key(self, spec: TaskSpecBase, **context: Any) -> str | None:
        """The spec's merge key, or ``None`` when a credential renders at dispatch."""
        if self.renders:
            return None
        return credential_merge_key(spec, self.refs, **context)


@dataclass(frozen=True)
class InlineCredentials:
    # What the vault stores, keyed by ref; one ref per distinct value in the submission.
    values: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, TaskCredentials] = field(default_factory=dict)
    # Each agent's model key ref by task id; the key reaches the model gateway through
    # its binding's ``secret_ref`` and is never restored into the spec.
    model_keys: dict[str, str] = field(default_factory=dict)


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
    spec: TaskSpecTemplateBase, refs: CredentialRefs
) -> TaskCredentials:
    """Mask every inline credential in ``spec`` in place, minting a ref for each from
    ``refs``."""
    found = find_spec_credentials(spec)
    taken = TaskCredentials(
        refs={pointer: refs.ref(value) for pointer, value in found.items()},
        renders=any(holds_placeholder(value) for value in found.values()),
    )
    mask_spec_values(spec, found)
    return taken


def _take_model_key(spec: TaskSpecTemplateBase) -> SecretStr | None:
    if not isinstance(spec, AgentSpecTemplate) or spec.model_binding is None:
        return None
    key, spec.model_binding.api_key = spec.model_binding.api_key, None
    return key


def take_inline_credentials(parsed: ParsedWorkflow) -> InlineCredentials:
    """Take every inline credential out of ``parsed`` in place and return them.

    A task-spec credential is replaced by its marker where it sits and an agent's model
    key is removed from its binding, so no value reaches a persisted record or the
    compiled template and plan.
    """
    refs = CredentialRefs()
    tasks: dict[str, TaskCredentials] = {}
    model_keys: dict[str, str] = {}
    for task in parsed.tasks:
        spec = task.task.spec
        if (key := _take_model_key(spec)) is not None:
            model_keys[task.task_id] = refs.ref(key.get_secret_value())
        taken = tasks[task.task_id] = take_spec_credentials(spec, refs)
        task.masked_credentials = frozenset(taken.refs)
    return InlineCredentials(values=refs.values, tasks=tasks, model_keys=model_keys)


def mask_inline_credentials(parsed: ParsedWorkflow) -> None:
    """Take every inline credential out of ``parsed`` in place, vaulting nothing."""
    take_inline_credentials(parsed)


def credential_merge_key(
    spec: TaskSpecBase, refs: Mapping[str, str], **context: Any
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
