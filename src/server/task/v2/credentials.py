import json
from typing import Any

import yaml
from pydantic import SecretStr

from shared.tasks.specs import AgentSpecStrict, AgentSpecTemplate
from shared.utils.redact import REDACTED, redact_credential_fields

from ..parser import ParsedWorkflow


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
