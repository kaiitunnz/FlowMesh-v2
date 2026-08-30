import json
from typing import Any

import yaml
from pydantic import SecretStr

from shared.tasks.specs import AgentSpecStrict, AgentSpecTemplate

from ..parser import ParsedWorkflow

_REDACTED = "***redacted***"


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


def redact_source_text(raw_payload: str, format: str) -> str:
    """Return the submitted source with every inline ``api_key`` masked.

    Redaction is structural: the payload is parsed, each ``api_key`` field is masked,
    and the document is re-serialized, so an inline ``api_key`` never survives in the
    captured source regardless of how it was quoted or escaped. A payload with no
    ``api_key`` (or one that does not parse) is returned unchanged.
    """
    try:
        doc = yaml.safe_load(raw_payload)
    except yaml.YAMLError:
        return raw_payload
    if not _mask_api_keys(doc):
        return raw_payload
    if format == "json":
        return json.dumps(doc, indent=2)
    return yaml.safe_dump(doc, sort_keys=False)


def _mask_api_keys(value: Any) -> bool:
    """Mask every ``api_key`` field in a nested structure, in place.

    Returns whether any masking occurred so an unaffected payload keeps its original
    formatting.
    """
    found = False
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "api_key" and nested is not None:
                value[key] = _REDACTED
                found = True
            elif _mask_api_keys(nested):
                found = True
    elif isinstance(value, list):
        for item in value:
            if _mask_api_keys(item):
                found = True
    return found
