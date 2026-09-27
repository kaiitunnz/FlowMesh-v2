import json

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


def redact_source_text(raw_payload: str, format: str) -> str:
    """Return the submitted source with every credential-keyed value masked.

    Redaction is structural: the payload is parsed with its submission format's parser,
    every credential-looking field is masked, and the document is re-serialized in that
    format, so a credential never survives in the captured source however it was quoted
    or escaped. A payload with no credential field is returned unchanged, and one that
    does not parse is replaced by a marker rather than kept verbatim.
    """
    try:
        doc = (
            json.loads(raw_payload) if format == "n8n" else yaml.safe_load(raw_payload)
        )
    except (json.JSONDecodeError, yaml.YAMLError):
        return REDACTED
    redacted = redact_credential_fields(doc)
    if redacted == doc:
        return raw_payload
    if format == "n8n":
        return json.dumps(redacted, indent=2)
    return yaml.safe_dump(redacted, sort_keys=False)
