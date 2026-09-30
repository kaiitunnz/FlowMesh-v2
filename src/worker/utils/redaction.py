import re
from urllib.parse import urlsplit, urlunsplit

from shared.utils.redact import redact_url

_ABSOLUTE_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")


def redact_urls(text: str, *urls: str | None) -> str:
    """``text`` with each of ``urls`` and every absolute URL in it redacted.

    A client error may quote a request URL whole or only its path and query, so each
    known URL is replaced in both forms.
    """
    for url in urls:
        if not url or (redacted := redact_url(url)) == url:
            continue
        text = text.replace(url, redacted)
        parts, masked = urlsplit(url), urlsplit(redacted)
        target = urlunsplit(("", "", parts.path, parts.query, ""))
        if parts.query and target != (
            safe := urlunsplit(("", "", masked.path, masked.query, ""))
        ):
            text = text.replace(target, safe)
    return _ABSOLUTE_URL.sub(lambda match: redact_url(match.group(0)), text)
