"""Pluggable web-search backends behind a narrow provider interface.

A provider performs one bounded, read-only query and returns ranked results or raises a
typed fault; the ``FabricToolBroker`` normalizes both into a typed ``ToolOutcome``. The
default ``duckduckgo`` provider is keyless; keyed providers take a deployment-env key.
No provider fetches result pages (snippets only), keeping the SSRF surface minimal.
"""

import html
import re
from typing import TYPE_CHECKING, Protocol
from urllib.parse import unquote

import requests
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ..config import WebSearchConfig

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124 Safari/537.36"
)
_RESULT = re.compile(
    r'result__a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
    r"result__snippet[^>]*>(?P<snippet>.*?)</a>",
    re.DOTALL,
)
_TAGS = re.compile(r"<[^>]+>")
_UDDG = re.compile(r"uddg=([^&]+)")


class SearchResult(BaseModel):
    """One ranked web-search result: a citation and its snippet."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str


class SearchTimeout(RuntimeError):
    """The provider did not answer within the bounded timeout."""


class SearchQuotaExceeded(RuntimeError):
    """The provider refused the query for quota/rate reasons."""


class SearchUnavailable(RuntimeError):
    """The provider was unreachable or returned an unusable response."""


class SearchProvider(Protocol):
    """A read-only web-search backend the broker calls off the agent's lane."""

    def search(
        self, query: str, *, max_results: int, timeout_sec: float
    ) -> list[SearchResult]: ...


def _clean(text: str) -> str:
    return html.unescape(_TAGS.sub("", text)).strip()


def _unwrap(url: str) -> str:
    return unquote(m.group(1)) if (m := _UDDG.search(url)) else url


class DuckDuckGoProvider:
    """Keyless web search over DuckDuckGo's HTML endpoint (snippets only)."""

    def search(
        self, query: str, *, max_results: int, timeout_sec: float
    ) -> list[SearchResult]:
        try:
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": _UA},
                timeout=timeout_sec,
            )
        except requests.Timeout as exc:
            raise SearchTimeout(str(exc)) from exc
        except requests.RequestException as exc:
            raise SearchUnavailable(str(exc)) from exc
        if resp.status_code == 429:
            raise SearchQuotaExceeded(f"provider returned {resp.status_code}")
        if resp.status_code >= 400:
            raise SearchUnavailable(f"provider returned {resp.status_code}")
        results: list[SearchResult] = []
        for i, m in enumerate(_RESULT.finditer(resp.text)):
            if i >= max_results:
                break
            results.append(
                SearchResult(
                    title=_clean(m.group("title")),
                    url=_unwrap(m.group("url")),
                    snippet=_clean(m.group("snippet")),
                )
            )
        return results


class SerperProvider:
    """Web search over the Serper google.serper.dev API (needs a deployment key)."""

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    def search(
        self, query: str, *, max_results: int, timeout_sec: float
    ) -> list[SearchResult]:
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": self._key, "Content-Type": "application/json"},
                timeout=timeout_sec,
            )
        except requests.Timeout as exc:
            raise SearchTimeout(str(exc)) from exc
        except requests.RequestException as exc:
            raise SearchUnavailable(str(exc)) from exc
        if resp.status_code in (401, 403, 429):
            raise SearchQuotaExceeded(f"provider returned {resp.status_code}")
        if resp.status_code >= 400:
            raise SearchUnavailable(f"provider returned {resp.status_code}")
        try:
            organic = resp.json().get("organic", [])
        except ValueError as exc:
            raise SearchUnavailable(str(exc)) from exc
        results: list[SearchResult] = []
        for hit in organic[:max_results]:
            if isinstance(hit, dict):
                results.append(
                    SearchResult(
                        title=str(hit.get("title", "")),
                        url=str(hit.get("link", "")),
                        snippet=str(hit.get("snippet", "")),
                    )
                )
        return results


def build_search_provider(config: "WebSearchConfig") -> SearchProvider:
    """The configured provider; a keyed provider needs ``WEB_SEARCH_API_KEY``."""
    if config.provider == "duckduckgo":
        return DuckDuckGoProvider()
    if config.provider == "serper":
        if not config.api_key:
            raise ValueError("the serper web-search provider needs WEB_SEARCH_API_KEY")
        return SerperProvider(config.api_key)
    raise ValueError(f"unknown web-search provider {config.provider!r}")
