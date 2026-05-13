from typing import Any, Optional

from ..utils.clients.brave_search_client import BraveSearchClient
from .search_provider import SearchProvider


class BraveSearchProvider(SearchProvider):
    provider_name = "brave"

    def __init__(self, api_key: str | None):
        self._client: BraveSearchClient | None = None
        self._error: str | None = None

        if not api_key:
            self._error = "Brave Search API key not configured."
            return

        try:
            self._client = BraveSearchClient(api_key)
        except ValueError as e:
            self._error = str(e)

    def is_available(self) -> bool:
        return self._client is not None

    def web_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self._client:
            return {"error": self._error or "Brave Search provider unavailable."}
        return self._client.web_search(q=q, count=count, offset=offset, freshness=freshness)

    def news_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self._client:
            return {"error": self._error or "Brave Search provider unavailable."}
        return self._client.news_search(q=q, count=count, offset=offset, freshness=freshness)
