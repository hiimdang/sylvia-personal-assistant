from typing import Any, Optional

import requests

from .search_provider import SearchProvider


class GoogleSearchProvider(SearchProvider):
    provider_name = "google"
    _endpoint = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str | None, cse_id: str | None):
        self._api_key = api_key
        self._cse_id = cse_id

    def is_available(self) -> bool:
        return bool(self._api_key and self._cse_id)

    def _base_error(self) -> dict[str, Any]:
        return {"error": "Google Search provider not configured. Missing API key or CSE id."}

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self.is_available():
            return self._base_error()

        enriched = {
            "key": self._api_key,
            "cx": self._cse_id,
            **params,
        }

        try:
            response = requests.get(self._endpoint, params=enriched, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def web_search(
        self,
        q: str,
        count: int = 10,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> dict[str, Any]:
        num = max(5, min(count, 10))
        start = max(1, offset + 1)

        params: dict[str, Any] = {
            "q": q,
            "num": num,
            "start": start,
        }
        if freshness:
            params["dateRestrict"] = freshness

        return self._request(params)

    def news_search(
        self,
        q: str,
        count: int = 10,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> dict[str, Any]:
        # Google Custom Search has no dedicated "news" endpoint.
        # We bias results toward news sources via query enrichment.
        news_query = f"{q} news"
        return self.web_search(q=news_query, count=count, offset=offset, freshness=freshness)
