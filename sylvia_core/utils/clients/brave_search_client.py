import requests
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BraveSearchClient:
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("Brave Search API key is required for BraveSearchClient.")
        self.api_key = api_key
        self.base_headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }

    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = requests.get(url, params=params, headers=self.base_headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning("Brave Search API request failed: %s", e)
            return {"error": str(e)}

    def web_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs a web search using the Brave Search API.

        Args:
            q: The search query.
            count: The number of results to return (max 20).
            offset: The offset for pagination.
            freshness: Filters search results by when they were discovered.
                       e.g., 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year),
                       or 'YYYY-MM-DDtoYYYY-MM-DD' for a date range.

        Returns:
            A dictionary containing the search results.
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": q, "count": count, "offset": offset}
        if freshness:
            params["freshness"] = freshness
        return self._make_request(url, params)

    def news_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs a news search using the Brave Search API.

        Args:
            q: The search query.
            count: The number of results to return (max 20).
            offset: The offset for pagination.
            freshness: Filters search results by when they were discovered.
                       e.g., 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year),
                       or 'YYYY-MM-DDtoYYYY-MM-DD' for a date range.

        Returns:
            A dictionary containing the news search results.
        """
        url = "https://api.search.brave.com/res/v1/news/search"
        params = {"q": q, "count": count, "offset": offset}
        if freshness:
            params["freshness"] = freshness
        return self._make_request(url, params)
