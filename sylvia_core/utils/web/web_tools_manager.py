import logging
from typing import Any, Dict, Optional

from ...providers.search_provider import SearchProvider

logger = logging.getLogger(__name__)


class WebToolsManager:
    MIN_SEARCH_COUNT = 5

    def __init__(self, search_provider: SearchProvider):
        self.search_provider = search_provider

        if self.search_provider.is_available():
            logger.info("Search provider initialized: %s", self.search_provider.provider_name)
        else:
            logger.warning("Search provider is unavailable: %s", self.search_provider.provider_name)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return

    def web_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_count = max(self.MIN_SEARCH_COUNT, count)
        return self.search_provider.web_search(
            q=q,
            count=safe_count,
            offset=offset,
            freshness=freshness,
        )

    def news_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_count = max(self.MIN_SEARCH_COUNT, count)
        return self.search_provider.news_search(
            q=q,
            count=safe_count,
            offset=offset,
            freshness=freshness,
        )
