from abc import ABC, abstractmethod
from typing import Any, Optional


class SearchProvider(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def web_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def news_search(
        self,
        q: str,
        count: int = 20,
        offset: int = 0,
        freshness: Optional[str] = None,
    ) -> dict[str, Any]:
        pass
