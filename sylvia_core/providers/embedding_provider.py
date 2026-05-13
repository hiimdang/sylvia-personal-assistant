from abc import ABC, abstractmethod
from typing import Any, List

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding_model(self) -> Any:
        """
        Returns an initialized embedding model (e.g., TextEmbedding).
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        """
        pass
