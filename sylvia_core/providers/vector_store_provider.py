from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorStoreProvider(ABC):
    @abstractmethod
    async def create_collection(self, collection_name: str, vector_size: int):
        """
        Creates a new vector collection if it doesn't exist.
        """
        pass

    @abstractmethod
    async def upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings: Any, # This will be an embedding provider instance
        batch_size: int = 50
    ):
        """
        Embeds documents and upserts them to the specified collection in batches.
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        keywords: List[str],
        limit: int = 5,
        query_text: Optional[str] = None # For potential text search alongside vector search
    ) -> List[Dict[str, Any]]:
        """
        Performs a vector search on the specified collection,
        combining vector search with keyword filtering.
        Returns a list of retrieved documents (payloads).
        """
        pass

    @abstractmethod
    async def get_client(self) -> Any:
        """
        Returns the underlying vector store client.
        """
        pass

    @abstractmethod
    async def aclose(self):
        """
        Closes any open resources or connections held by the vector store provider.
        """
        pass
