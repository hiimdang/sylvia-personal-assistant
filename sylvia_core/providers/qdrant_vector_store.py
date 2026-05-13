import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

import qdrant_client
from langchain_core.documents import Document
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    PointStruct,
    VectorParams,
)

from .embedding_provider import BaseEmbeddingProvider
from .vector_store_provider import BaseVectorStoreProvider

logger = logging.getLogger(__name__)


class QdrantVectorStoreProvider(BaseVectorStoreProvider):
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        self._qdrant_client = qdrant_client.AsyncQdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            check_compatibility=False,
        )

    async def create_collection(self, collection_name: str, vector_size: int):
        """Create a collection if it does not exist."""
        try:
            await self._qdrant_client.get_collection(collection_name=collection_name)
            logger.info("Using existing collection: '%s'", collection_name)
        except Exception:
            logger.info("Collection '%s' not found. Creating it...", collection_name)
            await self._qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Collection created.")

            try:
                await self._qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.question",
                    field_schema={
                        "type": "text",
                        "tokenizer": "word",
                        "min_token_len": 2,
                        "max_token_len": 10,
                        "lowercase": True,
                    },
                    wait=True,
                )
                logger.info("Created payload index for 'metadata.question'.")
            except Exception as e:
                logger.warning("Could not create payload index for 'metadata.question': %s", e)

            try:
                await self._qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.answer",
                    field_schema={
                        "type": "text",
                        "tokenizer": "word",
                        "min_token_len": 2,
                        "max_token_len": 10,
                        "lowercase": True,
                    },
                    wait=True,
                )
                logger.info("Created payload index for 'metadata.answer'.")
            except Exception as e:
                logger.warning("Could not create payload index for 'metadata.answer': %s", e)

    async def upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embedding_provider: BaseEmbeddingProvider,
        batch_size: int = 50,
    ):
        """Embed documents and upsert them to Qdrant in batches."""
        logger.info("Upserting %d documents into collection '%s'...", len(documents), collection_name)

        _ = embedding_provider.get_embedding_model()

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            logger.info("Processing batch %d/%d (%d docs).", batch_num, total_batches, len(batch))

            texts_to_embed = [doc["page_content"] for doc in batch]
            vectors = await embedding_provider.embed_documents(texts_to_embed)

            points_to_upsert = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=doc,
                )
                for doc, vector in zip(batch, vectors)
            ]

            await self._qdrant_client.upsert(
                collection_name=collection_name,
                points=points_to_upsert,
                wait=True,
            )

        logger.info("Qdrant upsert completed.")

    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        keywords: List[str],
        limit: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Document]:
        """Perform hybrid search and return unique documents."""
        qdrant_filter = None
        if keywords:
            qdrant_filter = Filter(
                should=[
                    condition
                    for keyword in keywords
                    for condition in (
                        FieldCondition(
                            key="metadata.question",
                            match=MatchText(text=keyword),
                        ),
                        FieldCondition(
                            key="metadata.answer",
                            match=MatchText(text=keyword),
                        ),
                    )
                ]
            )

        async def _search_with_filter():
            logger.debug("Running filtered search (keywords + vector).")
            return await self._qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=10,
                with_payload=True,
            )

        async def _search_vector_only():
            logger.debug("Running vector-only search.")
            return await self._qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=None,
                limit=max(limit, 5),
                with_payload=True,
            )

        filter_result, vector_result = await asyncio.gather(
            _search_with_filter(),
            _search_vector_only(),
        )

        unique_documents: Dict[Any, Document] = {}
        for hit in filter_result + vector_result:
            if hit.id not in unique_documents:
                unique_documents[hit.id] = Document(
                    page_content=hit.payload.get("page_content", ""),
                    metadata=hit.payload.get("metadata", {}),
                )

        return list(unique_documents.values())[:limit]

    def get_client(self) -> qdrant_client.AsyncQdrantClient:
        """Return the underlying Qdrant client."""
        return self._qdrant_client

    async def aclose(self):
        """Close the underlying Qdrant client."""
        if self._qdrant_client:
            logger.info("Closing QdrantVectorStoreProvider client...")
            await self._qdrant_client.close()
            logger.info("QdrantVectorStoreProvider client closed.")
