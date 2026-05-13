import asyncio
import logging
import time
from typing import List

from langchain_core.documents import Document

from sylvia_core.providers.embedding_provider import BaseEmbeddingProvider
from sylvia_core.providers.vector_store_provider import BaseVectorStoreProvider

logger = logging.getLogger(__name__)


class ChatRetriever:
    def __init__(
        self,
        vector_store_provider: BaseVectorStoreProvider,
        embedding_provider: BaseEmbeddingProvider,
        reranker_model,
        collection_name,
    ):
        self.vector_store_provider = vector_store_provider
        self.embedding_provider = embedding_provider
        self.reranker_model = reranker_model
        self.collection_name = collection_name

    async def _rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_n: int = 3,
    ) -> list[Document]:
        """Rerank documents by relevance using CrossEncoder."""
        rerank_start_time = time.time()
        logger.debug("Starting document reranking.")
        if not documents:
            logger.debug("No documents to rerank.")
            return []

        sentence_pairs = [[query, doc.page_content] for doc in documents]
        scores = await asyncio.to_thread(self.reranker_model.predict, sentence_pairs)
        doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Top %d results after reranking:", top_n)
            for i, (doc, score) in enumerate(doc_scores[:top_n]):
                logger.debug(
                    "  Rank %d (Score: %.4f): Q=%s",
                    i + 1,
                    score,
                    doc.metadata.get("question", "N/A")[:60],
                )

        logger.debug(
            "Reranking completed in %.4fs.",
            time.time() - rerank_start_time,
        )
        return [doc for doc, _score in doc_scores[:top_n]]

    async def search(self, query: str, keywords: list[str]) -> List[Document]:
        """Embed query then run retrieval and reranking."""
        retrieval_start_time = time.time()
        logger.debug("Starting retrieval.")

        embedding_start_time = time.time()
        try:
            logger.debug("Creating query embedding.")
            query_embedding = await self.embedding_provider.embed_query(query)
            logger.debug("Embedding created in %.4fs.", time.time() - embedding_start_time)
        except Exception as e:
            logger.error("Failed to create query embedding: %s", e)
            return []

        return await self._search_and_rerank(query, query_embedding, keywords, retrieval_start_time)

    async def search_with_embedding(
        self,
        query: str,
        query_embedding: List[float],
        keywords: list[str],
    ) -> List[Document]:
        """Run retrieval and reranking with a precomputed embedding."""
        retrieval_start_time = time.time()
        logger.debug("Starting retrieval with precomputed embedding.")
        return await self._search_and_rerank(query, query_embedding, keywords, retrieval_start_time)

    async def _search_and_rerank(
        self,
        query: str,
        query_embedding: List[float],
        keywords: list[str],
        start_time: float,
    ) -> List[Document]:
        """Shared retrieval + rerank pipeline."""
        retrieved_payloads = await self.vector_store_provider.search(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            keywords=keywords,
            limit=10,
            query_text=query,
        )

        retrieved_documents = retrieved_payloads
        logger.debug("Retrieved %d documents before reranking.", len(retrieved_documents))

        reranked_documents = await self._rerank_documents(query, retrieved_documents, top_n=3)
        logger.debug("Total retrieval pipeline took %.4fs.", time.time() - start_time)
        return reranked_documents
