import asyncio
import logging
from typing import Any, List, Dict, Optional

from langchain_core.documents import Document

from sylvia_core.utils.formatters import format_chat_history, format_retrieved_context

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, keyword_extractor, image_keyword_extractor, image_description_extractor, chat_retriever):
        self.keyword_extractor = keyword_extractor
        self.image_keyword_extractor = image_keyword_extractor
        self.image_description_extractor = image_description_extractor
        self.chat_retriever = chat_retriever

    async def get_keywords(
        self,
        prompt: str,
        chat_history: List[Dict] | None = None,
        image_base64: str = None,
    ) -> List[str]:
        """Extracts keywords from the user's prompt and image."""
        formatted_history = format_chat_history(chat_history)
        if image_base64:
            return [
                k.keyword
                for k in await self.image_keyword_extractor.extract(
                    prompt, image_base64, formatted_history
                )
            ]
        else:
            return [
                k.keyword
                for k in await self.keyword_extractor.extract(prompt, formatted_history)
            ]

    async def search(
        self,
        prompt: str,
        chat_history: List[Dict] | None = None,
        image_base64: str = None,
    ) -> List[Document]:
        """Performs a search in the vector database.

        Embedding and keyword extraction run in parallel since both only depend
        on the prompt string and are fully independent of each other.
        This saves ~200-400ms per request compared to sequential execution.
        """
        keywords_task = self.get_keywords(
            prompt=prompt, image_base64=image_base64, chat_history=chat_history
        )
        embedding_task = self.chat_retriever.embedding_provider.embed_query(prompt)

        keywords, query_embedding = await asyncio.gather(keywords_task, embedding_task)

        logger.debug("Keyword extraction completed: %d keywords.", len(keywords))

        return await self.chat_retriever.search_with_embedding(
            query=prompt,
            query_embedding=query_embedding,
            keywords=keywords if keywords else [],
        )

    async def get_image_summary(self, image_url: str) -> Any:
        """Gets a description and object list from an image."""
        try:
            vision_memory = await self.image_description_extractor.extract(image_url)
            return vision_memory
        except Exception as e:
            logger.error("Image description failed: %s", e)
            return "Unable to describe this image."

    def format_retrieved_context(self, retrieved_docs: List[Document]) -> str:
        """Formats the retrieved documents into a string for the prompt."""
        return format_retrieved_context(retrieved_docs)

