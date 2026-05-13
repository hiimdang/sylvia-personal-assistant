import asyncio
import logging
from typing import List

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from sylvia_core.config import settings

from .embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class FastEmbedEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        self._embedding_model = None

    def get_embedding_model(self) -> TextEmbedding:
        """Initialize and return the TextEmbedding model lazily."""
        if self._embedding_model is None:
            logger.info("Registering FastEmbed model: %s", settings.EMBEDDING_MODEL)
            TextEmbedding.add_custom_model(
                model=settings.EMBEDDING_MODEL,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=settings.EMBEDDING_MODEL),
                dim=settings.EMBEDDING_DIM,
            )
            logger.info("Creating FastEmbed model: %s", settings.EMBEDDING_MODEL)
            self._embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
            logger.info("FastEmbed model initialized.")
        return self._embedding_model

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents asynchronously."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.get_embedding_model().embed, texts)
        return list(result)

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query asynchronously."""
        loop = asyncio.get_running_loop()
        embeddings_iterator = await loop.run_in_executor(None, self.get_embedding_model().embed, text)
        return list(embeddings_iterator)[0]
