from sentence_transformers import CrossEncoder
from .llm_backends.factory import get_llm_backend
from .providers.fastembed_embedding import FastEmbedEmbeddingProvider
from .providers.qdrant_vector_store import QdrantVectorStoreProvider
from .providers.embedding_provider import BaseEmbeddingProvider
from .providers.vector_store_provider import BaseVectorStoreProvider
from .providers.search_provider import SearchProvider
from .providers.brave_search_provider import BraveSearchProvider
from .providers.google_search_provider import GoogleSearchProvider
from .utils.schemas import KeywordsList, VisionSemanticMemory


from .config import settings

class SylviaClientManager:
    def __init__(self):
        self.llm_backend = get_llm_backend()
        self.main_llm = self.llm_backend.get_chat_model(
            model_name=settings.DEFAULT_MODEL, temperature=settings.TEMPERATURE
        )
        self.embeddings = get_embedding_provider()
        self.vector_store_provider = get_vector_store_provider()
        self.search_provider = get_search_provider()
        self.reranker_model = get_reranker_model()


        self.structured_keyword_llm = self.llm_backend.get_structured_model(
            model_name=settings.KEYWORD_EXTRACTION_MODEL, temperature=0, output_schema=KeywordsList
        )
        self.structured_image_keyword_llm = self.llm_backend.get_structured_model(
            model_name=settings.IMAGE_KEYWORD_EXTRACTION_MODEL, temperature=0, output_schema=KeywordsList
        )
        self.structured_image_description_llm = self.llm_backend.get_structured_model(
            model_name=settings.IMAGE_DESCRIPTION_MODEL, temperature=0.3, output_schema=VisionSemanticMemory, method="function_calling"
        )


def get_main_llm():
    """Initializes and returns the main LLM client based on configuration."""
    llm_backend = get_llm_backend()
    return llm_backend.get_chat_model(
        model_name=settings.DEFAULT_MODEL, temperature=settings.TEMPERATURE
    )

def get_reranker_model():
    """Initializes and returns the reranker model."""
    return CrossEncoder(settings.RERANKER_MODEL)

def get_embedding_provider() -> BaseEmbeddingProvider:
    """Initializes and returns the embedding provider instance based on configuration."""
    if settings.EMBEDDING_PROVIDER_TYPE == "fastembed":
        return FastEmbedEmbeddingProvider()
    else:
        raise ValueError(f"Unsupported Embedding provider type: {settings.EMBEDDING_PROVIDER_TYPE}. Currently only 'fastembed' is supported.")

def get_vector_store_provider() -> BaseVectorStoreProvider:
    """Initializes and returns the vector store provider instance based on configuration."""
    if settings.VECTOR_STORE_PROVIDER_TYPE == "qdrant":
        return QdrantVectorStoreProvider(
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY
        )
    else:
        raise ValueError(f"Unsupported Vector Store provider type: {settings.VECTOR_STORE_PROVIDER_TYPE}. Currently only 'qdrant' is supported.")

def get_search_provider() -> SearchProvider:
    provider_type = settings.SEARCH_PROVIDER_TYPE.lower().strip()

    if provider_type == "none":
        class DisabledSearchProvider(SearchProvider):
            provider_name = "none"

            def is_available(self) -> bool:
                return False

            def web_search(self, q: str, count: int = 20, offset: int = 0, freshness: str | None = None) -> dict:
                return {"error": "Search provider is disabled."}

            def news_search(self, q: str, count: int = 20, offset: int = 0, freshness: str | None = None) -> dict:
                return {"error": "Search provider is disabled."}

        return DisabledSearchProvider()

    if provider_type == "brave":
        return BraveSearchProvider(api_key=settings.BRAVE_SEARCH_API_KEY)

    if provider_type == "google":
        return GoogleSearchProvider(
            api_key=settings.GOOGLE_API_KEY,
            cse_id=settings.GOOGLE_CSE_ID,
        )

    raise ValueError(
        f"Unsupported Search provider type: {settings.SEARCH_PROVIDER_TYPE}. "
        "Supported values are: brave, google, none."
    )


