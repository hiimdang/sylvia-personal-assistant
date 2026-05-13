from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ENV_FILE_PATH = Path(__file__).parent.parent / ".env"
if ENV_FILE_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE_PATH)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        extra='ignore',
        env_prefix='SYLVIA_',
    )

    DEFAULT_MODEL: str = "gpt-4.1-mini"
    IMAGE_DESCRIPTION_MODEL: str = "gpt-4.1-mini"
    KEYWORD_EXTRACTION_MODEL: str = "gpt-4.1-nano"
    IMAGE_KEYWORD_EXTRACTION_MODEL: str = "gpt-4.1-nano"
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIM: int = 384
    COLLECTION_NAME: str = "sylvia_dataset"
    BATCH_SIZE: int = 50
    ENABLE_INTERACTION_LOG: bool = False
    LOG_FILE: str | None = None
    TEMPERATURE: float = 0.3
    MEMORY_WINDOW_SIZE: int = 4

    LLM_PROVIDER_TYPE: str = "openai" # "openai"
    EMBEDDING_PROVIDER_TYPE: str = "fastembed" # "fastembed"
    VECTOR_STORE_PROVIDER_TYPE: str = "qdrant" # "qdrant"
    SEARCH_PROVIDER_TYPE: str = "brave"  # brave | google | none

    QDRANT_URL: str
    QDRANT_API_KEY: str

    BRAVE_SEARCH_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GOOGLE_CSE_ID: str | None = None

settings = Settings()
