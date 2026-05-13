from .base import LLMBackend
from .openai import OpenAILLMBackend
from sylvia_core.config import settings

def get_llm_backend() -> LLMBackend:
    """
    Returns an initialized LLMBackend instance based on configuration.
    """
    provider_type = settings.LLM_PROVIDER_TYPE.lower()
    if provider_type == "openai":
        return OpenAILLMBackend()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER_TYPE: {provider_type}")
