from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

if TYPE_CHECKING:
    from sylvia_core.utils.schemas import VisionSemanticMemory


class LLMBackend(ABC):
    """Defines an interface for interacting with various LLM providers using Langchain."""

    name: str = "base"

    @abstractmethod
    def get_chat_model(self, model_name: str, temperature: float) -> BaseChatModel:
        """
        Returns a configured Langchain ChatModel for general chat completions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_structured_model(
        self, model_name: str, temperature: float, output_schema: Type[BaseModel], method: str = "function_calling"
    ) -> BaseChatModel:
        """
        Returns a configured Langchain ChatModel for structured output.
        The `method` parameter can be "function_calling" or "json_mode".
        """
        raise NotImplementedError

    @abstractmethod
    async def invoke_vision(
        self,
        prompt: str,
        image_url: str,
        system_prompt: str | None,
        model_name: str,
        temperature: float,
        max_tokens: int | None,
    ) -> VisionSemanticMemory:
        """
        Invokes the LLM for vision tasks (e.g., image description) and returns the parsed VisionSemanticMemory object.
        """
        raise NotImplementedError
