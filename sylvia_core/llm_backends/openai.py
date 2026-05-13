from __future__ import annotations

import logging
from typing import Any, Type

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from .base import LLMBackend
from sylvia_core.config import settings
from sylvia_core.utils.schemas import VisionSemanticMemory

logger = logging.getLogger(__name__)


class OpenAILLMBackend(LLMBackend):
    """OpenAI LLM backend using Langchain's ChatOpenAI."""

    name: str = "openai"

    def get_chat_model(self, model_name: str, temperature: float) -> BaseChatModel:
        """
        Returns a configured ChatOpenAI instance for general chat completions.
        """
        return ChatOpenAI(model=model_name, temperature=temperature)

    def get_structured_model(
        self, model_name: str, temperature: float, output_schema: Type[BaseModel], method: str = "function_calling"
    ) -> BaseChatModel:
        """
        Returns a configured ChatOpenAI instance for structured output.
        """
        return ChatOpenAI(model=model_name, temperature=temperature).with_structured_output(
            output_schema, method=method
        )

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
        Invokes the ChatOpenAI for vision tasks (e.g., image description) and returns the parsed VisionSemanticMemory object.
        """
        llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ]
            )
        )
        
        raw_response_content = await llm.ainvoke(messages)
        

        cleaned_output = str(raw_response_content.content).strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json"):].strip()
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-len("```")].strip()
        
        try:
            vision_memory = VisionSemanticMemory.model_validate_json(cleaned_output)
            return vision_memory
        except ValidationError as e:
            logger.exception("Failed to parse VisionSemanticMemory from LLM output: %s", e)
            logger.debug("Raw LLM output: %s", raw_response_content.content)
            raise
