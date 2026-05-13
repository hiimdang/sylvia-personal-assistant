import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..prompts import prompt_manager
from ..schemas import VisionSemanticMemory

logger = logging.getLogger(__name__)


class ImageDescriptionExtractor:
    def __init__(self, structured_llm: BaseChatModel):
        self.structured_llm = structured_llm
        self.system_prompt = prompt_manager.get("describe_image_prompt")

    async def extract(self, image_url: str) -> VisionSemanticMemory:
        """Extract a structured description from an image URL."""
        try:
            logger.debug("Starting image description with structured LLM.")
            t_start = time.time()

            message_content = [
                {"type": "text", "text": self.system_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]

            vision_memory: VisionSemanticMemory = await self.structured_llm.ainvoke(
                [HumanMessage(content=message_content)]
            )

            logger.debug("Image description completed in %.4fs.", time.time() - t_start)
            return vision_memory

        except Exception as e:
            logger.error("Error during image description: %s", e)
            return VisionSemanticMemory(
                global_summary="Unable to describe this image.",
                entities=[],
                attributes={},
                perception_notes=["Image processing error."],
            )
