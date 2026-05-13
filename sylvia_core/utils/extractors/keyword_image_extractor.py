import logging
import time
from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from ..prompts import prompt_manager
from ..schemas import Keyword, KeywordsList

logger = logging.getLogger(__name__)


class ImageKeywordExtractor:
    def __init__(self, structured_llm: Runnable):
        self.structured_llm = structured_llm
        self.prompt_text = prompt_manager.get("keyword_image_extraction_prompt")

    async def extract(
        self,
        query: str,
        image_base64: str,
        formatted_history: str = "",
    ) -> List[Keyword]:
        """Extract keywords from image input and optional text context."""
        try:
            logger.debug("Starting image keyword extraction.")
            t_start = time.time()

            prompt = self.prompt_text.format(formatted_history=formatted_history, query=query)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    },
                ]
            )

            result: KeywordsList = await self.structured_llm.ainvoke([message])
            final_keywords = result.keywords

            logger.debug(
                "Image keyword extraction completed in %.4fs. Keywords: %s",
                time.time() - t_start,
                [k.keyword for k in final_keywords],
            )
            return final_keywords

        except Exception as e:
            logger.error("Error during structured image keyword extraction: %s", e)
            return [Keyword(keyword=q) for q in query.split() if len(q) > 2]
