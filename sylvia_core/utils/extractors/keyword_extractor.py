import logging
import time
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from ..prompts import prompt_manager
from ..schemas import Keyword, KeywordsList

logger = logging.getLogger(__name__)


class KeywordExtractor:
    def __init__(self, structured_llm: Runnable):
        self.structured_llm = structured_llm
        prompt_text = prompt_manager.get("keyword_extraction_prompt")
        self.keyword_prompt_template = PromptTemplate.from_template(prompt_text)

    async def extract(self, query: str, formatted_history: str) -> List[Keyword]:
        """Extract keywords from text query and chat history."""
        try:
            logger.debug("Starting text keyword extraction.")
            t_start = time.time()

            chain = self.keyword_prompt_template | self.structured_llm
            result: KeywordsList = await chain.ainvoke(
                {
                    "formatted_history": formatted_history,
                    "query": query,
                }
            )

            final_keywords = result.keywords
            logger.debug(
                "Text keyword extraction completed in %.4fs. Keywords: %s",
                time.time() - t_start,
                [k.keyword for k in final_keywords],
            )
            return final_keywords

        except Exception as e:
            logger.error("Error during structured keyword extraction: %s", e)
            return [Keyword(keyword=q) for q in query.split() if len(q) > 2]
