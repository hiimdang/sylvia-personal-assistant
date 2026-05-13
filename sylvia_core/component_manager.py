from sylvia_core.utils.extractors.keyword_extractor import KeywordExtractor
from sylvia_core.utils.extractors.keyword_image_extractor import ImageKeywordExtractor
from sylvia_core.utils.retrieval import ChatRetriever
from sylvia_core.utils.extractors.describe_image import ImageDescriptionExtractor
from sylvia_core.utils.web.web_tools_manager import WebToolsManager
from sylvia_core.utils.web.search_tools import (
    get_web_search_tool,
    get_news_search_tool,
)

class SylviaComponentManager:
    def __init__(
        self,
        embeddings,
        reranker_model,
        vector_store_provider,
        structured_keyword_llm,
        structured_image_keyword_llm,
        structured_image_description_llm,
        collection_name: str,
        web_tools_manager: WebToolsManager,
    ):

        self.keyword_extractor = KeywordExtractor(structured_llm=structured_keyword_llm)
        self.image_keyword_extractor = ImageKeywordExtractor(
            structured_llm=structured_image_keyword_llm
        )
        self.image_description_extractor = ImageDescriptionExtractor(
            structured_llm=structured_image_description_llm
        )
        self.chat_retriever = ChatRetriever(
            vector_store_provider=vector_store_provider,
            embedding_provider=embeddings,
            reranker_model=reranker_model,
            collection_name=collection_name,
        )
        self.web_tools = web_tools_manager
        
        self.tools = []

        if self.web_tools.search_provider.is_available():
            self.web_search_tool = get_web_search_tool(self.web_tools)
            self.tools.append(self.web_search_tool)

            self.news_search_tool = get_news_search_tool(self.web_tools)
            self.tools.append(self.news_search_tool)

        


