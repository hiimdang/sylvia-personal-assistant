from sylvia_core.clients import SylviaClientManager, get_search_provider
from sylvia_core.component_manager import SylviaComponentManager
from sylvia_core.utils.web.web_tools_manager import WebToolsManager
from sylvia_core.rag_core.memory_manager import MemoryManager
from sylvia_core.rag_core.agent_builder import AgentBuilder
from sylvia_core.rag_core.rag_core import SylviaRAGCore
from sylvia_core.config import settings


def build_sylvia_rag_core() -> SylviaRAGCore:
    client_manager = SylviaClientManager()

    search_provider = get_search_provider()
    web_tools_manager = WebToolsManager(search_provider=search_provider)

    component_manager = SylviaComponentManager(
        embeddings=client_manager.embeddings,
        reranker_model=client_manager.reranker_model,
        vector_store_provider=client_manager.vector_store_provider,
        structured_keyword_llm=client_manager.structured_keyword_llm,
        structured_image_keyword_llm=client_manager.structured_image_keyword_llm,
        structured_image_description_llm=client_manager.structured_image_description_llm,
        collection_name=settings.COLLECTION_NAME,
        web_tools_manager=web_tools_manager,
    )

    memory_manager = MemoryManager(main_llm=client_manager.main_llm)

    rag_chain = AgentBuilder(
        main_llm=client_manager.main_llm,
        tools=component_manager.tools,
    ).get_rag_chain()

    return SylviaRAGCore(
        client_manager=client_manager,
        component_manager=component_manager,
        memory_manager=memory_manager,
        rag_chain=rag_chain,
    )
