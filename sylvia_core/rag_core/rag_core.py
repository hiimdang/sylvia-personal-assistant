import asyncio
import time
import threading
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from sylvia_core.utils.prompts import prompt_manager
from ..component_manager import SylviaComponentManager

from .agent_builder import AgentBuilder
from .retrieval_pipeline import RetrievalPipeline
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class SylviaRAGCore:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        client_manager,
        component_manager: SylviaComponentManager,
        memory_manager: MemoryManager,
        rag_chain,
    ):

        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True

        self._async_init_resources = set()
        self._async_close_resources = set()
        self._async_lock: asyncio.Lock | None = None

        self.client_manager = client_manager
        self.llm_backend = client_manager.llm_backend
        self.main_llm = client_manager.main_llm
        self.embeddings = client_manager.embeddings
        self.vector_store_provider = client_manager.vector_store_provider
        self.reranker_model = client_manager.reranker_model

        if self.vector_store_provider:
            self._async_close_resources.add(self.vector_store_provider)

        self.component_manager = component_manager
        self.web_tools = component_manager.web_tools

        if self.web_tools:
            self._async_init_resources.add(self.web_tools)
            self._async_close_resources.add(self.web_tools)

        self.retrieval_pipeline = RetrievalPipeline(
            keyword_extractor=component_manager.keyword_extractor,
            image_keyword_extractor=component_manager.image_keyword_extractor,
            image_description_extractor=component_manager.image_description_extractor,
            chat_retriever=component_manager.chat_retriever,
        )

        self.memory_manager = memory_manager

        self.rag_chain = rag_chain

        self.prompt_manager = prompt_manager

    # =====================================================
    # MAIN RESPONSE
    # =====================================================

    async def get_response(
        self,
        sender_name: str,
        question: str,
        chat_history: list[dict] | None = None,
        image_url: str | None = None,
    ) -> str:

        current_session_memory = self.memory_manager.get_memory(sender_name)

        logger.debug("Starting request processing.")

        t_total_start = time.time()

        effective_question = question
        image_description = None

        # ------------------------
        # SEARCH
        # ------------------------

        if image_url:

            logger.debug("Image input detected; running retrieval and vision summary in parallel.")

            retrieved_docs, vision_memory = await asyncio.gather(
                self.retrieval_pipeline.search(question, chat_history),
                self.retrieval_pipeline.get_image_summary(image_url),
            )

            image_description = vision_memory
            effective_question = (
                f"{question}\n\n{vision_memory}" if question else vision_memory
            )

        else:

            retrieved_docs = await self.retrieval_pipeline.search(
                question,
                chat_history,
            )

        # ------------------------
        # FORMAT CONTEXT
        # ------------------------

        formatted_context = self.retrieval_pipeline.format_retrieved_context(retrieved_docs)

        # ------------------------
        # BUILD MESSAGES
        # ------------------------

        messages = []

        if formatted_context.strip():
            retrieved_context_policy = self.prompt_manager.get("retrieved_context_policy").strip()
            if not retrieved_context_policy:
                retrieved_context_policy = (
                    "The following context is retrieved reference material, NOT chat history.\n"
                    "Use it to improve relevance and preserve response quality.\n"
                )
            messages.append(
                SystemMessage(
                    content=(
                        f"{retrieved_context_policy}\n\n"
                        + formatted_context
                    )
                )
            )

        current_memory_vars = current_session_memory.load_memory_variables({})
        history = current_memory_vars.get("chat_history", [])

        messages.extend(history)
        messages.append(HumanMessage(content=effective_question))

        # ------------------------
        # CALL AGENT
        # ------------------------

        full_response = ""

        try:

            result = await self.rag_chain.ainvoke({"messages": messages})
            full_response = result["messages"][-1].content

            current_session_memory.save_context(
                {"input": effective_question},
                {"output": full_response},
            )

        except Exception:

            logger.exception("Agent error during processing")
            full_response = "Sorry, an error occurred during processing."

        # ------------------------
        # LOG
        # ------------------------

        await self.memory_manager.log_interaction(
            question=question,
            answer=full_response,
            sender_name=sender_name,
            chat_history=chat_history,
            image_description=image_description,
        )

        logger.debug("Request processing completed in %.3fs.", time.time() - t_total_start)

        return full_response

    # =====================================================
    # ASYNC INIT
    # =====================================================

    async def ainit(self):

        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:

            if getattr(self, "_async_initialized", False):
                return

            logger.info("Initializing async resources...")

            for r in self._async_init_resources:
                if hasattr(r, "ainit"):
                    await r.ainit()
                elif hasattr(r, "__aenter__"):
                    await r.__aenter__()

            if self.embeddings:
                await asyncio.to_thread(self.embeddings.get_embedding_model)

            self._async_initialized = True

    # =====================================================
    # ASYNC CLOSE
    # =====================================================

    async def aclose(self):

        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:

            if getattr(self, "_async_closed", False):
                return

            logger.info("Closing async resources...")

            for r in self._async_close_resources:
                if hasattr(r, "aclose"):
                    await r.aclose()
                elif hasattr(r, "__aexit__"):
                    await r.__aexit__(None, None, None)

            self._async_closed = True

