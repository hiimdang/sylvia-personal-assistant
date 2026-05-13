import logging
import sys

from fastapi import FastAPI, Response, status, HTTPException
from sylvia_core.bootstrap import build_sylvia_rag_core
import time
import asyncio
import os
from sylvia_core.utils.schemas import ChatRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set asyncio event loop policy for Windows compatibility with Playwright
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

logger.info("Initializing chatbot...")
rag_core_instance = build_sylvia_rag_core()
get_ai_response = rag_core_instance.get_response
logger.info("Chatbot initialized.")

_is_rag_core_initialized: bool = False

# Register a function to be called upon startup
@app.on_event("startup")
async def startup_event():
    global _is_rag_core_initialized
    logger.info("Performing RAGCore async initialization...")
    await rag_core_instance.ainit()
    _is_rag_core_initialized = True
    logger.info("RAGCore async initialization complete.")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}

@app.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    return {"status": "live"}

@app.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    if _is_rag_core_initialized:
        return {"status": "ready"}
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAGCore is not initialized yet.")

# Register a function to be called upon shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Performing RAGCore async shutdown...")
    await rag_core_instance.aclose()
    logger.info("RAGCore async shutdown complete.")


@app.post("/chat")
async def chat(request_data: ChatRequest):
    """Endpoint to receive questions and return answers."""

    question = request_data.question
    sender_name = request_data.sender_name
    incoming_chat_history = request_data.chat_history
    image_url = request_data.image_url

    if not question and not image_url:
        logger.warning("Request is missing both question and image")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question or image is required")

    current_chat_history = (
        incoming_chat_history if incoming_chat_history is not None else []
    )

    start_time = time.time()

    try:
        full_response = await get_ai_response(
            sender_name=sender_name,
            question=question,
            chat_history=current_chat_history,
            image_url=image_url,
        )
        return Response(content=full_response, media_type="text/plain; charset=utf-8")
    except Exception:
        logger.exception("A critical error occurred during chat processing")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during processing, please try again later.")
    finally:
        end_time = time.time()
        logger.debug("Request finished in %.2f seconds", end_time - start_time)
