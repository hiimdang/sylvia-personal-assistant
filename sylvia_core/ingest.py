import json
import logging

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from .config import settings
from .qdrant_manager import (
    ensure_collection_exists,
    get_qdrant_client,
    upsert_documents_to_qdrant,
)

logger = logging.getLogger(__name__)

SOURCE_FILE = "converted_output.jsonl"
COLLECTION_NAME = settings.COLLECTION_NAME
BATCH_SIZE = settings.BATCH_SIZE
EMBEDDING_DIM = settings.EMBEDDING_DIM
CUSTOM_MODEL_NAME = settings.EMBEDDING_MODEL


def main():
    """Process converted dataset and upsert it into Qdrant."""
    logger.info("Starting ingestion from '%s'.", SOURCE_FILE)

    try:
        logger.info("Connecting to Qdrant...")
        qdrant_cloud_client = get_qdrant_client(settings.QDRANT_URL, settings.QDRANT_API_KEY)
        logger.info("Connected to Qdrant.")

        logger.info("Registering FastEmbed model: %s", CUSTOM_MODEL_NAME)
        TextEmbedding.add_custom_model(
            model=CUSTOM_MODEL_NAME,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=CUSTOM_MODEL_NAME),
            dim=EMBEDDING_DIM,
        )

        logger.info("Initializing FastEmbed model...")
        embeddings_model = TextEmbedding(model_name=CUSTOM_MODEL_NAME)
        logger.info("FastEmbed model initialized.")

    except Exception as e:
        logger.exception("Critical initialization failure: %s", e)
        return

    ensure_collection_exists(qdrant_cloud_client, COLLECTION_NAME, EMBEDDING_DIM)

    all_documents_to_upsert = []

    try:
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Handle back-to-back JSON objects in malformed JSONL sources.
        json_objects_str = "".join(lines).replace("}{", "}\n{").splitlines()

        for i, line in enumerate(json_objects_str):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping line %d due to invalid JSON.", i + 1)
                continue

            question = item.get("question", "")
            answer = item.get("answer", "")
            history = item.get("history", [])
            sender_name = item.get("sender_name", "unknown")
            image_description = item.get("image_description")

            text_parts = []
            if question:
                text_parts.append(f"Question: {question}")
            if answer:
                text_parts.append(f"Answer: {answer}")
            if image_description:
                text_parts.append(f"Image description: {image_description}")
            text_for_embedding = "\n".join(text_parts)

            if not text_for_embedding.strip():
                logger.warning("Skipping line %d because it has no embeddable content.", i + 1)
                continue

            payload = {
                "page_content": text_for_embedding,
                "metadata": {
                    "doc_type": "chat_dialogue",
                    "question": question,
                    "answer": answer,
                    "history": history,
                    "sender_name": sender_name,
                    "image_description": image_description,
                },
            }
            all_documents_to_upsert.append(payload)

        if all_documents_to_upsert:
            logger.info("Upserting %d documents...", len(all_documents_to_upsert))
            upsert_documents_to_qdrant(
                qdrant_cloud_client,
                COLLECTION_NAME,
                all_documents_to_upsert,
                embeddings_model,
                BATCH_SIZE,
            )

        logger.info("Ingestion completed. Total upserted documents: %d", len(all_documents_to_upsert))

    except Exception as e:
        logger.exception("Unexpected ingestion failure: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
