import logging
import uuid
from typing import List

import qdrant_client
from fastembed import TextEmbedding
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


def get_qdrant_client(qdrant_url: str, qdrant_api_key: str):
    """Initialize and return a sync Qdrant client."""
    return qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def ensure_collection_exists(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    embedding_dim: int,
):
    """Ensure a Qdrant collection exists and has payload indexes."""
    try:
        client.get_collection(collection_name=collection_name)
        logger.info("Using existing collection: '%s'", collection_name)
    except Exception:
        logger.info("Collection '%s' not found. Creating it...", collection_name)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        logger.info("Collection created.")

        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.question",
                field_schema={
                    "type": "text",
                    "tokenizer": "word",
                    "min_token_len": 2,
                    "max_token_len": 10,
                    "lowercase": True,
                },
                wait=True,
            )
            logger.info("Created payload index for 'metadata.question'.")
        except Exception as e:
            logger.warning("Could not create payload index for 'metadata.question': %s", e)

        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.answer",
                field_schema={
                    "type": "text",
                    "tokenizer": "word",
                    "min_token_len": 2,
                    "max_token_len": 10,
                    "lowercase": True,
                },
                wait=True,
            )
            logger.info("Created payload index for 'metadata.answer'.")
        except Exception as e:
            logger.warning("Could not create payload index for 'metadata.answer': %s", e)


def upsert_documents_to_qdrant(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    documents: List[dict],
    embeddings_model: TextEmbedding,
    batch_size: int = 50,
):
    """Embed and upsert documents to Qdrant in batches."""
    logger.info("Upserting %d documents into collection '%s'...", len(documents), collection_name)
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logger.info("Processing batch %d/%d (%d docs).", batch_num, total_batches, len(batch))

        texts_to_embed = [doc["page_content"] for doc in batch]
        vectors = embeddings_model.embed(texts_to_embed)

        points_to_upsert = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=doc,
            )
            for doc, vector in zip(batch, vectors)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True,
        )

    logger.info("Qdrant upsert completed.")
