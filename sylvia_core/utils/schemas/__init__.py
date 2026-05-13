from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Keyword(BaseModel):
    keyword: str = Field(
        description=(
            "A key keyword or entity extracted from context (question, chat history, or image). "
            "Examples: 'cat', 'Makoto Yuki', 'Aigis'."
        ),
    )


class KeywordsList(BaseModel):
    keywords: List[Keyword] = Field(
        description="List of extracted keywords or key entities."
    )


class ChatHistoryEntry(BaseModel):
    type: str
    sender: str
    content: str
    image_refs: List[str] = []


class ChatRequest(BaseModel):
    question: Optional[str] = None
    sender_name: str = "default_user"
    chat_history: Optional[List[ChatHistoryEntry]] = None
    image_url: Optional[str] = None


class SemanticEntity(BaseModel):
    id: str = Field(description="Unique entity id (for example: e1, e2, e3).")
    type: str = Field(
        description="Free-form semantic type (for example: living_being, object, text_block)."
    )
    label: str = Field(description="Human-readable label.")
    salience: float = Field(
        ge=0.0,
        le=1.0,
        description="Visual salience score in range [0, 1].",
    )


class VisionSemanticMemory(BaseModel):
    global_summary: str = Field(
        description="Overall semantic summary of the image at high level."
    )
    entities: List[SemanticEntity] = Field(
        default_factory=list,
        description="Distinct salient entities identified in the image.",
    )
    attributes: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Open semantic attributes per entity id.",
    )
    perception_notes: List[str] = Field(
        default_factory=list,
        description="Notes about uncertainty, occlusion, or alternate interpretations.",
    )
