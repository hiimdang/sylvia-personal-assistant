from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .web_tools_manager import WebToolsManager


class WebSearchInput(BaseModel):
    q: str = Field(description="The search query.")
    count: int = Field(default=20, ge=5, description="The number of results to return (minimum 5).")
    offset: int = Field(default=0, description="The offset for pagination.")
    freshness: Optional[str] = Field(
        default=None,
        description="Optional recency filter, provider-specific.",
    )


class NewsSearchInput(BaseModel):
    q: str = Field(description="The search query.")
    count: int = Field(default=20, ge=5, description="The number of results to return (minimum 5).")
    offset: int = Field(default=0, description="The offset for pagination.")
    freshness: Optional[str] = Field(
        default=None,
        description="Optional recency filter, provider-specific.",
    )


def get_web_search_tool(web_tools_manager_instance: WebToolsManager) -> StructuredTool:
    return StructuredTool.from_function(
        func=web_tools_manager_instance.web_search,
        name="web_search",
        description=(
            "Performs a web search using the configured search provider. "
            "Use when the user asks for current or external information."
        ),
        args_schema=WebSearchInput,
    )


def get_news_search_tool(web_tools_manager_instance: WebToolsManager) -> StructuredTool:
    return StructuredTool.from_function(
        func=web_tools_manager_instance.news_search,
        name="news_search",
        description=(
            "Performs a news search using the configured search provider. "
            "Use for recent news or current events."
        ),
        args_schema=NewsSearchInput,
    )
