"""Framework integrations (OpenAI Assistants, LangChain, CrewAI, DSPy, LlamaIndex)."""

from .tools import (
    OpenAIAssistantTool,
    as_crewai_tool,
    as_langchain_tool,
    as_openai_tool,
    make_retriever_fn,
)

__all__ = [
    "OpenAIAssistantTool",
    "as_openai_tool",
    "as_langchain_tool",
    "as_crewai_tool",
    "make_retriever_fn",
]
