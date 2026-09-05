"""Pluggable, bring-your-own-key embedding backends.

Backends are imported lazily so that the base ``dynavec`` install (boto3 + numpy)
never pulls in openai / google-generativeai / sentence-transformers unless you
actually construct that embedder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Embedder, Vector

if TYPE_CHECKING:  # for type checkers / IDEs only
    from .bedrock import BedrockEmbedder
    from .gemini import GeminiEmbedder
    from .openai import OpenAIEmbedder
    from .sentence_transformers import SentenceTransformerEmbedder

__all__ = [
    "Embedder",
    "Vector",
    "OpenAIEmbedder",
    "GeminiEmbedder",
    "BedrockEmbedder",
    "SentenceTransformerEmbedder",
]

_LAZY = {
    "OpenAIEmbedder": ("dynavec.embeddings.openai", "OpenAIEmbedder"),
    "GeminiEmbedder": ("dynavec.embeddings.gemini", "GeminiEmbedder"),
    "BedrockEmbedder": ("dynavec.embeddings.bedrock", "BedrockEmbedder"),
    "SentenceTransformerEmbedder": (
        "dynavec.embeddings.sentence_transformers",
        "SentenceTransformerEmbedder",
    ),
}


def __getattr__(name: str):  # PEP 562 lazy submodule attribute access
    if name in _LAZY:
        import importlib

        module_path, attr = _LAZY[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'dynavec.embeddings' has no attribute {name!r}")
