"""Pluggable, bring-your-own-key embedding backends.

Backends are imported lazily so that the base ``dynavec`` install (boto3 + numpy)
never pulls in openai / google-generativeai / sentence-transformers unless you
actually construct that embedder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Embedder, Vector

if TYPE_CHECKING:  # for type checkers / IDEs only
    from .bedrock import BedrockEmbedder, BedrockTitanMultimodalEmbedder
    from .gemini import GeminiEmbedder
    from .mistral import MistralEmbedder
    from .ollama import OllamaEmbedder
    from .openai import OpenAIEmbedder
    from .sentence_transformers import SentenceTransformerEmbedder
    from .voyage import VoyageEmbedder
    from .huggingface import HFInferenceEmbedder

__all__ = [
    "Embedder",
    "Vector",
    "OpenAIEmbedder",
    "GeminiEmbedder",
    "BedrockEmbedder",
    "BedrockTitanMultimodalEmbedder",
    "SentenceTransformerEmbedder",
    "VoyageEmbedder",
    "MistralEmbedder",
    "OllamaEmbedder",
    "HFInferenceEmbedder",
]

_LAZY = {
    "OpenAIEmbedder": ("dynavec.embeddings.openai", "OpenAIEmbedder"),
    "GeminiEmbedder": ("dynavec.embeddings.gemini", "GeminiEmbedder"),
    "BedrockEmbedder": ("dynavec.embeddings.bedrock", "BedrockEmbedder"),
    "BedrockTitanMultimodalEmbedder": (
        "dynavec.embeddings.bedrock",
        "BedrockTitanMultimodalEmbedder",
    ),
    "SentenceTransformerEmbedder": (
        "dynavec.embeddings.sentence_transformers",
        "SentenceTransformerEmbedder",
    ),
    "VoyageEmbedder": ("dynavec.embeddings.voyage", "VoyageEmbedder"),
    "MistralEmbedder": ("dynavec.embeddings.mistral", "MistralEmbedder"),
    "OllamaEmbedder": ("dynavec.embeddings.ollama", "OllamaEmbedder"),
    "HFInferenceEmbedder": ("dynavec.embeddings.huggingface", "HFInferenceEmbedder"),
}


def __getattr__(name: str):  # PEP 562 lazy submodule attribute access
    if name in _LAZY:
        import importlib

        module_path, attr = _LAZY[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'dynavec.embeddings' has no attribute {name!r}")
