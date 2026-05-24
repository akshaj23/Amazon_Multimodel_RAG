"""
Models module for CLIP embeddings and LLM integration
"""

from .clip_embeddings import CLIPEmbeddingModel
from .llm_integration import LLMInterface

__all__ = ["CLIPEmbeddingModel", "LLMInterface"]
