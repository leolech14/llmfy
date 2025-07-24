"""
llmfy - Quality-First Document Processing for LLMs

Transform your documents into high-quality, LLM-ready knowledge chunks.
"""

__version__ = "0.1.0"

# Import main entry point for CLI
from .llmfy import main

__all__ = ["main", "__version__"]