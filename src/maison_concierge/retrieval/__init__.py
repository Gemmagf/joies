"""Retrieval layer: text RAG over catalog and heritage, plus CLIP visual search."""

from .catalog_rag import CatalogRAG, CatalogSearchResult
from .heritage_rag import HeritageRAG, HeritageSearchResult
from .visual_search import VisualSearch, VisualSearchResult

__all__ = [
    "CatalogRAG",
    "CatalogSearchResult",
    "HeritageRAG",
    "HeritageSearchResult",
    "VisualSearch",
    "VisualSearchResult",
]
