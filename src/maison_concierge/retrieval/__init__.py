"""Retrieval layer: dense + BM25 + CLIP visual search.

The catalog has two interchangeable backends — `CatalogRAG` (dense only) and
`HybridCatalogRAG` (BM25 + dense fused via RRF). The eval framework runs both so the
quality delta is measurable.
"""

from .catalog_rag import CatalogRAG, CatalogSearchResult
from .heritage_rag import HeritageRAG, HeritageSearchResult
from .hybrid import HybridCatalogRAG, reciprocal_rank_fusion
from .visual_search import VisualSearch, VisualSearchResult

__all__ = [
    "CatalogRAG",
    "CatalogSearchResult",
    "HeritageRAG",
    "HeritageSearchResult",
    "HybridCatalogRAG",
    "VisualSearch",
    "VisualSearchResult",
    "reciprocal_rank_fusion",
]
