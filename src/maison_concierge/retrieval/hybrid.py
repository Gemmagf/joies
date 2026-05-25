"""Hybrid BM25 + dense retrieval for the catalog, fused via Reciprocal Rank Fusion.

Why hybrid here. The catalog is small (60 pieces) but product attributes are highly
token-discriminative — "rose gold Alhambra carnelian" should hit on exact tokens, not
on semantic neighbours. Dense embeddings score "warm carnelian" close to "warm orange",
which is wrong for a precise inventory query. BM25 captures the lexical signal that
dense undersells; dense captures the paraphrase signal that BM25 cannot reach. RRF
(Cormack et al., 2009) combines them without needing comparable score scales.

This module is a drop-in for `CatalogRAG`: same interface, same return type. The eval
runner can score either, so we can produce honest before/after numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from rank_bm25 import BM25Okapi

from ..data_loader import load_catalog, piece_by_id
from ..models import CatalogPiece
from .catalog_rag import CatalogRAG, CatalogSearchResult

RRF_K = 60  # standard constant from Cormack et al., 2009

# Tokenization: lowercase, split on anything that isn't a letter/digit, drop stopwords.
_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
_STOPWORDS = frozenset(
    {
        # English
        "the", "a", "an", "of", "and", "or", "for", "with", "in", "on", "at",
        "is", "are", "be", "to", "from",
        # French
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "pour",
        "avec", "dans", "sur", "à", "au", "aux", "en", "est", "sont",
    }
)


def _tokenize(text: str) -> list[str]:
    return [tok for tok in (m.group(0).lower() for m in _TOKEN.finditer(text)) if tok not in _STOPWORDS]


def _piece_document(piece: CatalogPiece) -> str:
    """Same text representation the dense index uses — keep the indexed surface aligned."""
    return (
        f"{piece.name.en} / {piece.name.fr}. "
        f"Collection: {piece.collection.value}. Category: {piece.category}. "
        f"Material: {piece.material}. Stones: {', '.join(piece.stones) or 'none'}. "
        f"{piece.description.en} {piece.description.fr} "
        f"Tags: {', '.join(piece.tags)}."
    )


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = RRF_K
) -> dict[str, float]:
    """Fuse multiple ranked lists of document IDs via RRF.

    Each input list is ordered most-relevant first. Returns a dict mapping doc_id to
    fused score. Higher is better. Pure function — easy to unit-test.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass(slots=True)
class _BM25Index:
    pieces: list[CatalogPiece]
    bm25: BM25Okapi
    piece_ids: list[str]


@lru_cache(maxsize=1)
def _bm25_index() -> _BM25Index:
    pieces = load_catalog()
    corpus = [_tokenize(_piece_document(p)) for p in pieces]
    return _BM25Index(
        pieces=pieces,
        bm25=BM25Okapi(corpus),
        piece_ids=[p.id for p in pieces],
    )


class HybridCatalogRAG:
    """RRF-fused BM25 + dense retrieval. Same surface as `CatalogRAG`."""

    def __init__(
        self,
        *,
        dense: CatalogRAG | None = None,
        candidate_pool: int = 20,
        rrf_k: int = RRF_K,
    ) -> None:
        self._dense = dense or CatalogRAG()
        self._candidate_pool = candidate_pool
        self._rrf_k = rrf_k
        # Populated on every `search()` call so the orchestrator's trace panel can
        # show which retriever contributed what.
        self.last_diagnostics: dict[str, list[str]] = {
            "bm25_top": [],
            "dense_top": [],
            "fused_top": [],
        }

    def index(self, *, force: bool = False) -> int:
        """Build both indices. BM25 is in-memory and rebuilds on every process start;
        dense is persisted in Chroma.
        """
        if force:
            _bm25_index.cache_clear()
        _ = _bm25_index()
        return self._dense.index(force=force)

    def _bm25_ranking(self, query: str, *, n: int) -> list[str]:
        idx = _bm25_index()
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = idx.bm25.get_scores(tokens)
        ranked = sorted(zip(idx.piece_ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked[:n] if _ > 0.0]

    def _dense_ranking(self, query: str, *, n: int) -> list[str]:
        return [r.piece.id for r in self._dense.search(query, k=n)]

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        category: Literal["ring", "necklace", "bracelet", "earrings", "pendant", "brooch", "watch"] | None = None,
        max_price_chf: float | None = None,
    ) -> list[CatalogSearchResult]:
        bm25_ranking = self._bm25_ranking(query, n=self._candidate_pool)
        dense_ranking = self._dense_ranking(query, n=self._candidate_pool)
        fused_scores = reciprocal_rank_fusion(
            [bm25_ranking, dense_ranking], k=self._rrf_k
        )
        ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
        self.last_diagnostics = {
            "bm25_top": bm25_ranking[:5],
            "dense_top": dense_ranking[:5],
            "fused_top": [pid for pid, _ in ordered[:5]],
        }

        results: list[CatalogSearchResult] = []
        for piece_id, score in ordered:
            piece = piece_by_id(piece_id)
            if piece is None:
                continue
            if category is not None and piece.category != category:
                continue
            if max_price_chf is not None and piece.price_chf > max_price_chf:
                continue
            results.append(CatalogSearchResult(piece=piece, score=score))
            if len(results) >= k:
                break
        return results
