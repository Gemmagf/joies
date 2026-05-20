"""Semantic search over the catalog (text-only — visual search lives in visual_search.py).

Embeddings: sentence-transformers `all-MiniLM-L6-v2` by default — small, multilingual-ish,
and bundled lazily so unit tests can stub it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sentence_transformers import SentenceTransformer

from ..data_loader import load_catalog, piece_by_id
from ..models import CatalogPiece
from ._chroma import get_chroma_client

COLLECTION_NAME = "catalog_v1"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(slots=True)
class CatalogSearchResult:
    piece: CatalogPiece
    score: float


class CatalogRAG:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL) -> None:
        self._client = get_chroma_client()
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._encoder: SentenceTransformer | None = None
        self._embedding_model = embedding_model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._embedding_model)
        return self._encoder.encode(texts, normalize_embeddings=True).tolist()

    def _piece_document(self, piece: CatalogPiece) -> str:
        return (
            f"{piece.name.en} / {piece.name.fr}. "
            f"Collection: {piece.collection.value}. Category: {piece.category}. "
            f"Material: {piece.material}. Stones: {', '.join(piece.stones) or 'none'}. "
            f"{piece.description.en} {piece.description.fr} "
            f"Tags: {', '.join(piece.tags)}."
        )

    def index(self, *, force: bool = False) -> int:
        existing = self._collection.count()
        if existing > 0 and not force:
            return existing
        if force and existing > 0:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        pieces = load_catalog()
        ids = [p.id for p in pieces]
        docs = [self._piece_document(p) for p in pieces]
        metadatas = [
            {
                "collection": p.collection.value,
                "category": p.category,
                "material": p.material,
                "price_chf": p.price_chf,
                "is_high_jewelry": p.is_high_jewelry,
            }
            for p in pieces
        ]
        self._collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=self._encode(docs),
        )
        return len(ids)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        category: Literal["ring", "necklace", "bracelet", "earrings", "pendant", "brooch", "watch"] | None = None,
        max_price_chf: float | None = None,
    ) -> list[CatalogSearchResult]:
        where: dict[str, object] = {}
        if category:
            where["category"] = category
        if max_price_chf is not None:
            where["price_chf"] = {"$lte": max_price_chf}

        result = self._collection.query(
            query_embeddings=self._encode([query]),
            n_results=k,
            where=where or None,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out: list[CatalogSearchResult] = []
        for piece_id, dist in zip(ids, distances, strict=True):
            piece = piece_by_id(piece_id)
            if piece is None:
                continue
            out.append(CatalogSearchResult(piece=piece, score=max(0.0, 1.0 - dist)))
        return out
