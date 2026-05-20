"""Semantic search over heritage stories with citation-ready snippets."""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from ..data_loader import heritage_by_id, load_heritage
from ..models import HeritageDocument
from ._chroma import get_chroma_client

COLLECTION_NAME = "heritage_v1"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(slots=True)
class HeritageSearchResult:
    document: HeritageDocument
    score: float
    snippet: str


class HeritageRAG:
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

    def _doc_text(self, doc: HeritageDocument) -> str:
        return (
            f"{doc.title.en} / {doc.title.fr}. "
            f"{doc.body.en} {doc.body.fr} "
            f"Year: {doc.year or 'n/a'}. Tags: {', '.join(doc.tags)}."
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
        docs = load_heritage()
        ids = [d.id for d in docs]
        bodies = [self._doc_text(d) for d in docs]
        self._collection.add(
            ids=ids,
            documents=bodies,
            metadatas=[
                {
                    "year": d.year or 0,
                    "collection": d.collection.value if d.collection else "",
                }
                for d in docs
            ],
            embeddings=self._encode(bodies),
        )
        return len(ids)

    def search(self, query: str, *, k: int = 3, locale: str = "en") -> list[HeritageSearchResult]:
        result = self._collection.query(
            query_embeddings=self._encode([query]),
            n_results=k,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out: list[HeritageSearchResult] = []
        for doc_id, dist in zip(ids, distances, strict=True):
            doc = heritage_by_id(doc_id)
            if doc is None:
                continue
            body = doc.body.fr if locale == "fr" else doc.body.en
            snippet = body[:280] + ("…" if len(body) > 280 else "")
            out.append(
                HeritageSearchResult(
                    document=doc, score=max(0.0, 1.0 - dist), snippet=snippet
                )
            )
        return out
