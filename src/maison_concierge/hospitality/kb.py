"""Property Knowledge Base — RAG over the hand-written markdown pages.

Each page in `data/property_kb/` has a YAML-style frontmatter with an `id`
(the stable citation key), a `title`, `tags`, and a `property` scope
(`lisboa`, `algarve`, or `both`). Chunking is per-page: pages are short and
each answers one thing, so a whole-page chunk keeps citations coherent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..retrieval._chroma import get_chroma_client

COLLECTION_NAME = "property_kb_v1"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_FRONTMATTER_RE = re.compile(r"^---\n(?P<meta>.*?)\n---\n(?P<body>.*)$", re.DOTALL)


@dataclass(slots=True, frozen=True)
class KBPage:
    id: str
    title: str
    tags: list[str]
    property_scope: str  # lisboa | algarve | both
    body: str            # markdown body (frontmatter stripped)
    path: Path

    def snippet(self, max_chars: int = 320) -> str:
        cleaned = re.sub(r"\s+", " ", self.body).strip()
        return cleaned[:max_chars] + ("…" if len(cleaned) > max_chars else "")


@dataclass(slots=True, frozen=True)
class KBSearchResult:
    page: KBPage
    score: float
    snippet: str


def _kb_dir() -> Path:
    return get_settings().data_dir / "property_kb"


def _parse_page(path: Path) -> KBPage:
    raw = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(f"KB page {path.name} missing YAML frontmatter")
    meta_lines = m.group("meta").splitlines()
    body = m.group("body").strip()
    meta: dict[str, object] = {}
    for line in meta_lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value
    return KBPage(
        id=str(meta.get("id", path.stem)),
        title=str(meta.get("title", path.stem)),
        tags=list(meta.get("tags", []) or []),
        property_scope=str(meta.get("property", "both")),
        body=body,
        path=path,
    )


@lru_cache(maxsize=1)
def load_kb_pages() -> list[KBPage]:
    directory = _kb_dir()
    if not directory.exists():
        return []
    return [_parse_page(p) for p in sorted(directory.glob("*.md"))]


def page_by_id(kb_id: str) -> KBPage | None:
    for p in load_kb_pages():
        if p.id == kb_id:
            return p
    return None


class PropertyKB:
    """Chroma-backed dense retrieval over the property KB."""

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
        pages = load_kb_pages()
        if not pages:
            return 0
        ids = [p.id for p in pages]
        docs = [f"{p.title}\n{p.body}" for p in pages]
        self._collection.add(
            ids=ids,
            documents=docs,
            metadatas=[
                {
                    "title": p.title,
                    "property": p.property_scope,
                    "tags": ",".join(p.tags),
                }
                for p in pages
            ],
            embeddings=self._encode(docs),
        )
        return len(ids)

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        property_scope: str | None = None,
    ) -> list[KBSearchResult]:
        self.index()
        result = self._collection.query(
            query_embeddings=self._encode([query]),
            n_results=k,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out: list[KBSearchResult] = []
        for kb_id, dist in zip(ids, distances, strict=True):
            page = page_by_id(kb_id)
            if page is None:
                continue
            if (
                property_scope is not None
                and page.property_scope not in {property_scope, "both"}
            ):
                continue
            out.append(
                KBSearchResult(
                    page=page,
                    score=max(0.0, 1.0 - dist),
                    snippet=page.snippet(),
                )
            )
        return out
