"""Load synthetic catalog and heritage data from the data/ directory."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import get_settings
from .models import CatalogPiece, HeritageDocument


def _data_root() -> Path:
    return get_settings().data_dir


@lru_cache(maxsize=1)
def load_catalog() -> list[CatalogPiece]:
    path = _data_root() / "catalog" / "pieces.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [CatalogPiece.model_validate(piece) for piece in raw]


@lru_cache(maxsize=1)
def load_heritage() -> list[HeritageDocument]:
    path = _data_root() / "heritage" / "stories.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [HeritageDocument.model_validate(doc) for doc in raw]


def piece_by_id(piece_id: str) -> CatalogPiece | None:
    for piece in load_catalog():
        if piece.id == piece_id:
            return piece
    return None


def heritage_by_id(doc_id: str) -> HeritageDocument | None:
    for doc in load_heritage():
        if doc.id == doc_id:
            return doc
    return None
