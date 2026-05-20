"""Thin wrapper around a persistent Chroma client used by both RAGs.

Chroma is chosen over pgvector for the demo to avoid running Postgres locally; the public
interface (collection.add / collection.query) is similar enough that swapping the backend
later is a localised change.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.api.ClientAPI:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
