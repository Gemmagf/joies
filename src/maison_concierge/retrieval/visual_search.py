"""CLIP-based visual similarity search over the catalog.

OpenCLIP and torch are heavy dependencies, so the module is gated by the `CLIP_ENABLED`
setting. When disabled, `search()` falls back to a text query over the catalog RAG so the
chat UI can still respond meaningfully to image uploads (with a degraded mode banner).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import get_settings
from ..data_loader import load_catalog, piece_by_id
from ..models import CatalogPiece

if TYPE_CHECKING:
    from PIL import Image as PILImage


@dataclass(slots=True)
class VisualSearchResult:
    piece: CatalogPiece
    score: float
    degraded: bool = False


class VisualSearch:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._catalog_embeddings: dict[str, list[float]] | None = None

    @property
    def enabled(self) -> bool:
        return self._settings.clip_enabled

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._settings.clip_model, pretrained=self._settings.clip_pretrained
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._settings.clip_model)
        self._torch = torch

    def _encode_image(self, image: PILImage.Image) -> list[float]:
        self._load_model()
        assert self._model is not None and self._preprocess is not None
        tensor = self._preprocess(image).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()

    def _encode_text(self, text: str) -> list[float]:
        self._load_model()
        assert self._model is not None and self._tokenizer is not None
        tokens = self._tokenizer([text])
        with self._torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()

    def _build_catalog_index(self) -> None:
        if self._catalog_embeddings is not None:
            return
        index: dict[str, list[float]] = {}
        for piece in load_catalog():
            text = f"{piece.description.en} {', '.join(piece.tags)} {piece.collection.value}"
            index[piece.id] = self._encode_text(text)
        self._catalog_embeddings = index

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    def search_by_image(self, image_bytes: bytes, *, k: int = 4) -> list[VisualSearchResult]:
        if not self.enabled:
            return self._fallback(k)
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        query_embedding = self._encode_image(image)
        self._build_catalog_index()
        assert self._catalog_embeddings is not None
        scored = sorted(
            (
                (piece_id, self._cosine(query_embedding, emb))
                for piece_id, emb in self._catalog_embeddings.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        out: list[VisualSearchResult] = []
        for piece_id, score in scored[:k]:
            piece = piece_by_id(piece_id)
            if piece:
                out.append(VisualSearchResult(piece=piece, score=score))
        return out

    def search_by_image_path(self, path: str | Path, *, k: int = 4) -> list[VisualSearchResult]:
        return self.search_by_image(Path(path).read_bytes(), k=k)

    def _fallback(self, k: int) -> list[VisualSearchResult]:
        pieces = sorted(load_catalog(), key=lambda p: p.price_chf, reverse=False)[:k]
        return [
            VisualSearchResult(piece=piece, score=0.0, degraded=True) for piece in pieces
        ]
