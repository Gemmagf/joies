"""Catalog and heritage domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, NonNegativeInt, PositiveFloat


class Collection(StrEnum):
    ALHAMBRA = "alhambra"
    FRIVOLE = "frivole"
    BALLERINAS = "ballerinas"
    LUCKY_ANIMALS = "lucky_animals"
    PERLEE = "perlee"


PieceCategory = Literal["ring", "necklace", "bracelet", "earrings", "pendant", "brooch", "watch"]
Material = Literal["yellow_gold", "white_gold", "rose_gold", "platinum"]


class LocalisedText(BaseModel):
    en: str
    fr: str


class CatalogPiece(BaseModel):
    id: str = Field(pattern=r"^VCA-[A-Z0-9-]+$")
    collection: Collection
    category: PieceCategory
    name: LocalisedText
    description: LocalisedText
    material: Material
    stones: list[str] = Field(default_factory=list)
    price_chf: PositiveFloat
    year_introduced: NonNegativeInt
    image_path: str | None = None
    is_high_jewelry: bool = False
    tags: list[str] = Field(default_factory=list)

    def display_name(self, locale: Literal["en", "fr"] = "en") -> str:
        return self.name.en if locale == "en" else self.name.fr


class HeritageDocument(BaseModel):
    id: str
    title: LocalisedText
    body: LocalisedText
    year: NonNegativeInt | None = None
    collection: Collection | None = None
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    image_url: HttpUrl | None = None
