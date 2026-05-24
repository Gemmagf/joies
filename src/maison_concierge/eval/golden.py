"""Load the hand-curated evaluation set."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import get_settings
from ..models import ClientIntent


class GoldenItem(BaseModel):
    id: str
    text: str
    locale: Literal["en", "fr"]
    expected_intent: ClientIntent
    expected_routes: list[Literal["catalog", "heritage", "visual"]] = Field(default_factory=list)
    expected_piece_ids_top_k: list[str] = Field(default_factory=list)
    expected_heritage_ids: list[str] = Field(default_factory=list)
    expected_escalate: bool = False


@lru_cache(maxsize=1)
def load_golden_set() -> list[GoldenItem]:
    path: Path = get_settings().data_dir / "eval" / "golden_set.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenItem.model_validate(item) for item in raw]
