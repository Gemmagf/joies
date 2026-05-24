"""Load synthetic client profiles."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..models import ClientProfile


@lru_cache(maxsize=1)
def load_profiles() -> list[ClientProfile]:
    path: Path = get_settings().data_dir / "clients" / "profiles.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ClientProfile.model_validate(item) for item in raw]


def profile_by_id(client_id: str) -> ClientProfile | None:
    for profile in load_profiles():
        if profile.id == client_id:
            return profile
    return None
