"""Shared Anthropic client factory.

We use the official Anthropic SDK (Sonnet 4.6 per the project brief). Prompt caching
is applied to the system prompt — it's the largest stable block and is reused on every
turn within a conversation.
"""

from __future__ import annotations

from functools import lru_cache

import anthropic

from ..config import get_settings


@lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key or "missing-key")


def cached_system_blocks(text: str) -> list[dict[str, object]]:
    """Wrap a system prompt in a single cache-controlled block."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
