"""Intent classification using Claude's structured outputs (parse helper)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..config import get_settings
from .client import cached_system_blocks, get_anthropic_client
from .prompts import INTENT_CLASSIFIER_SYSTEM


class IntentResult(BaseModel):
    intent: Literal[
        "browse",
        "gift",
        "investment_piece",
        "celebration",
        "heritage_inquiry",
        "appointment",
        "price_inquiry",
        "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    locale: Literal["en", "fr"]
    needs_visual_search: bool = False
    needs_heritage: bool = False
    needs_catalog: bool = False
    escalate: bool = False


def classify_intent(
    user_message: str, *, default_locale: Literal["en", "fr"] = "en", has_image: bool = False
) -> IntentResult:
    settings = get_settings()
    client = get_anthropic_client()

    user_content = (
        f"Locale hint: {default_locale}. Image attached: {has_image}.\n\n"
        f"Client message:\n{user_message}"
    )

    response = client.messages.parse(
        model=settings.claude_model,
        max_tokens=400,
        system=cached_system_blocks(INTENT_CLASSIFIER_SYSTEM),
        messages=[{"role": "user", "content": user_content}],
        output_format=IntentResult,
    )
    if response.parsed_output is None:
        return IntentResult(
            intent="unknown",
            confidence=0.0,
            locale=default_locale,
            escalate=True,
        )
    return response.parsed_output
