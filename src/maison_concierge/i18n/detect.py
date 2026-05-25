"""Lightweight locale detection for EN/FR conversation routing.

Production would use a proper classifier or the user's profile setting; for a portfolio
demo a small lexicon catches the obvious cases without an extra model call.
"""

from __future__ import annotations

import re
from typing import Literal

Locale = Literal["en", "fr"]

_FR_MARKERS = {
    "bonjour", "merci", "je", "tu", "vous", "nous", "voudrais", "souhaite", "cherche",
    "voir", "bague", "collier", "bracelet", "boucles", "oreilles", "cadeau", "anniversaire",
    "où", "quel", "quelle", "comment", "pourquoi", "est-ce", "très", "magnifique", "histoire",
}
_EN_MARKERS = {
    "hello", "hi", "thanks", "thank", "looking", "want", "would", "like", "ring", "necklace",
    "bracelet", "earrings", "gift", "anniversary", "where", "what", "which", "how", "why",
    "story", "history", "appointment",
}

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ]+")


def detect_locale(text: str, default: Locale = "en") -> Locale:
    tokens = {tok.lower() for tok in _TOKEN.findall(text)}
    if not tokens:
        return default
    fr_score = len(tokens & _FR_MARKERS)
    en_score = len(tokens & _EN_MARKERS)
    if fr_score == en_score == 0:
        return default
    return "fr" if fr_score > en_score else "en"
