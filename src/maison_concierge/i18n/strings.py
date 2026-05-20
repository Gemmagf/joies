"""UI and system-prompt string tables.

The catalog and heritage data also carry per-locale fields. These strings cover the
chat UI, dashboard chrome, and a few standardised assistant utterances that should
not be regenerated from the model on every turn.
"""

from __future__ import annotations

from typing import Literal

Locale = Literal["en", "fr"]

STRINGS: dict[str, dict[Locale, str]] = {
    "app.title": {
        "en": "Maison Concierge",
        "fr": "Maison Concierge",
    },
    "app.tagline": {
        "en": "Your private advisor — discover pieces, heritage and appointments.",
        "fr": "Votre conseiller privé — découvrez pièces, patrimoine et rendez-vous.",
    },
    "chat.placeholder": {
        "en": "Tell me what you are looking for…",
        "fr": "Dites-moi ce que vous recherchez…",
    },
    "chat.upload_image": {
        "en": "Upload an inspiration image",
        "fr": "Téléchargez une image d'inspiration",
    },
    "chat.escalation": {
        "en": "I'd like to introduce you to one of our advisors who can help further. May I arrange a private appointment?",
        "fr": "Je souhaite vous mettre en contact avec l'un de nos conseillers. Puis-je organiser un rendez-vous privé ?",
    },
    "appointment.confirmed": {
        "en": "Your appointment is reserved. A confirmation has been sent to your advisor.",
        "fr": "Votre rendez-vous est réservé. Une confirmation a été envoyée à votre conseiller.",
    },
    "dashboard.title": {
        "en": "Stakeholder dashboard",
        "fr": "Tableau de bord",
    },
    "dashboard.quality": {
        "en": "Conversation quality",
        "fr": "Qualité des conversations",
    },
    "dashboard.intents": {
        "en": "Top client intents",
        "fr": "Intentions clients principales",
    },
    "dashboard.escalations": {
        "en": "Escalations to human advisor",
        "fr": "Escalades vers un conseiller",
    },
}


def t(key: str, locale: Locale = "en") -> str:
    bundle = STRINGS.get(key)
    if not bundle:
        return key
    return bundle.get(locale) or bundle.get("en") or key
