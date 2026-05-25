"""Templated EN/FR composer — deterministic, no API key needed.

Produces replies in the maison's voice based on retrieved evidence. Honest framing:
templated replies are *less* expressive than Claude's, but they preserve the
demonstration of every other architectural feature (retrieval, citations,
escalation, bilingual routing). Citations are produced by the same path the
Claude composer uses, so groundedness checks still apply.
"""

from __future__ import annotations

from ..models import Citation, ClientIntent
from ..retrieval import CatalogSearchResult, HeritageSearchResult, VisualSearchResult


def _format_piece(piece, locale: str) -> str:
    name = piece.name.fr if locale == "fr" else piece.name.en
    return f"the {name} — {piece.id} (CHF {piece.price_chf:,.0f})" if locale == "en" else (
        f"le {name} — {piece.id} (CHF {piece.price_chf:,.0f})"
    )


_GREETING_EN = "Welcome to the maison. "
_GREETING_FR = "Bienvenue à la maison. "


def _gift_or_celebration(
    intent: ClientIntent, locale: str, hits: list[CatalogSearchResult]
) -> str:
    top = hits[:2]
    if not top:
        return (
            "I would honour your request. May I arrange a presentation in our Geneva boutique?"
            if locale == "en"
            else "Je serai honoré de vous accompagner. Puis-je organiser une présentation en boutique à Genève ?"
        )
    if locale == "en":
        opening = "An anniversary calls for something quietly memorable. " if intent is ClientIntent.CELEBRATION else "Two refined options for you. "
        if len(top) == 1:
            body = f"I would suggest {_format_piece(top[0].piece, locale)} — a luminous choice."
        else:
            body = (
                f"I would suggest {_format_piece(top[0].piece, locale)} as the founding silhouette, "
                f"or {_format_piece(top[1].piece, locale)} as a warmer counterpoint."
            )
        closing = " Would you like me to arrange for them to be presented privately?"
        return opening + body + closing
    opening = "Un anniversaire appelle quelque chose de discrètement mémorable. " if intent is ClientIntent.CELEBRATION else "Deux lectures raffinées vous attendent. "
    if len(top) == 1:
        body = f"Je vous propose {_format_piece(top[0].piece, locale)} — une signature lumineuse."
    else:
        body = (
            f"Je vous propose {_format_piece(top[0].piece, locale)} comme silhouette fondatrice, "
            f"ou {_format_piece(top[1].piece, locale)} pour une lecture plus chaleureuse."
        )
    closing = " Souhaitez-vous que je les fasse préparer pour une présentation privée ?"
    return opening + body + closing


def _heritage(locale: str, hits: list[HeritageSearchResult]) -> tuple[str, list[Citation]]:
    if not hits:
        return (
            (
                "The archives are vast. May I introduce you to one of our advisors so we can address your inquiry precisely?"
                if locale == "en"
                else "Les archives sont vastes. Puis-je vous mettre en contact avec un conseiller pour mieux répondre à votre demande ?"
            ),
            [],
        )
    top = hits[0]
    snippet = top.snippet
    if locale == "en":
        body = f"{snippet} [{top.document.id}]"
        closing = " Would you like me to bring you the pieces that descend from that lineage?"
    else:
        body = f"{snippet} [{top.document.id}]"
        closing = " Souhaitez-vous découvrir les pièces qui s'en inspirent ?"
    citations = [
        Citation(document_id=top.document.id, snippet=top.snippet, relevance=top.score)
    ]
    return body + closing, citations


def _price(locale: str, hits: list[CatalogSearchResult]) -> str:
    if not hits:
        return (
            "I'd rather give you exact pricing in front of the piece. May I arrange a private viewing?"
            if locale == "en"
            else "Je préfère vous donner le prix exact devant la pièce. Puis-je organiser une présentation privée ?"
        )
    lines = [
        (
            "Within that range, three readings stand out:" if locale == "en"
            else "Dans cette gamme, trois lectures se distinguent :"
        )
    ]
    for hit in hits[:3]:
        lines.append(f"  • {_format_piece(hit.piece, locale)}")
    closing = (
        " Would you like me to expand on any of them?"
        if locale == "en"
        else " Souhaitez-vous que je vous en dise davantage sur l'une d'elles ?"
    )
    return "\n".join(lines) + closing


def _appointment(locale: str, hits: list[CatalogSearchResult]) -> str:
    if locale == "en":
        opening = "With pleasure. Camille Laurent welcomes clients at our Geneva atelier on Rue du Rhône."
        if hits:
            preview = f" If you have a preference, the {hits[0].piece.name.en} ({hits[0].piece.id}) can be prepared in advance."
        else:
            preview = ""
        return opening + preview + " Which day next week would suit you?"
    opening = "Avec plaisir. Camille Laurent reçoit les clients à notre atelier genevois, rue du Rhône."
    if hits:
        preview = f" Si vous le souhaitez, le {hits[0].piece.name.fr} ({hits[0].piece.id}) peut être préparé d'avance."
    else:
        preview = ""
    return opening + preview + " Quel jour de la semaine prochaine vous conviendrait ?"


def _investment(locale: str, hits: list[CatalogSearchResult]) -> str:
    if locale == "en":
        opening = "Investment pieces ask for personal attention. "
        if hits:
            opening += (
                f"From our archive, {_format_piece(hits[0].piece, locale)} is a strong candidate"
                f"{' — alongside ' + _format_piece(hits[1].piece, locale) if len(hits) > 1 else ''}."
            )
        return opening + " Shall I arrange a private appointment with our archive advisor?"
    opening = "Les pièces d'investissement appellent une attention personnelle. "
    if hits:
        opening += (
            f"Parmi nos archives, {_format_piece(hits[0].piece, locale)} est un fort candidat"
            f"{' — aux côtés de ' + _format_piece(hits[1].piece, locale) if len(hits) > 1 else ''}."
        )
    return opening + " Puis-je organiser un rendez-vous privé avec notre conseiller archives ?"


def _browse_or_unknown(locale: str, hits: list[CatalogSearchResult]) -> str:
    if not hits:
        if locale == "en":
            return "Tell me a little more about the occasion and I will draw a selection from our collections."
        return "Dites-m'en un peu plus sur l'occasion et je vous tirerai une sélection de nos collections."
    return _gift_or_celebration(ClientIntent.BROWSE, locale, hits)


def compose_reply_templated(
    *,
    intent: ClientIntent,
    locale: str,
    catalog_hits: list[CatalogSearchResult],
    heritage_hits: list[HeritageSearchResult],
    visual_hits: list[VisualSearchResult],
) -> tuple[str, list[Citation]]:
    if intent is ClientIntent.HERITAGE_INQUIRY:
        return _heritage(locale, heritage_hits)
    if intent in (ClientIntent.GIFT, ClientIntent.CELEBRATION):
        return _gift_or_celebration(intent, locale, catalog_hits), []
    if intent is ClientIntent.PRICE_INQUIRY:
        return _price(locale, catalog_hits), []
    if intent is ClientIntent.APPOINTMENT:
        return _appointment(locale, catalog_hits), []
    if intent is ClientIntent.INVESTMENT_PIECE:
        return _investment(locale, catalog_hits), []
    if visual_hits and locale == "en":
        top = visual_hits[0]
        return (
            f"Drawing from your inspiration, {_format_piece(top.piece, locale)} reads as the closest match in our catalog.",
            [],
        )
    return _browse_or_unknown(locale, catalog_hits), []
