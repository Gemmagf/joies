"""Segment-aware behaviour: per-segment context the orchestrator appends to the prompt.

The maison's voice is fixed (it's in the system prompt and cached). What changes per
segment is *strategy*: which pieces to surface first, how aggressively to propose an
appointment, when to invoke heritage. Putting this in a `<segment_context>` block
appended to the user message keeps the cached system prompt intact while still
personalising behaviour.
"""

from __future__ import annotations

from ..models import ClientProfile, ClientSegment

SEGMENT_GUIDANCE: dict[ClientSegment, dict[str, str]] = {
    ClientSegment.PROSPECT: {
        "en": (
            "Strategy: educational, warm, low-pressure. Lead with the maison's most accessible "
            "silhouettes (Vintage Alhambra mother-of-pearl, mini Frivole). Offer heritage "
            "context generously. Do not propose a high-jewelry appointment."
        ),
        "fr": (
            "Stratégie : pédagogique, chaleureux, sans pression. Mettez en avant les silhouettes "
            "les plus accessibles (Vintage Alhambra nacre, mini Frivole). Offrez généreusement "
            "le contexte patrimonial. Ne proposez pas de rendez-vous haute joaillerie."
        ),
    },
    ClientSegment.ESTABLISHED: {
        "en": (
            "Strategy: catalog-forward. Propose two to three pieces aligned with collections "
            "this client already owns. Mention heritage only when it adds meaning. A boutique "
            "visit is appropriate when the basket exceeds CHF 10,000."
        ),
        "fr": (
            "Stratégie : centrée sur le catalogue. Proposez deux à trois pièces alignées avec "
            "les collections déjà détenues. Mentionnez le patrimoine si cela enrichit le propos. "
            "Une visite en boutique est appropriée au-delà de CHF 10 000."
        ),
    },
    ClientSegment.VIP: {
        "en": (
            "Strategy: high-touch. Always offer a private viewing (boutique or residence). "
            "Surface pieces of investment grade or rare editions first. Reference the client's "
            "owned collections when proposing complementary pieces."
        ),
        "fr": (
            "Stratégie : haute attention. Proposez systématiquement une présentation privée "
            "(boutique ou domicile). Mettez en avant les pièces d'investissement ou les "
            "éditions rares. Faites référence aux collections détenues."
        ),
    },
    ClientSegment.COLLECTOR: {
        "en": (
            "Strategy: archive and heritage are the lead. Surface high jewelry, archival "
            "Ballerinas, Cadenas watches, rare-stone Alhambra. Heritage citations expected on "
            "every relevant piece. Always offer a private appointment with an archive advisor."
        ),
        "fr": (
            "Stratégie : l'archive et le patrimoine guident l'échange. Mettez en avant la haute "
            "joaillerie, les Ballerines d'archive, les montres Cadenas, l'Alhambra aux pierres "
            "rares. Citations patrimoniales attendues pour chaque pièce. Proposez toujours un "
            "rendez-vous privé avec un conseiller archives."
        ),
    },
}


def segment_context_for_prompt(profile: ClientProfile, *, locale: str = "en") -> str:
    """Compact, model-readable block summarising who the client is and how to treat them."""
    guidance = SEGMENT_GUIDANCE[profile.segment][locale]
    collections = ", ".join(c.value for c in profile.collections_owned) or "none"
    lines = [
        "<segment_context>",
        f"  segment: {profile.segment.value}",
        f"  tenure_years: {profile.tenure_years:.1f}",
        f"  lifetime_spend_chf: {profile.lifetime_spend_chf:,.0f}",
        f"  collections_owned: {collections}",
        f"  preferred_language: {profile.preferred_language}",
        f"  guidance: {guidance}",
        "</segment_context>",
    ]
    return "\n".join(lines)
