"""Reply composition — assembles retrieved evidence and asks Claude to draft the reply.

The composer is the only node that produces user-facing prose. Other nodes only retrieve
or classify. This keeps the maison's voice consistent — one prompt, one stable cache key.
"""

from __future__ import annotations

from ..config import get_settings
from ..models import Citation
from ..retrieval import CatalogSearchResult, HeritageSearchResult, VisualSearchResult
from .client import cached_system_blocks, get_anthropic_client
from .prompts import orchestrator_system


def _format_catalog_block(results: list[CatalogSearchResult], locale: str) -> str:
    if not results:
        return ""
    lines = ["[Catalog candidates retrieved for this turn]"]
    for r in results:
        name = r.piece.name.fr if locale == "fr" else r.piece.name.en
        lines.append(
            f"- {name} — {r.piece.id} (CHF {r.piece.price_chf:,.0f}) "
            f"[{r.piece.collection.value}] score={r.score:.2f}"
        )
    return "\n".join(lines)


def _format_heritage_block(results: list[HeritageSearchResult], locale: str) -> str:
    if not results:
        return ""
    lines = ["[Heritage citations available]"]
    for r in results:
        title = r.document.title.fr if locale == "fr" else r.document.title.en
        lines.append(f"- [{r.document.id}] {title}: {r.snippet}")
    return "\n".join(lines)


def _format_visual_block(results: list[VisualSearchResult], locale: str) -> str:
    if not results:
        return ""
    header = (
        "[Visually similar pieces (degraded text fallback)]"
        if results[0].degraded
        else "[Visually similar pieces from CLIP search]"
    )
    lines = [header]
    for r in results:
        name = r.piece.name.fr if locale == "fr" else r.piece.name.en
        lines.append(f"- {name} — {r.piece.id} (CHF {r.piece.price_chf:,.0f}) score={r.score:.2f}")
    return "\n".join(lines)


def compose_reply(
    *,
    user_message: str,
    locale: str,
    catalog_hits: list[CatalogSearchResult],
    heritage_hits: list[HeritageSearchResult],
    visual_hits: list[VisualSearchResult],
) -> tuple[str, list[Citation]]:
    settings = get_settings()
    client = get_anthropic_client()

    evidence_parts = [
        block
        for block in (
            _format_catalog_block(catalog_hits, locale),
            _format_heritage_block(heritage_hits, locale),
            _format_visual_block(visual_hits, locale),
        )
        if block
    ]
    evidence = "\n\n".join(evidence_parts) if evidence_parts else "[No retrieval results available.]"

    user_content = (
        f"Client message ({locale}):\n{user_message}\n\n"
        f"Retrieval evidence:\n{evidence}\n\n"
        "Draft your reply now. Cite catalog refs and heritage IDs inline."
    )

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        thinking={"type": "adaptive"},
        system=cached_system_blocks(orchestrator_system(locale)),
        messages=[{"role": "user", "content": user_content}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")

    citations: list[Citation] = []
    for h in heritage_hits:
        if f"[{h.document.id}]" in text:
            citations.append(
                Citation(document_id=h.document.id, snippet=h.snippet, relevance=h.score)
            )
    return text, citations
