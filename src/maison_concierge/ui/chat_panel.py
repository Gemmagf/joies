"""Reusable Streamlit chat panel — renders inside whatever container is current.

Used by `apps/chat.py` (standalone) and `apps/dashboard.py` (as the first tab).
Page-level concerns (`st.set_page_config`, global CSS, the app title) stay in the
caller. This module only renders what goes inside the chat container.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import streamlit as st

from ..agents import Orchestrator
from ..config import get_settings
from ..i18n import t
from ..memory import get_store
from ..models import ChatMessage, ConversationState
from ..observability import Event, get_recorder

_SESSION_KEY = "maison_conversation"


@st.cache_resource
def _orchestrator() -> Orchestrator:
    return Orchestrator()


def _ensure_state() -> ConversationState:
    if _SESSION_KEY not in st.session_state:
        conv_id = str(uuid.uuid4())
        state = ConversationState(conversation_id=conv_id)
        st.session_state[_SESSION_KEY] = state
        get_recorder().record(Event(type="conversation_started", conversation_id=conv_id))
    return cast(ConversationState, st.session_state[_SESSION_KEY])


def _render_message(msg: ChatMessage) -> None:
    avatar = "◆" if msg.role == "assistant" else "•"
    with st.chat_message(msg.role, avatar=avatar):
        st.markdown(msg.content)
        if msg.citations:
            with st.expander("Sources" if msg.locale == "en" else "Patrimoine"):
                for c in msg.citations:
                    st.caption(f"[{c.document_id}] - {c.snippet}")


def _render_trace(trace: dict) -> None:
    if not trace:
        return
    intent_info = trace.get("intent", {})
    catalog_info = trace.get("catalog", {})
    heritage_info = trace.get("heritage", {})
    compose_info = trace.get("compose", {})
    groundedness_info = trace.get("groundedness", {})

    with st.expander("Why this reply (debug trace)"):
        if intent_info:
            esc = " · **escalated**" if intent_info.get("escalate") else ""
            st.markdown(
                f"**Intent** · `{intent_info.get('detected', '?')}` "
                f"· confidence `{intent_info.get('confidence', 0):.2f}` "
                f"· locale `{intent_info.get('locale', '?')}` "
                f"· mode `{intent_info.get('mode', '?')}`{esc}"
            )
        if catalog_info and not catalog_info.get("skipped"):
            st.markdown(f"**Catalog retrieval** · {catalog_info.get('retriever', '?')}")
            cols = st.columns(3)
            cols[0].caption("BM25 top-5")
            for pid in catalog_info.get("bm25_top", []):
                cols[0].markdown(f"- `{pid}`")
            cols[1].caption("Dense top-5")
            for pid in catalog_info.get("dense_top", []):
                cols[1].markdown(f"- `{pid}`")
            cols[2].caption("RRF-fused top-5")
            for pid in catalog_info.get("fused_top", []):
                cols[2].markdown(f"- `{pid}`")
            st.caption(f"latency: {catalog_info.get('latency_ms', 0)} ms")
        if heritage_info and not heritage_info.get("skipped"):
            st.markdown(
                f"**Heritage retrieval** · {heritage_info.get('retriever', '?')} · "
                f"{heritage_info.get('n_hits', 0)} hits · "
                f"{heritage_info.get('latency_ms', 0)} ms"
            )
            if heritage_info.get("top_ids"):
                st.code("\n".join(heritage_info["top_ids"]), language="text")
        if compose_info:
            st.markdown(
                f"**Composer** · {compose_info.get('mode', '?')} "
                f"· template intent `{compose_info.get('intent', '?')}` "
                f"· {compose_info.get('reply_length_chars', 0)} chars "
                f"· {compose_info.get('n_citations', 0)} citations "
                f"· {compose_info.get('latency_ms', 0)} ms"
            )
        if groundedness_info:
            is_grounded = groundedness_info.get("is_grounded", True)
            badge = "PASS" if is_grounded else f"FAIL ({groundedness_info.get('hallucination_count', 0)})"
            findings = groundedness_info.get("findings_by_kind", {})
            findings_str = f" · findings: {findings}" if findings else ""
            st.markdown(f"**Groundedness** · {badge}{findings_str}")


def render_chat_panel(*, with_header: bool = True) -> None:
    """Render the full chat experience inside the current container.

    Args:
        with_header: include the title + tagline. Standalone chat sets True;
                     when embedded in the dashboard tab the dashboard already has
                     its own title, so pass False to avoid double headings.
    """
    settings = get_settings()
    state = _ensure_state()
    locale = state.locale

    if with_header:
        st.title(t("app.title", locale))
        st.caption(t("app.tagline", locale))

    if _orchestrator().demo_mode:
        if locale == "en":
            st.info(
                "**Demo mode** - replies are produced by a rule-based intent classifier and "
                "templated EN/FR composer over the real retrieval layer (BM25 + dense + RRF). "
                "Add `ANTHROPIC_API_KEY` to `.env` to switch to live Claude Sonnet 4.6 generation.  \n"
                "*In production this would step up to Claude Opus 4.7 on AWS Bedrock / Vertex AI with "
                "EU data residency, fine-tuned vision on the maison's archive, and broader language support. "
                "See the README - Scaling beyond the demo.*",
                icon="◆",
            )
        else:
            st.info(
                "**Mode démo** - les réponses sont produites par un classificateur d'intention à règles "
                "et un composeur EN/FR templaté, au-dessus de la couche de recherche réelle "
                "(BM25 + dense + RRF). Ajoutez `ANTHROPIC_API_KEY` à `.env` pour activer Claude Sonnet 4.6.  \n"
                "*En production, le système basculerait sur Claude Opus 4.7 hébergé sur AWS Bedrock / Vertex AI, "
                "avec résidence des données en Europe et support de langues étendu. Voir le README.*",
                icon="◆",
            )

    controls_left, controls_right = st.columns([1, 2])
    with controls_left:
        new_locale = st.selectbox(
            "Language" if locale == "en" else "Langue",
            options=["en", "fr"],
            index=0 if locale == "en" else 1,
            format_func=lambda x: "English" if x == "en" else "Français",
            key="chat_locale_select",
        )
        if new_locale != locale:
            state.locale = new_locale  # type: ignore[assignment]
    with controls_right:
        uploaded = st.file_uploader(
            t("chat.upload_image", state.locale),
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            key="chat_image_upload",
        )

    st.caption(
        f"Conversation `{state.conversation_id[:8]}…` · Model: `{settings.claude_model}` · "
        f"{'demo' if _orchestrator().demo_mode else 'claude'} mode"
    )

    for message in state.messages:
        _render_message(message)

    user_input = st.chat_input(t("chat.placeholder", state.locale))
    if not user_input:
        return

    image_bytes = uploaded.getvalue() if uploaded is not None else None

    user_msg = ChatMessage(role="user", content=user_input, locale=state.locale)
    state.messages.append(user_msg)
    _render_message(user_msg)

    with st.chat_message("assistant", avatar="◆"):
        with st.spinner("…"):
            result = _orchestrator().run(
                conversation_id=state.conversation_id,
                user_message=user_input,
                locale=state.locale,
                image_bytes=image_bytes,
            )
        reply_text = result.get("assistant_reply", "")
        citations = result.get("citations", [])
        st.markdown(reply_text)

        catalog_hits = result.get("catalog_hits") or []
        if catalog_hits:
            with st.expander("Catalog matches"):
                for hit in catalog_hits:
                    name = hit.piece.display_name(state.locale)
                    st.markdown(
                        f"**{name}** - `{hit.piece.id}` · CHF {hit.piece.price_chf:,.0f} · "
                        f"score {hit.score:.2f}"
                    )

        visual_hits = result.get("visual_hits") or []
        if visual_hits:
            label = "Visual matches" if not visual_hits[0].degraded else "Catalog (text fallback)"
            with st.expander(label):
                for hit in visual_hits:
                    st.markdown(
                        f"**{hit.piece.display_name(state.locale)}** - `{hit.piece.id}` · "
                        f"CHF {hit.piece.price_chf:,.0f}"
                    )

        _render_trace(result.get("trace") or {})

    assistant_msg = ChatMessage(
        role="assistant",
        content=reply_text,
        locale=state.locale,
        citations=list(citations),
        timestamp=datetime.now(UTC),
    )
    state.messages.append(assistant_msg)

    if result.get("escalate"):
        state.escalated = True
        state.escalation_reason = result.get("escalation_reason")

    get_store().save(state)
